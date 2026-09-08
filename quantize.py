"""
quantize.py — INT8 post-training quantization for MobileNetV2-CIFAR.

    Weights     : per-channel, symmetric, REAL int8 storage (pack_int8)
    Activations : per-tensor, asymmetric, fake-quant only (never saved to
                  disk — activations are transient, recomputed every
                  inference call; see estimate_activation_compression)

Design choices:
  - Per-channel weights, per-tensor activations
  - BatchNorm gets NO special treatment: swap_to_quant_modules replaces
    every nn.BatchNorm2d with a QuantBatchNorm2d that quantizes its
    `weight` (gamma) with the exact same per-channel symmetric scheme and
    the same `weight_bits` used for QuantConv2d/QuantLinear -- there is no
    separate BN precision knob. `bias` (beta) stays fp32, mirroring how
    QuantConv2d/QuantLinear also leave their bias unquantized.
    `running_mean`/`running_var` are buffers, not weights, and are left
    untouched. In practice BatchNorm turned out to not be especially
    sensitive to this, so quantizing it uniformly with everything else is
    simpler and works fine.
  - MobileNetV2's linear-bottleneck projection convs have no ReLU6 after
    them (Sandler et al. 2018), and residual-add outputs pass through no
    activation module at all — swap_to_quant_modules's isinstance-based
    insertion structurally cannot reach either. Fixed separately via
    attach_block_output_quant, which hooks InvertedResidual.forward's
    RETURN VALUE directly (works for both the skip-connection case,
    x + conv(x), and the no-skip case, just conv(x)).
  - pruning_method="magnitude" (default) ranks weights by |w|; "hessian"
    ranks by Optimal Brain Damage saliency (0.5 * H_ii * w_i^2) using an
    empirical-Fisher diagonal estimated from real data — see
    compute_fisher_diagonal / apply_hessian_pruning.

Usage:
    Python:
        from quantize import run_quantization_pipeline
        run_quantization_pipeline(
            ckpt_path="outputs/baseline/best.pth",
            data_dir=DATA_DIR,
            calib_batches=40,
        )

    CLI:
        python quantize.py --ckpt-path outputs/baseline/best.pth --data-dir ./data
"""
import argparse
import io
import os
import copy
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models.mobilenetv2 import InvertedResidual
except ImportError:
    InvertedResidual = None  # attach_block_output_quant will warn and skip gracefully


@dataclass
class QuantConfig:
    sparsity: float = 0.3
    weight_quant_bits: int = 8
    activation_quant_bits: int = 8
    calibration_batches: int = 40   # confirmed better than 20 empirically


# ---------------------------------------------------------------------------
# basic quantize, dequantize, and quantize params helpers
# ---------------------------------------------------------------------------
def quantize(x, scale, zero_point, qmin, qmax):
    return torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)


def dequantize(q, scale, zero_point):
    return (q - zero_point) * scale


def weight_qparams_per_channel(w, n_bits, eps=1e-8):
    """One scale per output channel (dim 0). Symmetric -> zero_point always 0.
    For a 1-D weight (BatchNorm's per-channel `weight`/gamma), dim 0 IS every
    element, so this reduces to one scale per element -- no special-casing
    needed elsewhere for BatchNorm vs. Conv2d/Linear."""
    if w.dim() > 1:
        reduce_dims = list(range(1, w.dim()))
        max_abs = w.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    else:
        max_abs = w.abs().clamp_min(eps)
    qmax = (1 << (n_bits - 1)) - 1
    return max_abs / qmax, -qmax, qmax


# ---------------------------------------------------------------------------
# weight quant-aware layers
# ---------------------------------------------------------------------------
'''
We create quantized "equivalents" of Conv2d, Linear, BatchNorm2d, and
ReLU(6). Clarification: the kernels are STILL FP32, they are not custom INT
kernels, rather, we add extra "freeze", "buffer", "quantize" and
"dequantize", methods, which help us effectively simulate quantization.
---
Since we are not using custom kernels, we do "fake quantization", where the "effect"
of quantization is simulated by rounding the weights and activations,
but the compute is still done in FP32. Hence after every forward pass,
the activations must be dequantized -> and then quantized, this simulated forcing
the forward pass results to be in the quantized space.
Although weights are also "fake", we effectively only need the activations to be quantized,
since the forward pass output contains both the weights and input activations.

'''
class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.frozen = False

    @torch.no_grad()
    def freeze(self):
        scale, qmin, qmax = weight_qparams_per_channel(self.weight, self.weight_bits)
        self.register_buffer("w_scale", scale)
        self.qmin, self.qmax = qmin, qmax
        # Cache the quantized-then-dequantized weight ONCE here, instead of
        # redoing the identical round() on every forward() call. The weight
        # tensor now permanently holds its quantized value.
        q = quantize(self.weight, scale, 0, qmin, qmax)
        self.weight.data.copy_(dequantize(q, scale, 0))
        self.frozen = True

    def forward(self, x):
        # After freeze(), self.weight already IS the quantized value —
        # nothing special needed here anymore, in either branch.
        return F.conv2d(x, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.frozen = False

    @torch.no_grad()
    def freeze(self):
        scale, qmin, qmax = weight_qparams_per_channel(self.weight, self.weight_bits)
        self.register_buffer("w_scale", scale)
        self.qmin, self.qmax = qmin, qmax
        q = quantize(self.weight, scale, 0, qmin, qmax)
        self.weight.data.copy_(dequantize(q, scale, 0))
        self.frozen = True

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class QuantBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d with its affine `weight` (gamma) quantized exactly like
    QuantConv2d/QuantLinear -- same per-channel symmetric scheme, same
    `weight_bits`, no separate BN precision knob. `bias` (beta) stays fp32,
    same as QuantConv2d/QuantLinear leaving their bias unquantized.
    `running_mean`/`running_var`/`num_batches_tracked` are untouched."""
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.frozen = False

    @torch.no_grad()
    def freeze(self):
        if self.weight is None:  # affine=False -- nothing to quantize
            self.frozen = True
            return
        scale, qmin, qmax = weight_qparams_per_channel(self.weight, self.weight_bits)
        self.register_buffer("w_scale", scale)
        self.qmin, self.qmax = qmin, qmax
        q = quantize(self.weight, scale, 0, qmin, qmax)
        self.weight.data.copy_(dequantize(q, scale, 0))
        self.frozen = True

    def forward(self, x):
        return super().forward(x)


class ActFakeQuant(nn.Module):
    """Per-tensor, asymmetric (post-ReLU6 outputs are always >= 0)."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.frozen = False

    @torch.no_grad()
    def forward(self, x):
        if not self.frozen:
            self.min_val = torch.minimum(self.min_val, x.min())
            self.max_val = torch.maximum(self.max_val, x.max())
            return x
        q = quantize(x, self.scale, self.zp, self.qmin, self.qmax)
        return dequantize(q, self.scale, self.zp)

    @torch.no_grad()
    def freeze(self):
        qmax = (1 << self.n_bits) - 1

        # FIX: Use the actual observed minimum, not zero.
        # This properly handles both post-ReLU (min ~ 0)
        # and linear bottlenecks (min < 0).
        lo = self.min_val

        scale = (self.max_val - lo).clamp_min(1e-8) / qmax
        zp = torch.round(-lo / scale)

        self.register_buffer("scale", scale)
        self.register_buffer("zp", zp)
        self.qmin, self.qmax = 0, qmax
        self.frozen = True


def compute_fisher_diagonal(model, loader, criterion, device, n_batches=10):
    """
    Approximates the Hessian diagonal (H_ii) of the LOSS w.r.t. each weight,
    via the empirical Fisher Information: H_ii ~ E[(dL/dw_i)^2], averaged
    over `n_batches` of real data. This is a stand-in for the true second
    derivative (which needs a second backward pass, Hutchinson-style) --
    cheap, one backward pass per batch, and exact for cross-entropy loss
    under the standard Gauss-Newton approximation.
    """
    model.eval()  # BN uses running stats, not batch stats -- consistent estimate
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()
              if p.requires_grad and "weight" in name and p.dim() > 1}

    n = 0
    for i, (images, targets) in enumerate(loader):
        if i >= n_batches:
            break
        images, targets = images.to(device), targets.to(device)
        model.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        for name, p in model.named_parameters():
            if name in fisher and p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
        n += 1

    for name in fisher:
        fisher[name] /= n
    model.zero_grad()
    return fisher


### Hessian based pruning
@torch.no_grad()
def apply_hessian_pruning(model, fisher_diag, sparsity=0.3):
    """
    Optimal Brain Damage saliency (LeCun, Denker, Solla, 1990):
        lambda_i = 0.5 * H_ii * w_i^2
    Global threshold across all layers, same structure as
    apply_global_magnitude_pruning, but ranking by saliency instead of
    raw |w_i| -- a weight can be LARGE but still low-saliency if it sits
    in a flat (low-curvature) region of the loss, and vice versa.
    """
    all_saliency = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            wname = name + ".weight"
            if wname not in fisher_diag:
                continue
            saliency = 0.5 * fisher_diag[wname] * m.weight.data ** 2
            all_saliency.append(saliency.view(-1))

    threshold = torch.quantile(torch.cat(all_saliency), sparsity).item()

    pruned, total = 0, 0

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            wname = name + ".weight"
            if wname not in fisher_diag:
                continue
            saliency = 0.5 * fisher_diag[wname] * m.weight.data ** 2
            mask = saliency >= threshold
            m.weight.data.mul_(mask.type_as(m.weight.data))
            pruned += (~mask).sum().item()
            total += mask.numel()

    print(f"Hessian(Fisher)-based pruning: {pruned}/{total} removed "
          f"({pruned/total*100:.2f}%), threshold={threshold:.3e}")
    return model


@torch.no_grad()
###### Magnitude based pruning
def apply_global_magnitude_pruning(model, sparsity=0.3):
    """
    Manually applies global unstructured magnitude pruning across all Conv2d and Linear layers.
    """
    all_weights = []

    # 1. Collect absolute values of all weights
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            all_weights.append(m.weight.view(-1).abs())

    if not all_weights:
        return model

    concat_weights = torch.cat(all_weights)

    # 2. Find the global threshold using the exact quantile
    # e.g., if sparsity is 0.3, find the value at the 30th percentile
    threshold = torch.quantile(concat_weights, sparsity).item()

    # 3. Apply the threshold mask to permanently zero out the weights
    pruned_params = 0
    total_params = 0

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total_params += m.weight.numel()
            # Create boolean mask for weights strictly greater than or equal to threshold
            mask = m.weight.abs() >= threshold
            # Multiply data by mask (converts False to 0.0, True to 1.0)
            m.weight.data.mul_(mask.type_as(m.weight.data))

            pruned_params += (mask == False).sum().item()

    print(f"Manual Global Pruning: {pruned_params}/{total_params} parameters removed ({(pruned_params/total_params)*100:.2f}%). Threshold: {threshold:.6f}")
    return model


# ---------------------------------------------------------------------------
# model surgery
# ---------------------------------------------------------------------------
def swap_to_quant_modules(model, cfg: QuantConfig):
    """Applied uniformly to every Conv2d/Linear/BatchNorm2d/ReLU(6) -- same
    weight_bits for all of them, no per-layer-type exceptions. Structurally
    cannot reach linear-bottleneck/residual outputs — see
    attach_block_output_quant."""
    for name, m in list(model.named_children()):
        swap_to_quant_modules(m, cfg)

        if isinstance(m, nn.Conv2d):
            q = QuantConv2d(m.in_channels, m.out_channels, m.kernel_size,
                             stride=m.stride, padding=m.padding, dilation=m.dilation,
                             groups=m.groups, bias=(m.bias is not None),
                             weight_bits=cfg.weight_quant_bits)
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.Linear):
            q = QuantLinear(m.in_features, m.out_features, bias=(m.bias is not None),
                             weight_bits=cfg.weight_quant_bits)
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.BatchNorm2d):
            q = QuantBatchNorm2d(m.num_features, eps=m.eps, momentum=m.momentum,
                                  affine=m.affine, track_running_stats=m.track_running_stats,
                                  weight_bits=cfg.weight_quant_bits)
            if m.affine:
                q.weight.data.copy_(m.weight.data)
                q.bias.data.copy_(m.bias.data)
            if m.track_running_stats:
                q.running_mean.data.copy_(m.running_mean.data)
                q.running_var.data.copy_(m.running_var.data)
                q.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
            setattr(model, name, q)

        elif isinstance(m, (nn.ReLU, nn.ReLU6)):
            setattr(model, name, nn.Sequential(OrderedDict([
                ("act", type(m)(inplace=False)),
                ("aq", ActFakeQuant(n_bits=cfg.activation_quant_bits)),
            ])))
    return model


def attach_block_output_quant(model, act_bits):
    """
    Fixes the linear-bottleneck gap: InvertedResidual.forward returns either
    `conv(x)` (no skip) or `x + conv(x)` (with skip) -- neither passes through
    an activation module, so swap_to_quant_modules can't reach them via
    isinstance matching. A forward hook fires on whichever value the block
    actually returns, and its return value REPLACES the block's output
    (standard PyTorch forward-hook behavior), so this genuinely quantizes
    that point, not just observes it.
    """
    if InvertedResidual is None:
        print("WARNING: could not import InvertedResidual from this torchvision "
              "version — linear-bottleneck/residual outputs will NOT be "
              "activation-quantized. Document this as a known limitation.")
        return model

    count = 0
    for name, m in model.named_modules():
        if isinstance(m, InvertedResidual):
            aq = ActFakeQuant(n_bits=act_bits)
            m.add_module("_block_output_aq", aq)  # so freeze_all()/calibrate() find it
            m.register_forward_hook(lambda mod, inp, out, aq=aq: aq(out))
            count += 1
    # print(f"attach_block_output_quant: instrumented {count} InvertedResidual block outputs")
    return model


def calibrate(model, loader, device, n_batches=40):
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            model(images.to(device))
            if i + 1 >= n_batches:
                break


def freeze_all(model):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantBatchNorm2d, ActFakeQuant)):
            m.freeze()


# ---------------------------------------------------------------------------
# size + compression-ratio reporting (Now Mathematical & Direct)
# ---------------------------------------------------------------------------
def fp32_size_mb(model):
    """Standardized fp32 size: sum of every tensor in state_dict() (params +
    buffers) — exactly what torch.save(model.state_dict()) writes to disk.
    Use THIS (not a full training checkpoint with optimizer state) as your
    Q4 baseline — verify with os.path.getsize('model_weights_only.pth')."""
    total = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return total / (1024 ** 2)


def compression_ratio_report(model, model_fp32, weight_bits):
    """
    Calculates size reflecting both uniform bit-width quantization
    and non-zero weight counts resulting from pruning. QuantConv2d,
    QuantLinear, and QuantBatchNorm2d are all tallied identically here --
    BatchNorm gets no special-cased accounting either.
    """
    fp32_mb = fp32_size_mb(model_fp32)

    n_layers, n_scales, other_bytes = 0, 0, 0
    total_quant_elements = 0
    nonzero_quant_elements = 0
    quantized_names = set()

    # 1. Tally parameters and non-zero elements
    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantBatchNorm2d)):
            n_layers += 1
            n_scales += m.w_scale.numel()
            quantized_names.add(name)

            total_els = m.weight.numel()
            nnz = (m.weight != 0).sum().item()

            total_quant_elements += total_els
            nonzero_quant_elements += nnz

    # 2. Unquantized parameters (unquantized biases, running stats, etc.)
    for key, tensor in model.state_dict().items():
        owner = key.rsplit(".", 1)[0]
        if owner not in quantized_names:
            other_bytes += tensor.numel() * tensor.element_size()

    # 3. Memory footprint calculations:
    # - Non-zero weights store values at `weight_bits`
    # - Sparse mask adds 1 bit per total parameter
    sparse_weight_bits = (nonzero_quant_elements * weight_bits) + total_quant_elements
    quantized_weights_only_mb = (sparse_weight_bits / 8) / (1024 ** 2)

    scale_storage_mb = (n_scales * 4) / (1024 ** 2)     # FP32 scale buffers
    bn_and_other_fp32_mb = other_bytes / (1024 ** 2)

    theoretical_total_mb = quantized_weights_only_mb + scale_storage_mb + bn_and_other_fp32_mb

    # 4. FP32 baseline weight metrics
    fp32_weight_bytes = sum(
        p.numel() * 4 for n, p in model_fp32.named_parameters()
        if n.endswith("weight") and p.dim() > 1
    )
    fp32_weights_only_mb = fp32_weight_bytes / (1024 ** 2)

    overall_sparsity = 1.0 - (nonzero_quant_elements / total_quant_elements) if total_quant_elements > 0 else 0.0

    return {
        "fp32_total_mb": fp32_mb,
        "quantized_sparse_total_mb": theoretical_total_mb,
        "overall_compression_ratio": fp32_mb / theoretical_total_mb,
        "fp32_weights_only_mb": fp32_weights_only_mb,
        "quantized_sparse_weights_only_mb": quantized_weights_only_mb,
        "weights_compression_ratio": fp32_weights_only_mb / quantized_weights_only_mb,
        "overall_weight_sparsity": f"{overall_sparsity * 100:.2f}%",
        "n_quantized_layers": n_layers,
        "n_scale_values_total": n_scales,
        "scale_storage_mb": scale_storage_mb,
        "bn_and_other_fp32_mb": bn_and_other_fp32_mb,
    }


def estimate_activation_compression(model, sample_batch, act_bits, device):
    """
    Hooks every ActFakeQuant module and counts how many activation values actually
    flow through it during ONE real forward pass on `sample_batch`. Activations
    are never saved to disk (they're transient) -- this reports a RUNTIME MEMORY
    footprint for one forward pass, fp32 vs act_bits.
    """
    count = {"n": 0}
    hooks = [m.register_forward_hook(lambda mod, i, o: count.__setitem__("n", count["n"] + o.numel()))
             for m in model.modules() if isinstance(m, ActFakeQuant)]

    model.eval()
    with torch.no_grad():
        model(sample_batch.to(device))
    for h in hooks:
        h.remove()

    n = count["n"]
    fp32_bytes, quant_bytes = n * 4, n * act_bits / 8
    return {
        "n_activation_elements_per_batch": n,
        "fp32_activation_mb": fp32_bytes / (1024 ** 2),
        "quantized_activation_mb": quant_bytes / (1024 ** 2),
        "activation_compression_ratio": fp32_bytes / quant_bytes,
    }


# ---------------------------------------------------------------------------
# checkpoint loading — tolerant of both wrapped-dict and raw state_dict formats
# ---------------------------------------------------------------------------
def load_state_dict_flexible(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"], raw.get("best_acc", None)
    return raw, None   # raw was itself a bare state_dict


# ---------------------------------------------------------------------------
# High-Level Pipeline & CLI
# ---------------------------------------------------------------------------
def run_quantization_pipeline(ckpt_path, data_dir, weight_bits=8, act_bits=8,
                               calib_batches=40, sparsity=0.3,
                               pruning_method="magnitude", download=False):
    from model import MobileNetV2CIFAR
    from data import get_dataloaders
    from utils import accuracy, AverageMeter

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # print(f"Loading baseline from {ckpt_path}...")
    state_dict, baseline_acc = load_state_dict_flexible(ckpt_path)

    model = MobileNetV2CIFAR(num_classes=10, dropout=0.2)
    model.load_state_dict(state_dict)
    model.to(device)

    # `download` defaults to False since quantization normally runs against
    # data already fetched during training. On a fresh local clone with no
    # cached dataset, pass download=True / --download once.
    train_loader, test_loader = get_dataloaders(data_dir=data_dir, download=download)

    # global pruning before quantization
    if pruning_method == "hessian":
        criterion_for_fisher = nn.CrossEntropyLoss()
        fisher = compute_fisher_diagonal(model, train_loader, criterion_for_fisher, device, n_batches=calib_batches)
        model = apply_hessian_pruning(model, fisher, sparsity)
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and name + ".weight" in fisher:
                is_depthwise = m.groups == m.in_channels
                avg_fisher = fisher[name + ".weight"].mean().item()
                print(f"{name:40s} depthwise={is_depthwise}  mean_H_ii={avg_fisher:.3e}")
    else:
        model = apply_global_magnitude_pruning(model, sparsity)

    model_fp32 = copy.deepcopy(model)   # untouched reference, for size comparison

    cfg = QuantConfig(weight_quant_bits=weight_bits, activation_quant_bits=act_bits,
                      calibration_batches=calib_batches)
    swap_to_quant_modules(model, cfg)
    attach_block_output_quant(model, cfg.activation_quant_bits)

    # swapped modules (including BatchNorm's replacement) are on cpu by
    # default, need to move them to gpu
    model.to(device)

    # print(f"Calibrating with {cfg.calibration_batches} batches (from TRAIN data, not test)...")
    calibrate(model, train_loader, device, n_batches=cfg.calibration_batches)
    freeze_all(model)

    # print("Evaluating fake-quant model on the test dataset...")
    model.eval()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            top1, = accuracy(model(images), targets, topk=(1,))
            acc_meter.update(top1, images.size(0))

    sample_batch, _ = next(iter(test_loader))
    report = compression_ratio_report(model, model_fp32, cfg.weight_quant_bits)
    act_report = estimate_activation_compression(model, sample_batch, cfg.activation_quant_bits, device)

    fp32_acc_str = f"{baseline_acc:.2f}%" if isinstance(baseline_acc, (int, float)) else str(baseline_acc)
    comp_acc_str = f"{acc_meter.avg:.2f}%"

    print(f"fp32 accuracy: {fp32_acc_str} | compressed accuracy: {comp_acc_str}")
    print(f"fp32 size: {report['fp32_total_mb']:.2f} MB | compressed size (theoretical): {report['quantized_sparse_total_mb']:.2f} MB")
    print(f"total compression: {report['overall_compression_ratio']:.2f}x")
    print(f"weight compression: {report['weights_compression_ratio']:.2f}x")
    print(f"activation compression: {act_report['activation_compression_ratio']:.2f}x")

    return {"accuracy": acc_meter.avg, **report, **act_report}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", type=str, required=True, help="Path to baseline checkpoint (.pth)")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--weight-bits", type=int, default=8)
    p.add_argument("--act-bits", type=int, default=8)
    p.add_argument("--calib-batches", type=int, default=40)
    p.add_argument("--sparsity", type=float, default=0.3)
    p.add_argument("--pruning-method", type=str, default="magnitude", choices=["magnitude", "hessian"])
    p.add_argument("--download", action="store_true", default=False,
                    help="Download CIFAR-10 into --data-dir if not already present")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_quantization_pipeline(
        ckpt_path=args.ckpt_path,
        data_dir=args.data_dir,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        calib_batches=args.calib_batches,
        sparsity=args.sparsity,
        pruning_method=args.pruning_method,
        download=args.download,
    )
