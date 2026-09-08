import argparse
import io
import math
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
    bias_quant_bits: int = 8
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
    For a 1-D weight/bias (BatchNorm's gamma/beta), dim 0 IS every element,
    so this reduces to one scale per element."""
    if w.dim() > 1:
        reduce_dims = list(range(1, w.dim()))
        max_abs = w.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    else:
        max_abs = w.abs().clamp_min(eps)
    qmax = (1 << (n_bits - 1)) - 1
    return max_abs / qmax, -qmax, qmax


# ---------------------------------------------------------------------------
# weight & bias quant-aware layers
# ---------------------------------------------------------------------------
def derive_bias_qparams(w_scale, act_min, act_max, act_bits, bias, bias_bits, eps=1e-8):
    """
    bias_scale = w_scale * act_scale -- the accumulator (weight_q @ input_q)
    naturally lives in units of w_scale*act_scale, so that's the scale bias
    has to be expressed in to add in correctly. No independent bias scale is
    stored: w_scale already exists (per output channel), and act_scale here
    comes from a tiny built-in observer on this layer's own input, so the
    only extra state needed downstream is act_scale itself (one scalar per
    layer), not one scale per bias element.

    Since this scale is DERIVED rather than fit to the bias values, `bias_bits`
    now controls dynamic range (clipping), not rounding precision -- the
    derived scale is already far finer than a value-fitted one, so if
    anything the risk is picking too few bits and clipping outlier biases,
    not coarse rounding. We warn (not silently clamp-and-move-on) if that
    happens.
    """
    act_qmax = (1 << act_bits) - 1
    act_scale = (act_max - act_min).clamp_min(eps) / act_qmax
    bias_scale = w_scale.view(-1) * act_scale
    bias_qmax = (1 << (bias_bits - 1)) - 1

    needed = (bias.abs() / bias_scale.clamp_min(1e-20)).max().item()
    if needed > bias_qmax:
        min_bits = math.ceil(math.log2(needed + 1)) + 2
        print(f"WARNING: bias would clip at bias_bits={bias_bits} "
              f"(derived scale needs >= {min_bits} bits) -- clamping, expect accuracy loss")

    return bias_scale, act_scale, -bias_qmax, bias_qmax


class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, weight_bits=8, bias_bits=16, act_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.bias_bits = bias_bits
        self.act_bits = act_bits
        self.frozen = False
        self.register_buffer("in_min", torch.tensor(float("inf")))
        self.register_buffer("in_max", torch.tensor(float("-inf")))

    @torch.no_grad()
    def freeze(self):
        scale, qmin, qmax = weight_qparams_per_channel(self.weight, self.weight_bits)
        self.register_buffer("w_scale", scale)
        self.qmin, self.qmax = qmin, qmax
        q = quantize(self.weight, scale, 0, qmin, qmax)
        self.weight.data.copy_(dequantize(q, scale, 0))

        if self.bias is not None:
            bias_scale, act_scale, b_qmin, b_qmax = derive_bias_qparams(
                scale, self.in_min, self.in_max, self.act_bits, self.bias, self.bias_bits)
            self.register_buffer("bias_act_scale", act_scale)  # one scalar, for size accounting
            q_b = quantize(self.bias, bias_scale, 0, b_qmin, b_qmax)
            self.bias.data.copy_(dequantize(q_b, bias_scale, 0))

        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            self.in_min = torch.minimum(self.in_min, x.min())
            self.in_max = torch.maximum(self.in_max, x.max())
        return F.conv2d(x, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    def __init__(self, *args, weight_bits=8, bias_bits=16, act_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.bias_bits = bias_bits
        self.act_bits = act_bits
        self.frozen = False
        self.register_buffer("in_min", torch.tensor(float("inf")))
        self.register_buffer("in_max", torch.tensor(float("-inf")))

    @torch.no_grad()
    def freeze(self):
        scale, qmin, qmax = weight_qparams_per_channel(self.weight, self.weight_bits)
        self.register_buffer("w_scale", scale)
        self.qmin, self.qmax = qmin, qmax
        q = quantize(self.weight, scale, 0, qmin, qmax)
        self.weight.data.copy_(dequantize(q, scale, 0))

        if self.bias is not None:
            bias_scale, act_scale, b_qmin, b_qmax = derive_bias_qparams(
                scale, self.in_min, self.in_max, self.act_bits, self.bias, self.bias_bits)
            self.register_buffer("bias_act_scale", act_scale)
            q_b = quantize(self.bias, bias_scale, 0, b_qmin, b_qmax)
            self.bias.data.copy_(dequantize(q_b, bias_scale, 0))

        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            self.in_min = torch.minimum(self.in_min, x.min())
            self.in_max = torch.maximum(self.in_max, x.max())
        return F.linear(x, self.weight, self.bias)


class QuantBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d with both affine weight (gamma) and bias (beta) quantized.
    Left untouched by the bias-scale-derivation trick above: BN's beta isn't
    added into a weight*activation accumulator the way a conv/linear bias is,
    so there's no w_scale*act_scale to derive it from -- it keeps its own
    independent per-element scale, same as before."""
    def __init__(self, *args, weight_bits=8, bias_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.bias_bits = bias_bits
        self.frozen = False

    @torch.no_grad()
    def freeze(self):
        if not self.affine:
            self.frozen = True
            return

        if self.weight is not None:
            scale, qmin, qmax = weight_qparams_per_channel(self.weight, self.weight_bits)
            self.register_buffer("w_scale", scale)
            q = quantize(self.weight, scale, 0, qmin, qmax)
            self.weight.data.copy_(dequantize(q, scale, 0))

        if self.bias is not None:
            b_scale, b_qmin, b_qmax = weight_qparams_per_channel(self.bias, self.bias_bits)
            self.register_buffer("b_scale", b_scale)
            q_b = quantize(self.bias, b_scale, 0, b_qmin, b_qmax)
            self.bias.data.copy_(dequantize(q_b, b_scale, 0))

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
        lo = self.min_val
        scale = (self.max_val - lo).clamp_min(1e-8) / qmax
        zp = torch.round(-lo / scale)

        self.register_buffer("scale", scale)
        self.register_buffer("zp", zp)
        self.qmin, self.qmax = 0, qmax
        self.frozen = True


def compute_fisher_diagonal(model, loader, criterion, device, n_batches=10):
    model.eval()
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


@torch.no_grad()
def apply_hessian_pruning(model, fisher_diag, sparsity=0.3):
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
def apply_global_magnitude_pruning(model, sparsity=0.3):
    all_weights = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            all_weights.append(m.weight.view(-1).abs())

    if not all_weights:
        return model

    concat_weights = torch.cat(all_weights)
    threshold = torch.quantile(concat_weights, sparsity).item()

    pruned_params, total_params = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total_params += m.weight.numel()
            mask = m.weight.abs() >= threshold
            m.weight.data.mul_(mask.type_as(m.weight.data))
            pruned_params += (mask == False).sum().item()

    print(f"Manual Global Pruning: {pruned_params}/{total_params} parameters removed ({(pruned_params/total_params)*100:.2f}%). Threshold: {threshold:.6f}")
    return model


# ---------------------------------------------------------------------------
# model surgery
# ---------------------------------------------------------------------------
def swap_to_quant_modules(model, cfg: QuantConfig):
    for name, m in list(model.named_children()):
        swap_to_quant_modules(m, cfg)

        if isinstance(m, nn.Conv2d):
            q = QuantConv2d(m.in_channels, m.out_channels, m.kernel_size,
                             stride=m.stride, padding=m.padding, dilation=m.dilation,
                             groups=m.groups, bias=(m.bias is not None),
                             weight_bits=cfg.weight_quant_bits, bias_bits=cfg.bias_quant_bits,
                             act_bits=cfg.activation_quant_bits)
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.Linear):
            q = QuantLinear(m.in_features, m.out_features, bias=(m.bias is not None),
                             weight_bits=cfg.weight_quant_bits, bias_bits=cfg.bias_quant_bits,
                             act_bits=cfg.activation_quant_bits)
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.BatchNorm2d):
            q = QuantBatchNorm2d(m.num_features, eps=m.eps, momentum=m.momentum,
                                  affine=m.affine, track_running_stats=m.track_running_stats,
                                  weight_bits=cfg.weight_quant_bits, bias_bits=cfg.bias_quant_bits)
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
    if InvertedResidual is None:
        print("WARNING: could not import InvertedResidual — skipping bottleneck output hook.")
        return model

    for name, m in model.named_modules():
        if isinstance(m, InvertedResidual):
            aq = ActFakeQuant(n_bits=act_bits)
            m.add_module("_block_output_aq", aq)
            m.register_forward_hook(lambda mod, inp, out, aq=aq: aq(out))
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
# size + compression-ratio reporting
# ---------------------------------------------------------------------------
def fp32_size_mb(model):
    total = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return total / (1024 ** 2)


def compression_ratio_report(model, model_fp32, weight_bits, bias_bits=8):
    fp32_mb = fp32_size_mb(model_fp32)

    n_layers, n_scales, other_bytes = 0, 0, 0
    total_quant_elements, nonzero_quant_elements = 0, 0
    total_bias_bits = 0
    quantized_names = set()

    # 1. Tally parameters
    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantBatchNorm2d)):
            n_layers += 1
            if hasattr(m, "w_scale"):
                n_scales += m.w_scale.numel()
            if hasattr(m, "b_scale"):
                n_scales += m.b_scale.numel()  # QuantBatchNorm2d: one scale per element (unchanged)
            if hasattr(m, "bias_act_scale"):
                n_scales += 1  # QuantConv2d/QuantLinear: one shared scalar per layer, not per bias element

            quantized_names.add(name)

            # Weight metrics
            total_els = m.weight.numel()
            nnz = (m.weight != 0).sum().item()
            total_quant_elements += total_els
            nonzero_quant_elements += nnz

            # Bias metrics (quantized)
            if m.bias is not None:
                total_bias_bits += m.bias.numel() * bias_bits

    # 2. Unquantized parameters (running stats, counters, etc.)
    for key, tensor in model.state_dict().items():
        owner = key.rsplit(".", 1)[0]
        if owner not in quantized_names:
            other_bytes += tensor.numel() * tensor.element_size()

    # 3. Memory footprint calculations
    sparse_weight_bits = (nonzero_quant_elements * weight_bits) + total_quant_elements
    quantized_weights_and_biases_mb = ((sparse_weight_bits + total_bias_bits) / 8) / (1024 ** 2)
    quantized_weights_mb = ((sparse_weight_bits + total_bias_bits) / 8) / (1024 ** 2)
    scale_storage_mb = (n_scales * 4) / (1024 ** 2)
    bn_and_other_fp32_mb = other_bytes / (1024 ** 2)

    theoretical_total_mb = quantized_weights_and_biases_mb + scale_storage_mb + bn_and_other_fp32_mb

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
        "quantized_sparse_weights_only_mb": quantized_weights_mb,
        "weights_compression_ratio": fp32_weights_only_mb / quantized_weights_mb,
        "overall_weight_sparsity": f"{overall_sparsity * 100:.2f}%",
        "n_quantized_layers": n_layers,
        "n_scale_values_total": n_scales,
        "scale_storage_mb": scale_storage_mb,
        "bn_and_other_fp32_mb": bn_and_other_fp32_mb,
    }


def estimate_activation_compression(model, sample_batch, act_bits, device):
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


def load_state_dict_flexible(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"], raw.get("best_acc", None)
    return raw, None


# ---------------------------------------------------------------------------
# High-Level Pipeline & CLI
# ---------------------------------------------------------------------------
def run_quantization_pipeline(ckpt_path, data_dir, weight_bits=8, bias_bits=8, act_bits=8,
                               calib_batches=40, sparsity=0.3,
                               pruning_method="magnitude", download=False, seed=42):
    from model import MobileNetV2CIFAR
    from data import get_dataloaders
    from utils import accuracy, AverageMeter, set_seed

    set_seed(seed)  # calibration draws shuffled batches from train_loader -- fix that draw too
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict, baseline_acc = load_state_dict_flexible(ckpt_path)

    model = MobileNetV2CIFAR(num_classes=10, dropout=0.2)
    model.load_state_dict(state_dict)
    model.to(device)

    train_loader, test_loader = get_dataloaders(data_dir=data_dir, download=download)

    if pruning_method == "hessian":
        criterion_for_fisher = nn.CrossEntropyLoss()
        fisher = compute_fisher_diagonal(model, train_loader, criterion_for_fisher, device, n_batches=calib_batches)
        model = apply_hessian_pruning(model, fisher, sparsity)
    else:
        model = apply_global_magnitude_pruning(model, sparsity)

    model_fp32 = copy.deepcopy(model)

    cfg = QuantConfig(weight_quant_bits=weight_bits, bias_quant_bits=bias_bits,
                      activation_quant_bits=act_bits, calibration_batches=calib_batches)
    swap_to_quant_modules(model, cfg)
    attach_block_output_quant(model, cfg.activation_quant_bits)

    model.to(device)

    calibrate(model, train_loader, device, n_batches=cfg.calibration_batches)
    freeze_all(model)

    # Print remaining unquantized parts
    # print_unquantized_parts(model)

    model.eval()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            top1, = accuracy(model(images), targets, topk=(1,))
            acc_meter.update(top1, images.size(0))

    sample_batch, _ = next(iter(test_loader))
    report = compression_ratio_report(model, model_fp32, cfg.weight_quant_bits, cfg.bias_quant_bits)
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
    p.add_argument("--bias-bits", type=int, default=8)
    p.add_argument("--act-bits", type=int, default=8)
    p.add_argument("--calib-batches", type=int, default=40)
    p.add_argument("--sparsity", type=float, default=0.3)
    p.add_argument("--pruning-method", type=str, default="magnitude", choices=["magnitude", "hessian"])
    p.add_argument("--download", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_quantization_pipeline(
        ckpt_path=args.ckpt_path,
        data_dir=args.data_dir,
        weight_bits=args.weight_bits,
        bias_bits=args.bias_bits,
        act_bits=args.act_bits,
        calib_batches=args.calib_batches,
        sparsity=args.sparsity,
        pruning_method=args.pruning_method,
        download=args.download,
        seed=args.seed,
    )
