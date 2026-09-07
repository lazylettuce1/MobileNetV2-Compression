"""
quantize.py — INT8 post-training quantization for MobileNetV2-CIFAR.

    Weights     : per-channel, symmetric, REAL int8 storage (pack_int8)
    Activations : per-tensor, asymmetric, fake-quant only (never saved to
                  disk — activations are transient, recomputed every
                  inference call; see estimate_activation_compression)

Design choices:
  - Per-channel weights, per-tensor activations
  - BatchNorm layers are replaced with quantized-counterparts,
    since it has few parameters, and is not a bottle neck.
  - MobileNetV2's linear-bottleneck projection convs have no ReLU6 after
    them (Sandler et al. 2018), and residual-add outputs pass through no
    activation module at all — swap_to_quant_modules's isinstance-based
    insertion structurally cannot reach either. Fixed separately via
    attach_block_output_quant, which hooks InvertedResidual.forward's
    RETURN VALUE directly (works for both the skip-connection case,
    x + conv(x), and the no-skip case, just conv(x)).

Usage:
    from quantize import run_quantization_pipeline
    run_quantization_pipeline(
        ckpt_path="/kaggle/working/MobileNetV2-Compression/outputs/baseline/best.pth",
        data_dir=DATA_DIR,
        calib_batches=40,
    )
"""
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
    """One scale per output channel (dim 0). Symmetric -> zero_point always 0."""
    reduce_dims = list(range(1, w.dim()))
    max_abs = w.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    qmax = (1 << (n_bits - 1)) - 1
    return max_abs / qmax, -qmax, qmax


# ---------------------------------------------------------------------------
# weight quant-aware layers
# ---------------------------------------------------------------------------
'''
We create quantized "equivalents" of Conv2d, Linear, and ReLU(6)
clarification: the kernels are STILL FP32, they are not custom INT kernels,
rather, we add extra "freeze", "buffer", "quantize" and "dequantize", methods,
which help us effectively simulate quantization.
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

@torch.no_grad()
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
    """Applied uniformly to every Conv2d/Linear/ReLU(6). Structurally cannot
    reach linear-bottleneck/residual outputs — see attach_block_output_quant."""
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
        if isinstance(m, (QuantConv2d, QuantLinear, ActFakeQuant)):
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
    and non-zero weight counts resulting from pruning.
    """
    fp32_mb = fp32_size_mb(model_fp32)
    
    n_layers, n_scales, other_bytes = 0, 0, 0
    total_quant_elements = 0
    nonzero_quant_elements = 0
    quantized_names = set()
    
    # 1. Tally parameters and non-zero elements
    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            n_layers += 1
            n_scales += m.w_scale.numel()
            quantized_names.add(name)
            
            total_els = m.weight.numel()
            nnz = (m.weight != 0).sum().item()
            
            total_quant_elements += total_els
            nonzero_quant_elements += nnz

    # 2. Unquantized parameters (BatchNorms, unquantized biases, etc.)
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
                               calib_batches=40, sparsity=0.3):
    from model import MobileNetV2CIFAR
    from data import get_dataloaders
    from utils import accuracy, AverageMeter

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # print(f"Loading baseline from {ckpt_path}...")
    state_dict, baseline_acc = load_state_dict_flexible(ckpt_path)

    model = MobileNetV2CIFAR(num_classes=10, dropout=0.2)
    model.load_state_dict(state_dict)
    # global pruning before quantization
    model = apply_global_magnitude_pruning(model, sparsity)
    model_fp32 = copy.deepcopy(model)   # untouched reference, for size comparison

    train_loader, test_loader = get_dataloaders(data_dir=data_dir, download=False)

    cfg = QuantConfig(weight_quant_bits=weight_bits, activation_quant_bits=act_bits,
                       calibration_batches=calib_batches)
    swap_to_quant_modules(model, cfg)
    attach_block_output_quant(model, cfg.activation_quant_bits)
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

