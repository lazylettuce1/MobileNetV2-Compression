"""
quantize.py — INT8 post-training quantization for MobileNetV2-CIFAR.

    Weights     : per-channel, symmetric, REAL int8 storage (pack_int8)
    Activations : per-tensor, asymmetric, fake-quant only (never saved to
                  disk — activations are transient, recomputed every
                  inference call; see estimate_activation_compression for
                  how their compression is measured instead)

Design choices, per the architecture analysis above:
  - Per-channel weights, per-tensor activations: the standard scheme
    (Krishnamoorthi 2018), chosen because depthwise conv channels in
    MobileNetV2 vary in magnitude enough that per-tensor weight
    quantization measurably hurts accuracy (Sheng et al. 2018).
  - BatchNorm layers are NOT swapped (they're not Conv2d/Linear/ReLU6) —
    kept fp32. This is standard practice: BN params are a tiny fraction
    of total parameters, but directly control activation statistics, so
    quantizing them risks accuracy for negligible size savings.
  - MobileNetV2's linear-bottleneck projection convs have no ReLU6 after
    them (Sandler et al. 2018) — their outputs, and residual-add outputs,
    are NOT activation-quantized, since there's no activation function to
    attach ActFakeQuant to. Documented exception, not an oversight.

Usage:
    cfg = QuantConfig(weight_quant_bits=8, activation_quant_bits=8)
    swap_to_quant_modules(model, cfg)
    calibrate(model, calib_loader, device)
    freeze_all(model)
    # ... run your existing evaluate() here for fake-quant accuracy ...
    packed = pack_int8(model)                    # REAL int8 tensors
    torch.save(packed, "quantized_int8.pth")      # genuinely smaller file
    report = compression_ratio_report(fp32_model, packed)
    act_report = estimate_activation_compression(model, sample_batch, cfg.activation_quant_bits, device)
"""
import io
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantConfig:
    weight_quant_bits: int = 8
    activation_quant_bits: int = 8
    calibration_batches: int = 20


# ---------------------------------------------------------------------------
# core int8 math
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
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        q = quantize(self.weight, self.w_scale, 0, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, 0)
        return F.conv2d(x, w_dq, self.bias, self.stride,
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
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.linear(x, self.weight, self.bias)
        q = quantize(self.weight, self.w_scale, 0, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, 0)
        return F.linear(x, w_dq, self.bias)


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
        lo = torch.zeros_like(self.min_val)
        scale = (self.max_val - lo).clamp_min(1e-8) / qmax
        zp = torch.round(-lo / scale)
        self.register_buffer("scale", scale)
        self.register_buffer("zp", zp)
        self.qmin, self.qmax = 0, qmax
        self.frozen = True


# ---------------------------------------------------------------------------
# model surgery
# ---------------------------------------------------------------------------
def swap_to_quant_modules(model, cfg: QuantConfig):
    """Applied uniformly to every Conv2d/Linear/ReLU(6) in the model — see
    module docstring for the two documented exceptions (BatchNorm, linear
    bottleneck outputs) that fall outside this by construction, not by a
    special case in this function."""
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


def calibrate(model, loader, device, n_batches=20):
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
# REAL int8 packing — this is what actually shrinks on disk
# ---------------------------------------------------------------------------
def pack_int8(model):
    """
    Converts every frozen QuantConv2d/QuantLinear's weight into a genuine
    torch.int8 tensor (not fake-quant float32). Returns a plain dict,
    directly saveable with torch.save() — the resulting file is actually
    smaller, not just theoretically smaller.
    """
    packed, quantized_names = {}, set()
    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            if not m.frozen:
                raise RuntimeError(f"{name} not frozen — call freeze_all(model) first")
            with torch.no_grad():
                q = quantize(m.weight, m.w_scale, 0, m.qmin, m.qmax)
            packed[name] = {
                "qweight": q.to(torch.int8),                    # <-- real int8, 1 byte/value
                "scale": m.w_scale.to(torch.float32).clone(),   # per-channel, fp32
                "bias": m.bias.detach().clone() if m.bias is not None else None,
                "type": "conv" if isinstance(m, QuantConv2d) else "linear",
                "conv_args": (dict(stride=m.stride, padding=m.padding,
                                    dilation=m.dilation, groups=m.groups)
                              if isinstance(m, QuantConv2d) else None),
            }
            quantized_names.add(name)

    # Everything else (BatchNorm weight/bias/running_mean/running_var/
    # num_batches_tracked) is the documented exception — kept fp32 as-is.
    other_fp32 = {}
    for key, tensor in model.state_dict().items():
        owner = key.rsplit(".", 1)[0]
        if owner not in quantized_names:
            other_fp32[key] = tensor.clone()
    packed["_other_fp32"] = other_fp32
    return packed


# ---------------------------------------------------------------------------
# size + compression-ratio reporting (weights)
# ---------------------------------------------------------------------------
def fp32_size_mb(model):
    """Standardized fp32 size: sum of every tensor in state_dict() (params +
    buffers), exactly what torch.save(model.state_dict()) writes to disk
    (plus a few KB of pickle/zip container overhead). This should closely
    match your observed baseline .pth file size — verify with:
        os.path.getsize('best.pth') / (1024**2)
    Any large mismatch means the checkpoint includes extra content (e.g.
    optimizer/scaler state), not just the model weights."""
    total = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return total / (1024 ** 2)


def real_packed_size_mb(packed):
    """The ACTUAL byte count if `packed` were saved right now — no estimation,
    genuinely serializes it in-memory and measures it."""
    buf = io.BytesIO()
    torch.save(packed, buf)
    return len(buf.getvalue()) / (1024 ** 2)


def metadata_overhead_report(packed):
    """Answers Q2(c): storage cost of everything beyond the raw int8 weights
    themselves — per-channel scale factors and the fp32-kept BN parameters."""
    n_layers, n_scales, scale_bytes = 0, 0, 0
    for name, entry in packed.items():
        if name == "_other_fp32":
            continue
        n_layers += 1
        n_scales += entry["scale"].numel()          # = Cout for that layer
        scale_bytes += entry["scale"].numel() * 4     # scales stored fp32

    other_bytes = sum(t.numel() * t.element_size() for t in packed["_other_fp32"].values())
    return {
        "n_quantized_layers": n_layers,
        "n_scale_values_total": n_scales,             # sum of Cout across all quantized layers
        "scale_storage_mb": scale_bytes / (1024 ** 2),
        "bn_and_other_fp32_mb": other_bytes / (1024 ** 2),
    }


def compression_ratio_report(model_fp32, packed):
    """Q4(a)/(d): overall and weights-only compression ratio + final size."""
    fp32_mb = fp32_size_mb(model_fp32)
    packed_mb = real_packed_size_mb(packed)
    meta = metadata_overhead_report(packed)

    fp32_weight_bytes = sum(
        p.numel() * 4 for n, p in model_fp32.named_parameters()
        if n.endswith("weight") and p.dim() > 1     # Conv/Linear weights only, not BN
    )
    quant_weight_bytes = sum(
        e["qweight"].numel() * 1 + e["scale"].numel() * 4
        for n, e in packed.items() if n != "_other_fp32"
    )

    return {
        "fp32_total_mb": fp32_mb,
        "quantized_total_mb": packed_mb,
        "overall_compression_ratio": fp32_mb / packed_mb,
        "fp32_weights_only_mb": fp32_weight_bytes / (1024 ** 2),
        "quantized_weights_only_mb": quant_weight_bytes / (1024 ** 2),
        "weights_compression_ratio": fp32_weight_bytes / quant_weight_bytes,
        **meta,
    }


# ---------------------------------------------------------------------------
# activation compression — measured, not estimated from a formula alone
# ---------------------------------------------------------------------------
def estimate_activation_compression(model, sample_batch, act_bits, device):
    """
    Q4(b) methodology: hooks every ActFakeQuant module and counts how many
    activation values actually flow through it during ONE real forward pass
    on `sample_batch`. Activations are never saved to disk (they're transient,
    recomputed every inference) -- this reports a RUNTIME MEMORY footprint
    for one forward pass, fp32 vs act_bits, not a file size.
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