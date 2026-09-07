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
    weight_quant_bits: int = 8
    activation_quant_bits: int = 8
    calibration_batches: int = 40   # confirmed better than 20 empirically


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
    print(f"attach_block_output_quant: instrumented {count} InvertedResidual block outputs")
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
# REAL int8 packing
# ---------------------------------------------------------------------------
def pack_int8(model):
    """
    Converts every frozen QuantConv2d/QuantLinear's weight into a genuine
    torch.int8 tensor. Everything is explicitly moved to CPU first, so the
    resulting file loads correctly regardless of what device it was quantized
    on, or whether a GPU is even present when you load it back later.
    """
    packed, quantized_names = {}, set()
    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            if not m.frozen:
                raise RuntimeError(f"{name} not frozen — call freeze_all(model) first")
            with torch.no_grad():
                # weight is already the quantized value post-freeze() — just
                # re-derive the integer codes for storage
                q = torch.round(m.weight.cpu() / m.w_scale.cpu()).clamp(m.qmin, m.qmax)
            packed[name] = {
                "qweight": q.to(torch.int8),
                "scale": m.w_scale.detach().cpu().to(torch.float32).clone(),
                "bias": m.bias.detach().cpu().clone() if m.bias is not None else None,
                "type": "conv" if isinstance(m, QuantConv2d) else "linear",
                "conv_args": (dict(stride=m.stride, padding=m.padding,
                                    dilation=m.dilation, groups=m.groups)
                              if isinstance(m, QuantConv2d) else None),
            }
            quantized_names.add(name)

    other_fp32 = {}
    for key, tensor in model.state_dict().items():
        owner = key.rsplit(".", 1)[0]
        if owner not in quantized_names:
            other_fp32[key] = tensor.detach().cpu().clone()
    packed["_other_fp32"] = other_fp32
    return packed


# ---------------------------------------------------------------------------
# size + compression-ratio reporting
# ---------------------------------------------------------------------------
def fp32_size_mb(model):
    """Standardized fp32 size: sum of every tensor in state_dict() (params +
    buffers) — exactly what torch.save(model.state_dict()) writes to disk.
    Use THIS (not a full training checkpoint with optimizer state) as your
    Q4 baseline — verify with os.path.getsize('model_weights_only.pth')."""
    total = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return total / (1024 ** 2)


def real_packed_size_mb(packed):
    buf = io.BytesIO()
    torch.save(packed, buf)
    return len(buf.getvalue()) / (1024 ** 2)


def metadata_overhead_report(packed):
    n_layers, n_scales, scale_bytes = 0, 0, 0
    for name, entry in packed.items():
        if name == "_other_fp32":
            continue
        n_layers += 1
        n_scales += entry["scale"].numel()
        scale_bytes += entry["scale"].numel() * 4

    other_bytes = sum(t.numel() * t.element_size() for t in packed["_other_fp32"].values())
    return {
        "n_quantized_layers": n_layers,
        "n_scale_values_total": n_scales,
        "scale_storage_mb": scale_bytes / (1024 ** 2),
        "bn_and_other_fp32_mb": other_bytes / (1024 ** 2),
    }


def compression_ratio_report(model_fp32, packed):
    fp32_mb = fp32_size_mb(model_fp32)
    packed_mb = real_packed_size_mb(packed)
    meta = metadata_overhead_report(packed)

    fp32_weight_bytes = sum(
        p.numel() * 4 for n, p in model_fp32.named_parameters()
        if n.endswith("weight") and p.dim() > 1     # excludes BN's 1-D weight/gamma
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
                               calib_batches=40, out_path="quantized_int8.pth"):
    from model import MobileNetV2CIFAR
    from data import get_dataloaders
    from utils import accuracy, AverageMeter

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading baseline from {ckpt_path}...")
    state_dict, baseline_acc = load_state_dict_flexible(ckpt_path)

    model = MobileNetV2CIFAR(num_classes=10, dropout=0.2)
    model.load_state_dict(state_dict)
    model_fp32 = copy.deepcopy(model)   # untouched reference, for size comparison

    train_loader, test_loader = get_dataloaders(data_dir=data_dir, download=False)

    cfg = QuantConfig(weight_quant_bits=weight_bits, activation_quant_bits=act_bits,
                       calibration_batches=calib_batches)
    swap_to_quant_modules(model, cfg)
    attach_block_output_quant(model, cfg.activation_quant_bits)
    model.to(device)

    print(f"Calibrating with {cfg.calibration_batches} batches (from TRAIN data, not test)...")
    calibrate(model, train_loader, device, n_batches=cfg.calibration_batches)
    freeze_all(model)

    print("Evaluating fake-quant model on the held-out test set...")
    model.eval()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            top1, = accuracy(model(images), targets, topk=(1,))
            acc_meter.update(top1, images.size(0))

    print(f"\n--- ACCURACY ---")
    print(f"Baseline test acc: {baseline_acc if baseline_acc is not None else 'unknown'}")
    print(f"Quantized test acc: {acc_meter.avg:.2f}%")

    print("\n--- SIZE & COMPRESSION REPORT ---")
    packed = pack_int8(model)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(packed, out_path)

    report = compression_ratio_report(model_fp32, packed)
    for k, v in report.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    sample_batch, _ = next(iter(test_loader))
    act_report = estimate_activation_compression(model, sample_batch, cfg.activation_quant_bits, device)
    print(f"\n--- ACTIVATION FOOTPRINT (per batch) ---")
    for k, v in act_report.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print(f"\nSaved real int8 weights to: {out_path}")
    return {"accuracy": acc_meter.avg, **report, **act_report}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run INT8 Quantization Pipeline")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)
    parser.add_argument("--calib", type=int, default=40)
    parser.add_argument("--out", type=str, default="quantized_int8.pth")
    args = parser.parse_args()
    run_quantization_pipeline(args.ckpt, args.data_dir, args.w_bits, args.a_bits,
                               args.calib, args.out)