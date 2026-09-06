"""
quant.py — INT8 post-training quantization: per-channel weights, per-tensor activations.

Flow:
    1. load_baseline()        -> plain float32 model, exactly as trained
    2. swap_to_quant_modules  -> replace Conv2d/Linear/ReLU(6) with quant-aware versions
    3. calibrate              -> run a few real batches through so activations can
                                  observe their true value range (weights don't need this —
                                  their range is just read directly off the tensor)
    4. freeze_all             -> lock in scale/zero_point for both weights and activations
    5. evaluate               -> same eval loop you already use, unchanged
"""
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# core int8 math — shared by weights (per-channel) and activations (per-tensor)
# ---------------------------------------------------------------------------
def quantize(x, scale, zero_point, qmin, qmax):
    return torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)


def dequantize(q, scale, zero_point):
    return (q - zero_point) * scale


# ---------------------------------------------------------------------------
# WEIGHTS — per-channel, symmetric (zero_point=0, since weights center on 0)
# ---------------------------------------------------------------------------
def weight_qparams_per_channel(w, n_bits=8, eps=1e-8):
    """
    w: (Cout, Cin/groups, kH, kW) for conv, or (out_features, in_features) for linear.
    Returns scale shaped so it broadcasts against w directly: (Cout, 1, 1, 1) or (Cout, 1).
    """
    reduce_dims = list(range(1, w.dim()))                     # every dim except channel (dim 0)
    max_abs = w.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    qmax = (1 << (n_bits - 1)) - 1                             # 127 for 8-bit
    scale = max_abs / qmax
    return scale, -qmax, qmax                                  # zero_point is always 0 here


class QuantConv2d(nn.Conv2d):
    """Drop-in replacement for Conv2d. Weights quantized per-output-channel."""
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.frozen = False

    @torch.no_grad()
    def freeze(self):
        # "calibrate": weight range is just read off the tensor directly — no
        # data needed, unlike activations, since these values already exist.
        scale, qmin, qmax = weight_qparams_per_channel(self.weight, self.weight_bits)
        self.register_buffer("w_scale", scale)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        # fake-quant: snap weights to the int8 grid, then immediately convert
        # back to float32 — simulates quantization's accuracy impact without
        # needing real int8 conv kernels
        q = quantize(self.weight, self.w_scale, 0, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, 0)
        return F.conv2d(x, w_dq, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    """Same idea as QuantConv2d, for the final classifier layer."""
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


# ---------------------------------------------------------------------------
# ACTIVATIONS — per-tensor, asymmetric (post-ReLU6 values are always >= 0)
# ---------------------------------------------------------------------------
class ActFakeQuant(nn.Module):
    """
    Inserted right after every ReLU/ReLU6. Two phases:
      - calibrating (frozen=False): just watches the data, records min/max seen
        so far. Doesn't touch the values.
      - frozen (frozen=True): quantizes+dequantizes every input using the
        scale/zero_point locked in by freeze().
    """
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
        qmax = (1 << self.n_bits) - 1                          # 255 for 8-bit, unsigned range
        lo = torch.zeros_like(self.min_val)                    # ReLU6 output is always >= 0
        scale = (self.max_val - lo).clamp_min(1e-8) / qmax
        zp = torch.round(-lo / scale)
        self.register_buffer("scale", scale)
        self.register_buffer("zp", zp)
        self.qmin, self.qmax = 0, qmax
        self.frozen = True


# ---------------------------------------------------------------------------
# model surgery: swap layers, then calibrate + freeze the whole model
# ---------------------------------------------------------------------------
def swap_to_quant_modules(model, weight_bits=8, act_bits=8):
    """Replaces Conv2d -> QuantConv2d, Linear -> QuantLinear,
    ReLU/ReLU6 -> Sequential(same activation, ActFakeQuant). Recurses into
    every submodule, copying original weights so nothing is lost."""
    for name, m in list(model.named_children()):
        swap_to_quant_modules(m, weight_bits, act_bits)

        if isinstance(m, nn.Conv2d):
            q = QuantConv2d(m.in_channels, m.out_channels, m.kernel_size,
                             stride=m.stride, padding=m.padding, dilation=m.dilation,
                             groups=m.groups, bias=(m.bias is not None), weight_bits=weight_bits)
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.Linear):
            q = QuantLinear(m.in_features, m.out_features, bias=(m.bias is not None), weight_bits=weight_bits)
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, (nn.ReLU, nn.ReLU6)):
            # NOTE: MobileNetV2 uses ReLU6, not ReLU — type(m)(...) recreates
            # whichever one it actually was, instead of hardcoding ReLU.
            setattr(model, name, nn.Sequential(OrderedDict([
                ("act", type(m)(inplace=False)),
                ("aq", ActFakeQuant(n_bits=act_bits)),
            ])))
    return model


def calibrate(model, loader, device, n_batches=20):
    """Feed a few real batches through so ActFakeQuant modules see genuine
    activation ranges before freezing. Weights don't need this step."""
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            model(images.to(device))
            if i + 1 >= n_batches:
                break


def freeze_all(model):
    """Locks in scale/zero_point everywhere, for both weights and activations."""
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, ActFakeQuant)):
            m.freeze()


# ---------------------------------------------------------------------------
# size estimate — accounts for per-channel scale storage overhead (Q2c)
# ---------------------------------------------------------------------------
def estimate_size_mb(model, weight_bits=8):
    total_bytes = 0
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            n_weights = m.weight.numel()
            total_bytes += n_weights * weight_bits // 8         # quantized weights
            total_bytes += m.w_scale.numel() * 4                # one float32 scale per channel
            if m.bias is not None:
                total_bytes += m.bias.numel() * 4                # biases kept fp32
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# end-to-end usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from model import MobileNetV2CIFAR
    from data import get_dataloaders
    from utils import accuracy, AverageMeter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_ckpt = "./outputs/baseline/best.pth"  # path to the trained float32 baseline

    # 1. load the trained float32 baseline
    model = MobileNetV2CIFAR(num_classes=10, dropout=0.2)
    ckpt = torch.load(baseline_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    _, test_loader = get_dataloaders(data_dir="...", download=False)

    # 2-4. quantize: swap -> calibrate -> freeze
    swap_to_quant_modules(model, weight_bits=8, act_bits=8)
    calibrate(model, test_loader, device, n_batches=20)
    freeze_all(model)

    # 5. evaluate — same accuracy() helper you already have, nothing new
    model.eval()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            top1, = accuracy(outputs, targets, topk=(1,))
            acc_meter.update(top1, images.size(0))

    print(f"Baseline test acc: {ckpt['best_acc']:.2f}%")
    print(f"INT8 quantized test acc: {acc_meter.avg:.2f}%")
    print(f"Estimated quantized size: {estimate_size_mb(model, weight_bits=8):.2f} MB")