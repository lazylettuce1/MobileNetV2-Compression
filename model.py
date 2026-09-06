"""
model.py — MobileNetV2 adapted for CIFAR-10 (32x32 inputs)

Stock torchvision MobileNetV2 is built for 224x224 ImageNet images and applies
5 stride-2 downsampling steps (total downsample factor 32x: 224 -> 7). Fed a
32x32 CIFAR image unmodified, that collapses to a 1x1 feature map before the
classifier and throws away almost all spatial structure.

Fix: remove exactly TWO of the five stride-2 steps (not more), which brings
total downsampling to 8x -> a 4x4 final feature map for a 32x32 input. This
matches the standard CIFAR-adapted MobileNetV2 recipe used across the
literature (e.g. kuangliu/pytorch-cifar) and keeps compute low, which matters
because you'll be running many more experiments during the compression phase.

Layers touched (torchvision's `features` Sequential layout):
    features[0]      : stem Conv2dNormActivation(3 -> 32*width_mult), stride 2 -> 1
    features[2].conv : first InvertedResidual block of the (t=6, c=24) stage,
                        depthwise conv stride 2 -> 1

Nothing else (channel widths, expansion ratios, number of blocks per stage) is
changed, so the topology stays a minimal, well-validated deviation from stock
MobileNetV2 rather than a redesign. Because stride is a runtime property (not
a weight shape), ImageNet-pretrained weights still load cleanly into this
architecture if you want to experiment with `pretrained=True`.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def MobileNetV2CIFAR(num_classes: int = 10,
                             width_mult: float = 1.0,
                             dropout: float = 0.2,
                             pretrained: bool = False) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights, width_mult=width_mult, dropout=dropout)

    # --- CIFAR stride surgery (see module docstring) ---------------------
    stem_conv = model.features[0][0]
    assert isinstance(stem_conv, nn.Conv2d), (
        "torchvision internals changed — inspect model.features[0] manually"
    )
    stem_conv.stride = (1, 1)

    stage2_block = model.features[2]
    depthwise_conv = stage2_block.conv[1][0]
    assert isinstance(depthwise_conv, nn.Conv2d), (
        "torchvision internals changed — inspect model.features[2].conv manually"
    )
    depthwise_conv.stride = (1, 1) 
    # -----------------------------------------------------------------------

    # Swap the classifier head for CIFAR-10 (num_classes), keep dropout as-is
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    m = MobileNetV2CIFAR()
    x = torch.randn(2, 3, 32, 32)
    out = m(x)
    print("output shape:", out.shape)  # expect torch.Size([2, 10])

    # sanity-check the final feature map size before global pooling
    feat = m.features(x)
    print("final feature map:", feat.shape)  # expect [2, 1280, 4, 4]

    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params/1e6:.2f}M")
