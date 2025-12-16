# Backbone Architectures

Feature extraction backbones optimized for multi-spectral satellite imagery.

---

## Overview

Ununennium provides backbone networks that support arbitrary input channels (1-200+ bands) unlike standard ImageNet-pretrained models that only accept 3 channels.

---

## ResNet Backbone

Deep residual networks with skip connections.

### Architecture

The residual block solves the vanishing gradient problem:

$$y = \mathcal{F}(x, \{W_i\}) + x$$

Where $\mathcal{F}$ is the residual mapping to be learned.

### Usage

```python
from ununennium.models.backbones import ResNetBackbone

# 12-channel Sentinel-2 input
backbone = ResNetBackbone(
    variant="resnet50",    # resnet18, resnet34, resnet50, resnet101
    in_channels=12,
    pretrained=True,       # ImageNet weights adapted for N-channels
    output_stride=32,
)

# Extract multi-scale features
features = backbone(x)  # List of feature maps at different scales
# features[0]: stride 4, features[1]: stride 8, etc.
```

### Variants

| Variant | Params | FLOPs (512×512) | Top-1 Acc |
|---------|--------|-----------------|-----------|
| ResNet-18 | 11.7M | 1.8G | 69.8% |
| ResNet-34 | 21.8M | 3.7G | 73.3% |
| ResNet-50 | 25.6M | 4.1G | 76.1% |
| ResNet-101 | 44.5M | 7.8G | 77.4% |

---

## EfficientNet Backbone

Compound scaling for optimal accuracy/efficiency trade-off.

### Compound Scaling

$$depth: d = \alpha^\phi$$
$$width: w = \beta^\phi$$
$$resolution: r = \gamma^\phi$$

Subject to: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

### Usage

```python
from ununennium.models.backbones import EfficientNetBackbone

backbone = EfficientNetBackbone(
    variant="efficientnet_b4",  # b0-b7
    in_channels=12,
    pretrained=True,
)

features = backbone(x)
```

### Variants

| Variant | Params | FLOPs | Resolution |
|---------|--------|-------|------------|
| B0 | 5.3M | 0.4G | 224 |
| B1 | 7.8M | 0.7G | 240 |
| B2 | 9.2M | 1.0G | 260 |
| B3 | 12M | 1.8G | 300 |
| B4 | 19M | 4.2G | 380 |
| B5 | 30M | 9.9G | 456 |

---

## Vision Transformer (ViT)

Attention-based backbone for global context modeling.

### Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Usage

```python
from ununennium.models.backbones import ViTBackbone

backbone = ViTBackbone(
    variant="vit_base_patch16",
    in_channels=12,
    img_size=224,
)
```

---

## Multi-Channel Adaptation

All backbones automatically adapt pretrained weights for arbitrary input channels:

```python
# Original: 3-channel ImageNet weights
# Adaptation: Average weights across channels, then tile to N channels
# Result: Pretrained knowledge preserved for multi-spectral inputs

backbone = ResNetBackbone(
    variant="resnet50",
    in_channels=12,  # 12-channel Sentinel-2
    pretrained=True,  # Adapted ImageNet weights
)
```

---

## Custom Backbone

Register your own backbone:

```python
from ununennium.models.registry import register_backbone

@register_backbone("my_backbone")
class MyBackbone(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        # Your implementation
        
    def forward(self, x):
        # Return list of multi-scale features
        return [feat1, feat2, feat3, feat4]
```
