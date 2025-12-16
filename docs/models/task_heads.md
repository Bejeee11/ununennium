# Task Heads

Output heads for different machine learning tasks.

---

## Overview

Task heads connect backbone features to task-specific outputs. Ununennium provides heads for:
- **Classification**: Scene-level predictions
- **Segmentation**: Per-pixel predictions
- **Detection**: Object localization and classification

---

## ClassificationHead

Global pooling followed by fully-connected layers for scene classification.

### Architecture

$$\hat{y} = \text{FC}(\text{GlobalPool}(f_L))$$

Where $f_L$ is the final backbone feature map.

### Usage

```python
from ununennium.models.heads import ClassificationHead

head = ClassificationHead(
    in_channels=2048,      # Backbone output channels
    num_classes=10,        # Number of classes
    dropout=0.5,           # Dropout rate
    pooling="avg",         # "avg" or "max"
)

# Forward pass
features = backbone(x)  # List of feature maps
logits = head(features)  # (B, num_classes)
```

---

## SegmentationHead

U-Net style decoder with skip connections for dense prediction.

### Architecture

$$x_l = D_l(\text{Upsample}(x_{l+1}), z_l)$$

Where $z_l$ is the encoder skip connection at level $l$.

### Usage

```python
from ununennium.models.heads import SegmentationHead

head = SegmentationHead(
    encoder_channels=[64, 128, 256, 512, 1024],
    decoder_channels=[256, 128, 64, 32],
    num_classes=10,
    dropout=0.1,
)

# Forward pass
features = backbone(x)  # Multi-scale features
segmentation = head(features)  # (B, num_classes, H, W)
```

---

## DetectionHead

Anchor-based detection head with classification and regression branches.

### Architecture

Separate conv branches for:
- **Classification**: $P(class|anchor)$
- **Regression**: $(\Delta x, \Delta y, \Delta w, \Delta h)$

### Focal Loss Initialization

$$\text{bias} = -\log\left(\frac{1 - \pi}{\pi}\right)$$

Where $\pi = 0.01$ is the prior probability of foreground.

### Usage

```python
from ununennium.models.heads import DetectionHead

head = DetectionHead(
    in_channels=256,       # FPN output channels
    num_classes=15,        # Object classes (excluding background)
    num_anchors=9,         # Anchors per location
    num_convs=4,           # Convolution depth per branch
)

# Forward pass (applied to each FPN level)
cls_logits, box_regression = head(fpn_features)
```

---

## FPNHead

Feature Pyramid Network head for multi-scale segmentation.

```python
from ununennium.models.heads import FPNHead

head = FPNHead(
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_classes=10,
)

output = head(backbone_features)
```

---

## Custom Head

Register your own head architecture:

```python
from ununennium.models.registry import register_head

@register_head("my_head")
class MyHead(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        # Your implementation
        
    def forward(self, features):
        # features: list of backbone feature maps
        return output
```
