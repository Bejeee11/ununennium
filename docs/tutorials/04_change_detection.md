# Change Detection Tutorial

Detecting changes in bitemporal imagery.

## Data Loading

Load pre-event and post-event images.

```python
pre_img = read_tensor("2024_pre.tif")
post_img = read_tensor("2025_post.tif")
```

## Siamese Network

Use a Siamese architecture to extract features from both times.

```python
from ununennium.models.architectures import SiameseUNet

model = SiameseUNet(in_channels=3, classes=2, fusion="diff")
# fusion can be "diff", "concat", or "dist"
```

## Training

The loss should handle class imbalance (change is rare).

```python
from ununennium.losses import DiceLoss

criterion = DiceLoss()
```
