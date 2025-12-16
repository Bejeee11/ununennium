# Models API

## Segmentation Models

### `UNet`

```python
UNet(in_channels=3, classes=2, encoder_name="resnet34")
```

Standard U-Net with ResNet encoder.

### `DeepLabV3Plus`

```python
DeepLabV3Plus(in_channels=3, classes=2, encoder_name="resnet50")
```

For multi-scale context aggregation.

## Object Detection

### `FasterRCNN`

```python
FasterRCNN(num_classes=10, backbone="resnet50_fpn")
```

## Registry

Models can be instantiated via registry:

```python
from ununennium.models import create_model

model = create_model("unet", encoder_name="efficientnet-b0")
```
