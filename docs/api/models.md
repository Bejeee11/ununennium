# Models API

The models module provides a registry of neural network architectures for Earth observation tasks.

---

## Model Registry

### create_model

```python
def create_model(
    name: str,
    in_channels: int = 3,
    num_classes: int = 10,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    """Create model from registry.
    
    Parameters
    ----------
    name : str
        Model name (e.g., "unet_resnet50").
    in_channels : int, optional
        Number of input channels. Default: 3.
    num_classes : int, optional
        Number of output classes. Default: 10.
    pretrained : bool, optional
        Load pretrained weights. Default: False.
    **kwargs
        Model-specific parameters.
    
    Returns
    -------
    nn.Module
        Initialized model.
    """
```

### list_models

```python
def list_models(task: str | None = None) -> list[str]:
    """List available models.
    
    Parameters
    ----------
    task : str | None, optional
        Filter by task: "segmentation", "classification", "detection".
    
    Returns
    -------
    list[str]
        Available model names.
    """
```

### Example

```python
from ununennium.models import create_model, list_models

# List segmentation models
print(list_models(task="segmentation"))
# ["unet_resnet50", "unet_efficientnet_b4", "deeplabv3_resnet101", ...]

# Create model
model = create_model(
    "unet_resnet50",
    in_channels=12,
    num_classes=10,
    pretrained=True,
)
```

---

## Segmentation Architectures

### U-Net

```python
class UNet(nn.Module):
    """U-Net architecture with configurable backbone.
    
    Parameters
    ----------
    backbone : str
        Encoder backbone name.
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of output classes.
    decoder_channels : list[int], optional
        Decoder channel sizes. Default: [256, 128, 64, 32].
    """
```

Available U-Net variants:

| Model Name | Backbone | Parameters |
|------------|----------|------------|
| `unet_resnet18` | ResNet-18 | 14M |
| `unet_resnet50` | ResNet-50 | 32M |
| `unet_efficientnet_b4` | EfficientNet-B4 | 19M |

### DeepLabV3+

```python
class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with atrous spatial pyramid pooling.
    
    Parameters
    ----------
    backbone : str
        Encoder backbone name.
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of output classes.
    output_stride : int, optional
        Output stride. Default: 16.
    aspp_dilations : list[int], optional
        ASPP dilation rates. Default: [6, 12, 18].
    """
```

---

## Backbones

### Available Backbones

| Name | Type | Pretrained |
|------|------|------------|
| `resnet18` | CNN | ImageNet |
| `resnet50` | CNN | ImageNet |
| `resnet101` | CNN | ImageNet |
| `efficientnet_b0` | CNN | ImageNet |
| `efficientnet_b4` | CNN | ImageNet |
| `vit_base_patch16` | ViT | ImageNet-21k |
| `swin_tiny` | Swin | ImageNet |

### Custom Backbone Registration

```python
from ununennium.models import register_backbone

@register_backbone("my_backbone")
class MyBackbone(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Implementation
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Return multi-scale features
        return [f1, f2, f3, f4]
```

---

## Task Heads

### SegmentationHead

```python
class SegmentationHead(nn.Module):
    """Segmentation output head.
    
    Parameters
    ----------
    in_channels : int
        Input feature channels.
    num_classes : int
        Number of output classes.
    kernel_size : int, optional
        Final convolution kernel size. Default: 3.
    """
```

### ClassificationHead

```python
class ClassificationHead(nn.Module):
    """Classification output head.
    
    Parameters
    ----------
    in_channels : int
        Input feature channels.
    num_classes : int
        Number of output classes.
    pooling : str, optional
        Pooling type: "avg", "max". Default: "avg".
    """
```

---

## Model Configuration

### From Config Dictionary

```python
config = {
    "name": "unet_resnet50",
    "in_channels": 12,
    "num_classes": 10,
    "decoder_channels": [256, 128, 64, 32],
}

model = create_model(**config)
```

### From YAML

```yaml
model:
  name: unet_resnet50
  in_channels: 12
  num_classes: 10
  pretrained: true
```

```python
import yaml
from ununennium.models import create_model

with open("config.yaml") as f:
    config = yaml.safe_load(f)

model = create_model(**config["model"])
```
