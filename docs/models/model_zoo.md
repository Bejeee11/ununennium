# Ununennium Model Zoo

Production-ready pretrained models for Earth Observation tasks.

---

## Quick Links

| Model Family | Task | Download |
|--------------|------|----------|
| [U-Net ResNet-50](#segmentation-models) | Segmentation | [Download](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/unet_resnet50.pt) |
| [U-Net EfficientNet-B4](#segmentation-models) | Segmentation | [Download](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/unet_efficientnet_b4.pt) |
| [DeepLabV3+](#segmentation-models) | Segmentation | [Download](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/deeplabv3_resnet50.pt) |
| [Pix2Pix](#generative-models) | Image Translation | [Download](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/pix2pix_sar2optical.pt) |
| [CycleGAN](#generative-models) | Domain Adaptation | [Download](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/cyclegan_summer2winter.pt) |
| [RetinaNet](#detection-models) | Object Detection | [Download](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/retinanet_resnet50.pt) |

---

## Installation of Pretrained Weights

```python
import torch
from ununennium.models import create_model

# Option 1: Load from URL
model = create_model("unet_resnet50", in_channels=12, num_classes=10)
state_dict = torch.hub.load_state_dict_from_url(
    "https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/unet_resnet50.pt",
    map_location="cpu"
)
model.load_state_dict(state_dict)

# Option 2: Load from local file
model.load_state_dict(torch.load("path/to/unet_resnet50.pt"))
```

---

## Segmentation Models

### U-Net Family

The workhorse of semantic segmentation for Earth Observation.

| Model | Backbone | Params | Input Size | mIoU (LandCover) | Download |
|-------|----------|--------|------------|------------------|----------|
| `unet_resnet50` | ResNet-50 | 32M | 512×512×12 | 0.78 | [.pt](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/unet_resnet50.pt) |
| `unet_efficientnet_b4` | EfficientNet-B4 | 19M | 512×512×12 | 0.81 | [.pt](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/unet_efficientnet_b4.pt) |

**Mathematical Formulation:**

Let $E$ be the encoder function and $D$ be the decoder:

$$z_l = E_l(z_{l-1}) \quad \text{(Downsampling)}$$

$$x_l = D_l(x_{l+1}, z_l) \quad \text{(Upsampling + Skip)}$$

The skip connection $z_l$ preserves high-frequency spatial details critical for boundary detection.

**Usage:**

```python
from ununennium.models import create_model

model = create_model(
    "unet_resnet50",
    in_channels=12,      # Sentinel-2 bands
    num_classes=10,      # Land cover classes
)

# Forward pass
import torch
x = torch.randn(4, 12, 512, 512)
output = model(x)  # (4, 10, 512, 512)
```

### DeepLabV3+

Multi-scale context capture using Atrous Spatial Pyramid Pooling (ASPP).

| Model | Backbone | Params | Input Size | mIoU | Download |
|-------|----------|--------|------------|------|----------|
| `deeplabv3_resnet50` | ResNet-50 | 40M | 512×512×12 | 0.80 | [.pt](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/deeplabv3_resnet50.pt) |

**Atrous (Dilated) Convolution:**

$$y[i] = \sum_k x[i + r \cdot k] w[k]$$

Where $r$ is the dilation rate, expanding the receptive field without increasing parameters.

---

## Generative Models

### Pix2Pix (Conditional GAN)

Paired image-to-image translation (SAR → Optical, Cloud → Clear).

| Model | Task | Params | Input Size | FID | Download |
|-------|------|--------|------------|-----|----------|
| `pix2pix_sar2optical` | SAR→Optical | 54M | 256×256×1 | 42.3 | [.pt](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/pix2pix_sar2optical.pt) |
| `pix2pix_cloud_removal` | Cloud Removal | 54M | 256×256×4 | 38.7 | [.pt](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/pix2pix_cloud_removal.pt) |

**cGAN Objective:**

$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{x,z}[\log(1 - D(x, G(x, z)))]$$

**L1 Reconstruction:**

$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y,z}[\| y - G(x, z) \|_1]$$

**Usage:**

```python
from ununennium.models.gan import Pix2Pix

model = Pix2Pix(in_channels=1, out_channels=3)
state_dict = torch.hub.load_state_dict_from_url(
    "https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/pix2pix_sar2optical.pt"
)
model.generator.load_state_dict(state_dict)

# Generate optical from SAR
sar_input = torch.randn(1, 1, 256, 256)
optical_output = model.generator(sar_input)
```

### CycleGAN (Unpaired Translation)

Domain adaptation without paired data (Summer → Winter, Sensor A → Sensor B).

| Model | Task | Params | FID | Download |
|-------|------|--------|-----|----------|
| `cyclegan_summer2winter` | Season Transfer | 28M | 65.2 | [.pt](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/cyclegan_summer2winter.pt) |
| `cyclegan_s1_to_s2` | Sentinel-1→2 | 28M | 58.4 | [.pt](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/cyclegan_s1_to_s2.pt) |

**Cycle Consistency Loss:**

$$x \to G(x) \to F(G(x)) \approx x$$

$$\mathcal{L}_{cyc} = \| F(G(x)) - x \|_1 + \| G(F(y)) - y \|_1$$

---

## Detection Models

### RetinaNet

Anchor-based object detection with Focal Loss.

| Model | Backbone | Params | mAP@0.5 | Download |
|-------|----------|--------|---------|----------|
| `retinanet_resnet50` | ResNet-50 FPN | 37M | 0.62 | [.pt](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/retinanet_resnet50.pt) |

**Focal Loss:**

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where $\gamma$ focuses training on hard negatives.

**Usage:**

```python
from ununennium.models.architectures.detection import RetinaNet

model = RetinaNet(num_classes=15, backbone="resnet50")

# Detection forward pass
images = torch.randn(2, 12, 512, 512)
detections = model(images)  # List of (boxes, scores, labels)
```

---

## Physics-Informed Models (PINN)

Neural networks constrained by physical laws.

**Total Loss:**

$$\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{PDE}$$

Where $\mathcal{L}_{PDE}$ is the residual of the governing equation (Diffusion, Advection).

**Usage:**

```python
from ununennium.models.pinn import PINN, DiffusionEquation, MLP

equation = DiffusionEquation(diffusivity=0.1)
pinn = PINN(
    network=MLP([2, 128, 128, 1]),
    equation=equation,
    lambda_pde=10.0,
)

# Training with physics constraints
losses = pinn.compute_loss(x_data, u_data, x_collocation)
```

---

## Vision Transformers

### ViT for Remote Sensing

Global context modeling for scene understanding.

**Self-Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Model | Params | Input Size | Accuracy | Download |
|-------|--------|------------|----------|----------|
| `vit_base_patch16` | 86M | 224×224×12 | 0.89 | [.pt](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/vit_base_patch16.pt) |

---

## Model Performance Summary

| Model | Task | Key Metric | GPU Memory | Throughput |
|-------|------|------------|------------|------------|
| U-Net ResNet-50 | Segmentation | mIoU: 0.78 | 12.4 GB | 142 img/s |
| U-Net EfficientNet-B4 | Segmentation | mIoU: 0.81 | 14.2 GB | 98 img/s |
| DeepLabV3+ | Segmentation | mIoU: 0.80 | 15.1 GB | 87 img/s |
| Pix2Pix | Translation | FID: 38.7 | 8.6 GB | 67 img/s |
| CycleGAN | Domain Adapt | FID: 58.4 | 10.2 GB | 45 img/s |
| RetinaNet | Detection | mAP: 0.62 | 11.8 GB | 52 img/s |
| ViT-Base | Classification | Acc: 0.89 | 18.1 GB | 256 img/s |

*Benchmarks on NVIDIA A100 80GB, PyTorch 2.1, CUDA 12.1, batch size varies by model.*

---

## License

All pretrained weights are released under the Apache 2.0 License, matching the library license.

**Citation:**

```bibtex
@software{ununennium2025,
  title = {Ununennium: Production-grade Satellite Imagery Machine Learning},
  author = {Laitinen Imanov, Olaf Yunus},
  year = {2025},
  version = {1.0.5},
  url = {https://github.com/olaflaitinen/ununennium}
}
```
