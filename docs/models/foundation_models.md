# Foundation Models

Adapters for pre-trained Earth Observation foundation models.

---

## Overview

Foundation models are large-scale models pre-trained on massive unlabeled satellite imagery datasets. They provide:
- **Transfer learning** to downstream tasks with limited labels
- **Rich representations** learned from self-supervised objectives
- **Multi-modal understanding** of Earth observation data

---

## Supported Models

| Model | Pre-training | Data | Modality |
|-------|--------------|------|----------|
| MAE | Masked Autoencoding | ImageNet | Optical |
| SatMAE | Masked Autoencoding | fMoW | Multi-spectral |
| Prithvi | Masked Autoencoding | HLS | Multi-spectral + Temporal |
| GFM | Contrastive Learning | Sen12MS | SAR + Optical |

---

## MAE (Masked Autoencoder)

Self-supervised learning by reconstructing masked patches.

### Pre-training Objective

$$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| x_i - \hat{x}_i \|^2$$

Where $\mathcal{M}$ is the set of masked patches.

### Usage

```python
from ununennium.models.foundation import MAEAdapter

# Load pretrained MAE
adapter = MAEAdapter(
    pretrained="satmae",  # "mae_vit_base", "satmae", "prithvi"
    in_channels=12,       # Sentinel-2 bands
    img_size=224,
)

# Fine-tune for segmentation
model = adapter.to_segmentation(num_classes=10)

# Or extract features
features = adapter.extract_features(x)
```

---

## SatMAE

MAE specifically designed for satellite imagery with:
- Support for arbitrary input channels
- Temporal positional embeddings
- Geographic positional embeddings

### Usage

```python
from ununennium.models.foundation import SatMAEAdapter

adapter = SatMAEAdapter(
    pretrained=True,
    temporal_embed=True,
    geo_embed=True,
)

# Process time series
features = adapter(
    images=time_series,        # (B, T, C, H, W)
    timestamps=timestamps,      # (B, T)
    coordinates=lat_lon,        # (B, 2)
)
```

---

## Prithvi

NASA/IBM's geospatial foundation model trained on Harmonized Landsat-Sentinel data.

### Features

- **Multi-temporal**: Handles image time series
- **Multi-spectral**: 6 HLS bands
- **100M+ parameters**: Extensive pre-training

### Usage

```python
from ununennium.models.foundation import PrithviAdapter

adapter = PrithviAdapter(
    model_size="100m",  # "100m", "300m", "600m"
    temporal_frames=3,
)

# Fine-tune for change detection
model = adapter.to_change_detection()
```

---

## Fine-tuning Strategies

### Linear Probing

Freeze backbone, train only head:

```python
adapter = MAEAdapter(pretrained="satmae")
adapter.freeze_backbone()

model = adapter.to_classification(num_classes=10)
# Only classification head is trained
```

### Full Fine-tuning

Train entire model with lower learning rate for backbone:

```python
model = adapter.to_segmentation(num_classes=10)

optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},
    {"params": model.head.parameters(), "lr": 1e-4},
])
```

### LoRA (Low-Rank Adaptation)

Parameter-efficient fine-tuning:

```python
adapter = MAEAdapter(pretrained="satmae")
model = adapter.to_segmentation(
    num_classes=10,
    lora_rank=8,        # Low-rank adaptation
    lora_alpha=16,
)
# Only ~0.1% of parameters are trainable
```

---

## Coming Soon

- **GFM**: Geospatial Foundation Model for SAR-Optical fusion
- **DINO-MC**: Self-distillation for multi-spectral imagery
- **Scale-MAE**: Multi-resolution pre-training

---

## References

1. He, K., et al. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR.
2. Cong, Y., et al. (2022). *SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery*. NeurIPS.
3. Jakubik, J., et al. (2023). *Prithvi: A Foundation Model for Earth Observation*. arXiv.
