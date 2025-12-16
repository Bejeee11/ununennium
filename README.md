# Ununennium

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/ununennium?color=blue)](https://pypi.org/project/ununennium/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ununennium)](https://pypi.org/project/ununennium/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/github/actions/workflow/status/olaflaitinen/ununennium/ci.yml?branch=main&label=CI)](https://github.com/olaflaitinen/ununennium/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-available-blue)](https://github.com/olaflaitinen/ununennium/tree/main/docs)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

**Production-grade Python library for satellite and geospatial imagery machine learning.**

[Documentation](docs/README.md) |
[PyPI](https://pypi.org/project/ununennium/) |
[GitHub](https://github.com/olaflaitinen/ununennium) |
[Examples](examples/)

</div>

---

## Overview

Ununennium (Element 119, the next alkali metal) is a comprehensive framework for Earth observation machine learning. The library provides unified, GPU-accelerated workflows spanning cloud-native data ingestion, geospatially-aware preprocessing, model training, and production deployment.

The framework addresses fundamental challenges in geospatial machine learning that general-purpose deep learning libraries overlook:

- Coordinate Reference System (CRS) preservation through the entire processing pipeline
- Radiometrically correct resampling with proper kernel selection
- Physics-informed loss functions for scientific applications
- Spatially-aware train/validation/test splitting that respects spatial autocorrelation

### Design Philosophy

| Challenge | Traditional Approach | Ununennium Solution |
|-----------|---------------------|---------------------|
| **CRS Handling** | Manual, error-prone transformations | Automatic CRS tracking via `GeoTensor` |
| **Large Rasters** | Memory overflow on load | Streaming I/O with COG/Zarr, lazy evaluation |
| **Multi-spectral Data** | Custom band handling per sensor | First-class n-band support with sensor profiles |
| **Reproducibility** | Ad-hoc seeds, non-deterministic | Deterministic pipeline with spatial CV |
| **Physics Constraints** | Purely data-driven predictions | Physics-informed neural networks (PINN) |
| **Spatial Leakage** | Random pixel-wise splits | Block-based spatial cross-validation |

---

## Key Features

| Module | Capability | Description |
|--------|------------|-------------|
| **`core`** | GeoTensor, GeoBatch | CRS-aware tensors with coordinate tracking and transform propagation |
| **`io`** | COG, STAC, Zarr | Cloud-native streaming with lazy loading, windowed reads |
| **`models`** | CNN, ViT, GAN, PINN | 15+ architectures with registry pattern and pretrained weights |
| **`training`** | Trainer, Callbacks | Mixed precision, gradient accumulation, DDP, checkpointing |
| **`preprocessing`** | Indices, Normalization | NDVI, EVI, SAVI with sensor-aware radiometric math |
| **`augmentation`** | Geometric, Radiometric | CRS-preserving transforms with deterministic replay |
| **`tiling`** | Sampler, Tiler | Overlap-aware patch extraction with importance sampling |
| **`metrics`** | IoU, Dice, ECE | Calibrated uncertainty with spatial stratification |
| **`export`** | ONNX, TorchScript | Production deployment with optimized inference |

---

## Mathematical Foundations

Ununennium treats geospatial machine learning with rigorous mathematical foundations.

### Spectral Index Computation

The Normalized Difference Vegetation Index (NDVI) is computed as:

$$
\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{Red}}}{\rho_{\text{NIR}} + \rho_{\text{Red}}}
$$

where $\rho$ denotes surface reflectance in the respective spectral band.

### Segmentation Metrics

Intersection over Union with proper handling of spatial autocorrelation:

$$
\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}
$$

### Physics-Informed Loss

Combined data fidelity with PDE residual constraints:

$$
\mathcal{L}_{\text{PINN}} = \underbrace{\frac{1}{N_d}\sum_{i=1}^{N_d}\|u(x_i) - u_i^{\text{obs}}\|^2}_{\text{Data Loss}} + \lambda \underbrace{\frac{1}{N_c}\sum_{j=1}^{N_c}\|\mathcal{N}[u](x_j)\|^2}_{\text{PDE Residual}}
$$

where $\mathcal{N}$ is the differential operator defining the governing equation.

---

## Performance Benchmarks

Benchmarks conducted on NVIDIA A100 80GB, PyTorch 2.1, CUDA 12.1:

| Model | Input Size | Batch | Throughput | Memory | mIoU |
|-------|------------|-------|------------|--------|------|
| U-Net ResNet-50 | 512 x 512 x 12 | 16 | 142 img/s | 12.4 GB | 0.78 |
| U-Net EfficientNet-B4 | 512 x 512 x 12 | 16 | 98 img/s | 14.2 GB | 0.81 |
| DeepLabV3+ ResNet-101 | 512 x 512 x 12 | 12 | 76 img/s | 16.8 GB | 0.82 |
| ViT-L/16 | 224 x 224 x 12 | 32 | 256 img/s | 18.1 GB | 0.83 |
| Pix2Pix (SAR to Optical) | 256 x 256 x 2 | 8 | 67 img/s | 8.6 GB | N/A |
| ESRGAN (4x) | 64 x 64 x 12 | 16 | 124 img/s | 6.2 GB | N/A |

---

## Installation

### Standard Installation

```bash
# Core installation (minimal dependencies)
pip install ununennium

# With geospatial dependencies (rasterio, pyproj, shapely)
pip install "ununennium[geo]"

# Full installation with all features
pip install "ununennium[all]"
```

### Development Installation

```bash
git clone https://github.com/olaflaitinen/ununennium.git
cd ununennium
pip install -e ".[dev]"
pre-commit install
```

### Requirements

| Dependency | Minimum Version | Purpose |
|------------|-----------------|---------|
| Python | 3.10+ | Runtime environment |
| PyTorch | 2.0+ | Deep learning backend |
| NumPy | 1.24+ | Numerical operations |
| rasterio | 1.3+ | Geospatial I/O (optional) |
| GDAL | 3.4+ | Coordinate transformations (optional) |

---

## Quick Start

### Load Satellite Imagery with CRS Preservation

```python
import ununennium as uu

# Read with automatic CRS detection and metadata preservation
tensor = uu.io.read_geotiff("sentinel2_l2a.tif")

print(f"Shape: {tensor.shape}")           # (12, 10980, 10980)
print(f"CRS: {tensor.crs}")               # EPSG:32632
print(f"Resolution: {tensor.resolution}") # (10.0, 10.0) meters
print(f"Bounds: {tensor.bounds}")         # Geographic extent
```

### Train a Semantic Segmentation Model

```python
from ununennium.models import create_model
from ununennium.training import Trainer, CheckpointCallback
from ununennium.losses import DiceLoss
import torch

# Create U-Net with ResNet-50 backbone
model = create_model(
    "unet_resnet50",
    in_channels=12,      # Sentinel-2 bands
    num_classes=10,      # Land cover classes
)

# Configure training with mixed precision
trainer = Trainer(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    loss_fn=DiceLoss(),
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=[
        CheckpointCallback("checkpoints/", monitor="val_iou"),
    ],
    mixed_precision=True,
    gradient_accumulation_steps=4,
)

# Train with progress tracking
history = trainer.fit(epochs=100)
```

### Physics-Informed Neural Networks

```python
from ununennium.models.pinn import PINN, DiffusionEquation, MLP, UniformSampler

# Define governing PDE (heat diffusion)
equation = DiffusionEquation(diffusivity=0.1)

# Create neural network approximator
network = MLP(
    layers=[2, 128, 128, 128, 1],  # (x, t) -> u
    activation="tanh",
)

# Create PINN with physics constraint
pinn = PINN(
    network=network,
    equation=equation,
    lambda_pde=10.0,
)

# Sample collocation points
sampler = UniformSampler(bounds=[(0, 1), (0, 1)], n_points=10000)
x_collocation = sampler.sample()

# Compute combined loss
losses = pinn.compute_loss(x_data, u_data, x_collocation)
total_loss = losses["data"] + losses["pde"]
```

### Image-to-Image Translation with GANs

```python
from ununennium.models.gan import Pix2Pix

# SAR to Optical translation
model = Pix2Pix(
    in_channels=2,       # VV, VH polarizations
    out_channels=3,      # RGB
    ngf=64,              # Generator feature maps
    ndf=64,              # Discriminator feature maps
    n_layers=3,          # PatchGAN depth
)

# Forward pass
fake_optical = model.generator(sar_input)

# Compute adversarial and reconstruction losses
g_loss, d_loss = model.compute_loss(sar_input, real_optical)
```

---

## Architecture

```
ununennium/
├── core/                 # GeoTensor, GeoBatch, types, CRS handling
├── io/                   # COG, STAC, Zarr readers/writers
├── preprocessing/        # Normalization, spectral indices, cloud masking
├── augmentation/         # Geometric and radiometric transforms
├── tiling/               # Spatial sampling and tiling strategies
├── datasets/             # Dataset abstractions and data loaders
├── models/               # Model architectures and registry
│   ├── backbones/        # ResNet, EfficientNet, ViT, Swin
│   ├── heads/            # Classification, Segmentation, Detection
│   ├── architectures/    # U-Net, DeepLabV3+, FPN
│   ├── gan/              # Pix2Pix, CycleGAN, ESRGAN
│   └── pinn/             # Physics-informed networks
├── losses/               # Dice, Focal, Perceptual, Physics losses
├── metrics/              # IoU, Dice, Calibration, Detection metrics
├── training/             # Trainer, Callbacks, Distributed training
├── export/               # ONNX, TorchScript export utilities
└── sensors/              # Sentinel-2, Landsat, MODIS specifications
```

---

## Supported Tasks

| Task | Models | Metrics | Use Cases |
|------|--------|---------|-----------|
| **Scene Classification** | ResNet, EfficientNet, ViT | Accuracy, F1, AUC | Land use, event detection |
| **Semantic Segmentation** | U-Net, DeepLabV3+, FPN | mIoU, Dice, PA | Land cover mapping |
| **Object Detection** | RetinaNet, Faster R-CNN | mAP, AP50, AR | Building detection |
| **Change Detection** | Siamese, Early/Late Fusion | F1, Kappa, OA | Deforestation monitoring |
| **Super-Resolution** | ESRGAN, Real-ESRGAN | PSNR, SSIM, LPIPS | Resolution enhancement |
| **Image Translation** | Pix2Pix, CycleGAN | FID, KID, SAM | SAR-to-optical |
| **Physics-Informed** | PINN, DeepONet | L2 Error, PDE Residual | SST interpolation |

---

## Documentation

Comprehensive documentation is available in the [docs/](docs/README.md) directory:

- **[Architecture](docs/architecture/overview.md)** - System design and data flow
- **[API Reference](docs/api/overview.md)** - Complete API documentation
- **[Tutorials](docs/tutorials/00_quickstart.md)** - Step-by-step guides
- **[Guides](docs/guides/datasets-and-splits.md)** - Best practices
- **[Research](docs/research/math-foundations.md)** - Mathematical foundations

---

## Authors

- **Olaf Yunus Laitinen Imanov** - Lead Architect and Creator
- **Hafiz Rzazade** - Contributor
- **Laman Mamedova** - Contributor
- **Farid Mirzaliyev** - Contributor
- **Ayan Ajili** - Contributor

---

## Citation

If you use Ununennium in your research, please cite:

```bibtex
@software{ununennium2025,
  title = {Ununennium: Production-grade Satellite Imagery Machine Learning},
  author = {Laitinen Imanov, Olaf Yunus and Rzazade, Hafiz and Mamedova, Laman and Mirzaliyev, Farid and Ajili, Ayan},
  year = {2025},
  version = {1.0.5},
  url = {https://github.com/olaflaitinen/ununennium},
  note = {Production-grade Python library for Earth observation machine learning}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full license text.

This license permits commercial use, modification, distribution, patent use, and private use, provided that proper attribution is given and the license and copyright notice are included in all copies or substantial portions of the software.

---

## Contributing

We welcome contributions from the community. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Code style and formatting (Ruff, type hints)
- Testing requirements (pytest, coverage thresholds)
- Documentation standards
- Pull request process

---

## Support

- **Issues**: [GitHub Issues](https://github.com/olaflaitinen/ununennium/issues)
- **Discussions**: [GitHub Discussions](https://github.com/olaflaitinen/ununennium/discussions)
- **Security**: See [SECURITY.md](SECURITY.md) for vulnerability reporting

---

<div align="center">

**Built for the era of petabyte-scale Earth observation.**

</div>
