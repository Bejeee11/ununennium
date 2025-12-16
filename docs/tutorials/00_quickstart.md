# Quickstart Tutorial

This guide covers the end-to-end workflow from data loading to model training and inference.

## 1. Installation

```bash
pip install ununennium[geo]
```

## 2. Loading Data

We'll load a sample multispectral image.

```python
from ununennium.core import GeoTensor
from ununennium.io import read_window

# Read a 512x512 window from a Sentinel-2 scene
path = "data/sentinel2_l2a.tif"
tensor = read_window(path, window=((0, 512), (0, 512)))

print(f"Shape: {tensor.shape}")  # (12, 512, 512)
print(f"CRS: {tensor.crs}")      # EPSG:32631
```

## 3. Preprocessing

Compute indices and normalize.

```python
from ununennium.preprocessing import compute_ndvi, min_max_scale

# Bands: 0=Blue, 1=Green, 2=Red, 3=NIR
red = tensor[2]
nir = tensor[3]

ndvi = compute_ndvi(nir, red)
tensor_norm = min_max_scale(tensor, 0, 10000)
```

## 4. Model Training

Train a U-Net on a small dataset.

```python
from ununennium.models import UNet
from ununennium.training import Trainer

model = UNet(in_channels=12, classes=1)
trainer = Trainer(model=model, accelerator="gpu")

# Assume train_loader is defined
trainer.fit(train_loader, epochs=10)
```

## 5. Inference

Predict on new data.

```python
pred = model(tensor_norm.unsqueeze(0))
```
