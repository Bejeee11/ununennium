# Tutorial 00: Quickstart

Get started with Ununennium in 15 minutes. This tutorial covers installation, loading data, creating models, and making predictions.

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)

---

## Step 1: Installation

```bash
# Full installation with all dependencies
pip install "ununennium[all]"

# Verify installation
python -c "import ununennium; print(ununennium.__version__)"
```

---

## Step 2: Load Satellite Imagery

Ununennium's `GeoTensor` preserves coordinate reference system (CRS) information through all operations.

```python
import ununennium as uu

# Load a GeoTIFF file
tensor = uu.io.read_geotiff("path/to/sentinel2.tif")

# Inspect metadata
print(f"Shape: {tensor.shape}")         # (12, 10980, 10980)
print(f"CRS: {tensor.crs}")             # EPSG:32632
print(f"Resolution: {tensor.resolution}")  # (10.0, 10.0) meters
print(f"Bounds: {tensor.bounds}")
```

### Working with Partial Reads

For large files, read only the region you need:

```python
from ununennium.core import BoundingBox

# Define area of interest
aoi = BoundingBox(
    left=500000,
    bottom=4490000,
    right=510000,
    top=4500000,
)

# Read only the AOI
tile = uu.io.read_geotiff("large_file.tif", window=aoi)
```

---

## Step 3: Create a Model

Use the model registry to create architectures:

```python
from ununennium.models import create_model, list_models

# List available models
print(list_models(task="segmentation"))
# ['unet_resnet50', 'unet_efficientnet_b4', 'deeplabv3_resnet101', ...]

# Create U-Net with ResNet-50 backbone
model = create_model(
    "unet_resnet50",
    in_channels=12,      # Sentinel-2 bands
    num_classes=10,      # Land cover classes
)

# Move to GPU
model = model.cuda()
```

---

## Step 4: Make Predictions

Run inference on your data:

```python
import torch

# Prepare input
x = tensor.data.unsqueeze(0).cuda()  # Add batch dimension, move to GPU

# Inference
model.eval()
with torch.no_grad():
    logits = model(x)
    prediction = logits.argmax(dim=1)

print(f"Prediction shape: {prediction.shape}")  # (1, 10980, 10980)
```

---

## Step 5: Compute Spectral Indices

Calculate common vegetation indices:

```python
from ununennium.preprocessing import compute_index

# Sentinel-2 band mapping (0-indexed)
bands = {
    "blue": 1,
    "green": 2,
    "red": 3,
    "nir": 7,
    "swir": 11,
}

# Compute NDVI
ndvi = compute_index(tensor, "ndvi", bands)
print(f"NDVI range: [{ndvi.min():.2f}, {ndvi.max():.2f}]")
```

---

## Step 6: Train a Model

Complete training loop with callbacks:

```python
from ununennium.training import Trainer, CheckpointCallback
from ununennium.losses import DiceLoss
from ununennium.metrics import IoU
import torch

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=DiceLoss(),
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=[
        CheckpointCallback("checkpoints/", monitor="val_iou"),
    ],
    metrics=[IoU(num_classes=10)],
    mixed_precision=True,
)

# Train
history = trainer.fit(epochs=50)

# Access training history
print(f"Final val IoU: {history['val_iou'][-1]:.4f}")
```

---

## Step 7: Export for Production

Export to ONNX for deployment:

```python
from ununennium.export import to_onnx

to_onnx(
    model,
    example_input=torch.randn(1, 12, 512, 512).cuda(),
    output_path="model.onnx",
    opset_version=17,
)
```

---

## Next Steps

- [Tutorial 01: Ingest and Tiling](01_ingest_and_tiling.md) - Data preparation
- [Tutorial 02: Train/Val/Test](02_train_val_test.md) - Proper experiment design
- [API Reference](../api/overview.md) - Complete API documentation
