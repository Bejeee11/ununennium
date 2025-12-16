# API Reference: Datasets

Dataset classes for loading and managing geospatial training data.

---

## Base Classes

### GeoDataset

Abstract base class for geospatial datasets with CRS-aware loading.

```python
from ununennium.datasets import GeoDataset

class MyDataset(GeoDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        
    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)
        return self.apply_transforms(image, mask)
```

---

## Built-in Datasets

### GeoTiffDataset

Load image-mask pairs from directories of GeoTIFF files.

```python
from ununennium.datasets import GeoTiffDataset

dataset = GeoTiffDataset(
    images_dir="data/images",
    masks_dir="data/masks",
    transform=train_transforms,
    bands=[0, 1, 2, 3],  # Select specific bands
)

image, mask = dataset[0]
print(f"Shape: {image.shape}, CRS: {image.crs}")
```

### STACDataset

Stream data directly from STAC catalogs.

```python
from ununennium.datasets import STACDataset

dataset = STACDataset(
    stac_url="https://earth-search.aws.element84.com/v0",
    collection="sentinel-s2-l2a-cogs",
    bbox=[10.0, 45.0, 11.0, 46.0],
    datetime="2023-01-01/2023-12-31",
    assets=["B02", "B03", "B04", "B08"],
)
```

### ZarrDataset

Efficient loading from Zarr arrays for large datasets.

```python
from ununennium.datasets import ZarrDataset

dataset = ZarrDataset(
    zarr_path="data/satellite.zarr",
    chunk_size=(256, 256),
    overlap=32,
)
```

---

## Synthetic Datasets

### SyntheticDataset

Generate synthetic data for testing and development.

```python
from ununennium.datasets import SyntheticDataset

dataset = SyntheticDataset(
    num_samples=1000,
    num_channels=12,
    image_size=(256, 256),
    num_classes=10,
    noise_level=0.1,
)

# Quick testing
for images, masks in DataLoader(dataset, batch_size=16):
    output = model(images)
    loss = loss_fn(output, masks)
```

### PatternDataset

Generate geometric patterns for segmentation testing.

```python
from ununennium.datasets import PatternDataset

dataset = PatternDataset(
    patterns=["circles", "rectangles", "lines"],
    num_samples=500,
    complexity=0.5,
)
```

---

## Data Loading Utilities

### create_dataloaders

Factory function for train/val/test split with spatial awareness.

```python
from ununennium.datasets import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    dataset=full_dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    batch_size=16,
    spatial_split=True,  # Prevent spatial leakage
    buffer_distance=1000,  # meters
)
```

### CollateFunction

Custom collate for variable-size geospatial data.

```python
from ununennium.datasets import geo_collate_fn

loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=geo_collate_fn,
)
```

---

## API Reference

::: ununennium.datasets.base
::: ununennium.datasets.synthetic
