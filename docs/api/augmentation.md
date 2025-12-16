# API Reference: Augmentation

Data augmentation transforms designed for geospatial imagery that preserve CRS and spatial metadata.

---

## Geometric Transforms

### RandomRotation

Rotate image by a random angle while preserving geospatial coordinates.

```python
from ununennium.augmentation import RandomRotation

transform = RandomRotation(
    degrees=(-30, 30),
    interpolation="bilinear",
    preserve_crs=True,
)
augmented = transform(geotensor)
```

### RandomFlip

Horizontal and vertical flips with proper coordinate transformation.

```python
from ununennium.augmentation import RandomHorizontalFlip, RandomVerticalFlip

transform = RandomHorizontalFlip(p=0.5)
```

### RandomCrop

Extract random crops with spatial bounds tracking.

```python
from ununennium.augmentation import RandomCrop

transform = RandomCrop(size=(256, 256))
cropped, new_bounds = transform(geotensor)
```

### RandomAffine

Combined rotation, translation, and scaling.

```python
from ununennium.augmentation import RandomAffine

transform = RandomAffine(
    degrees=15,
    translate=(0.1, 0.1),
    scale=(0.9, 1.1),
)
```

---

## Radiometric Transforms

### RandomBrightnessContrast

Adjust brightness and contrast for optical imagery.

```python
from ununennium.augmentation import RandomBrightnessContrast

transform = RandomBrightnessContrast(
    brightness_limit=0.2,
    contrast_limit=0.2,
    p=0.5,
)
```

### GaussianNoise

Add sensor-realistic noise.

```python
from ununennium.augmentation import GaussianNoise

transform = GaussianNoise(
    mean=0.0,
    std=0.02,
    per_channel=True,
)
```

### SpeckleNoise

SAR-specific multiplicative noise augmentation.

```python
from ununennium.augmentation import SpeckleNoise

transform = SpeckleNoise(intensity=0.1)
```

---

## Spectral Transforms

### BandDropout

Randomly drop spectral bands during training.

```python
from ununennium.augmentation import BandDropout

transform = BandDropout(
    drop_prob=0.1,
    fill_value=0.0,
)
```

### SpectralShift

Simulate sensor calibration variations.

```python
from ununennium.augmentation import SpectralShift

transform = SpectralShift(shift_range=(-0.05, 0.05))
```

---

## Compose

Chain multiple transforms together.

```python
from ununennium.augmentation import Compose, RandomRotation, RandomFlip, GaussianNoise

transform = Compose([
    RandomRotation(degrees=15),
    RandomFlip(p=0.5),
    GaussianNoise(std=0.01),
])

augmented = transform(geotensor, mask=segmentation_mask)
```

---

## Usage with DataLoader

```python
from ununennium.datasets import GeoDataset
from ununennium.augmentation import Compose, RandomRotation, RandomFlip

train_transforms = Compose([
    RandomRotation(degrees=30),
    RandomFlip(p=0.5),
])

dataset = GeoDataset(
    images_dir="data/images",
    masks_dir="data/masks",
    transform=train_transforms,
)
```

---

## API Reference

::: ununennium.augmentation.geometric
::: ununennium.augmentation.compose
