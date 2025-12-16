# Core API

The core module provides foundational data structures for geospatial machine learning.

---

## GeoTensor

CRS-aware tensor that preserves geospatial metadata through operations.

### Class Definition

```python
class GeoTensor:
    """A PyTorch tensor with geospatial metadata.
    
    Parameters
    ----------
    data : torch.Tensor
        The underlying tensor data. Shape: (C, H, W) or (H, W).
    crs : str | CRS
        Coordinate Reference System (e.g., "EPSG:32632").
    transform : Affine
        Affine transform from pixel to world coordinates.
    nodata : float | None, optional
        Value representing no data. Default: None.
    
    Attributes
    ----------
    shape : tuple[int, ...]
        Tensor shape.
    bounds : BoundingBox
        Geographic extent.
    resolution : tuple[float, float]
        Pixel size in CRS units (x, y).
    dtype : torch.dtype
        Data type.
    device : torch.device
        Device location.
    """
```

### Constructor

```python
from ununennium.core import GeoTensor
from affine import Affine
import torch

data = torch.randn(12, 512, 512)
transform = Affine.translation(500000, 4500000) * Affine.scale(10, -10)

tensor = GeoTensor(
    data=data,
    crs="EPSG:32632",
    transform=transform,
    nodata=-9999.0,
)
```

### Methods

#### `to(device)`

Move tensor to specified device.

```python
gpu_tensor = tensor.to("cuda:0")
cpu_tensor = tensor.to("cpu")
```

#### `numpy()`

Convert to NumPy array.

```python
array = tensor.numpy()  # Returns np.ndarray
```

#### `reproject(target_crs, resolution=None, method="bilinear")`

Reproject to different CRS.

```python
wgs84 = tensor.reproject(
    target_crs="EPSG:4326",
    resolution=(0.0001, 0.0001),
    method="bilinear",
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_crs` | `str` | Target CRS identifier |
| `resolution` | `tuple[float, float]` | Output resolution |
| `method` | `str` | Resampling method |

#### `crop(bounds)`

Crop to bounding box.

```python
from ununennium.core import BoundingBox

bounds = BoundingBox(502000, 4496000, 504000, 4498000)
cropped = tensor.crop(bounds)
```

#### `resample(resolution, method="bilinear")`

Resample to new resolution.

```python
resampled = tensor.resample(
    resolution=(20.0, 20.0),
    method="cubic",
)
```

---

## GeoBatch

Batch container for training with consistent metadata.

### Class Definition

```python
class GeoBatch:
    """A batch of samples for training.
    
    Parameters
    ----------
    images : torch.Tensor
        Image batch. Shape: (B, C, H, W).
    masks : torch.Tensor | None, optional
        Label batch. Shape: (B, H, W) or (B, C, H, W).
    crs : str | CRS
        Common CRS for all samples.
    transforms : list[Affine] | None, optional
        Per-sample transforms.
    
    Attributes
    ----------
    batch_size : int
        Number of samples.
    device : torch.device
        Device location.
    """
```

### Constructor

```python
from ununennium.core import GeoBatch
import torch

batch = GeoBatch(
    images=torch.randn(8, 12, 256, 256),
    masks=torch.randint(0, 10, (8, 256, 256)),
    crs="EPSG:32632",
)
```

### Methods

#### `to(device)`

```python
gpu_batch = batch.to("cuda:0")
```

#### `pin_memory()`

Pin memory for faster GPU transfer.

```python
pinned = batch.pin_memory()
```

---

## BoundingBox

Geographic extent container.

### Class Definition

```python
class BoundingBox:
    """Axis-aligned bounding box in CRS coordinates.
    
    Parameters
    ----------
    left : float
        Minimum x coordinate.
    bottom : float
        Minimum y coordinate.
    right : float
        Maximum x coordinate.
    top : float
        Maximum y coordinate.
    """
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `width` | `float` | `right - left` |
| `height` | `float` | `top - bottom` |
| `center` | `tuple[float, float]` | Center point |

### Methods

#### `intersects(other)`

Check if bounding boxes overlap.

```python
if bbox1.intersects(bbox2):
    print("Boxes overlap")
```

#### `intersection(other)`

Compute intersection.

```python
overlap = bbox1.intersection(bbox2)
```

#### `union(other)`

Compute union.

```python
combined = bbox1.union(bbox2)
```

#### `buffer(distance)`

Expand by distance.

```python
expanded = bbox.buffer(100)  # 100 CRS units
```

---

## Type Definitions

```python
from ununennium.core.types import (
    CRS,              # Union[str, pyproj.CRS]
    Transform,        # affine.Affine
    Resolution,       # tuple[float, float]
    Shape,            # tuple[int, ...]
)
```

---

## Exceptions

```python
from ununennium.core.exceptions import (
    CRSError,         # Invalid CRS
    BoundsError,      # Invalid bounds
    TransformError,   # Invalid transform
)
```

### Example Error Handling

```python
from ununennium.core import GeoTensor
from ununennium.core.exceptions import CRSError

try:
    tensor = GeoTensor(data, crs="INVALID")
except CRSError as e:
    print(f"CRS error: {e}")
```
