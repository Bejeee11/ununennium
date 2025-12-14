# Core Module API Reference

The `nunennium.core` module establishes the fundamental data structures for geospatial deep learning. It enforces coordinate reference system (CRS) awareness and bounds tracking throughout the tensor lifecycle.

## 1. GeoTensor

**`ununennium.core.geotensor.GeoTensor`**

A subclass of `torch.Tensor` that carries geospatial metadata. It behaves exactly like a standard PyTorch tensor but persists metadata through operations.

### Signature

```python
class GeoTensor(torch.Tensor):
    def __new__(cls, 
                data: ArrayLike, 
                crs: str | CRS, 
                transform: Affine, 
                bounds: Bounds | None = None) -> GeoTensor
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `crs` | `pyproj.CRS` | The Coordinate Reference System of the data. |
| `transform` | `affine.Affine` | The affine transformation matrix mapping pixel coordinates to map coordinates. |
| `bounds` | `BoundingBox` | The spatial extent (left, bottom, right, top). |
| `resolution` | `tuple[float, float]` | Pixel resolution $(dx, dy)$ derived from transform. |

### Methods

#### `reproject`

Reprojects the tensor to a new CRS.

```python
def reproject(self, dst_crs: str, resampling: Resampling = Resampling.bilinear) -> GeoTensor
```

*   **dst_crs**: Target EPSG code or WKT.
*   **resampling**: Interpolation kernel (see Resampling Theory).
*   **Returns**: New `GeoTensor` in target CRS.
*   **Raises**: `CRSError` if transformation is invalid.

#### `crop`

Spatial crop based on world coordinates.

```python
def crop(self, bbox: BoundingBox) -> GeoTensor
```

---

## 2. GeoBatch

**`ununennium.core.geobatch.GeoBatch`**

A container for a batch of `GeoTensor`s, ensuring that a batch collated from different locations maintains spatial context.

### Signature

```python
@dataclass
class GeoBatch:
    images: torch.Tensor
    masks: torch.Tensor | None
    metadatas: list[dict]
```

### Usage Pattern

```python
# In a DataLoader collation function
def collate_fn(batch):
    images = torch.stack([x["image"] for x in batch])
    metas = [x["metadata"] for x in batch]
    return GeoBatch(images=images, metadatas=metas)
```

---

## 3. Coordinate Reference Systems (CRS)

**`ununennium.core.crs.CRS`**

Wrapper around `pyproj.CRS` providing helper methods for deep learning compatibility.

*   `is_geographic`: Checks if units are degrees.
*   `is_projected`: Checks if units are meters.
*   `to_wkt()`: Returns Well-Known Text representation.
*   `to_epsg()`: Returns EPSG code integer.

### Best Practices

> [!IMPORTANT]
> **Always use Projected CRS for Training.**
> Convolutional kernels assume Euclidean distance. Training on Lat/Lon (degrees) grids causes spatial distortion variance as latitude changes. Ununennium will emit a warning if you train on a Geographic CRS.
