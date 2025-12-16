# Core API

## GeoTensor

`ununennium.core.GeoTensor`

The core data structure extending `torch.Tensor`.

### Constructor

```python
GeoTensor(data, affine, crs)
```

**Parameters:**
- `data` (torch.Tensor): The raw tensor data.
- `affine` (rasterio.Affine): Affine transform matrix.
- `crs` (CRS): Coordinate reference system.

### Properties

- `width`, `height`: Spatial dimensions.
- `bounds`: Bounding box `(minx, miny, maxx, maxy)`.
- `res`: Resolution `(x_res, y_res)`.

### Methods

#### `reproject`

```python
def reproject(dst_crs: CRS, dst_res: float = None) -> GeoTensor
```

Reproject tensor to a new CRS.

#### `crop`

```python
def crop(bbox: BBox) -> GeoTensor
```

Crop the tensor to a bounding box.

## GeoBatch

A container for a batch of `GeoTensor` objects, handling collation and scatter/gather operations.
