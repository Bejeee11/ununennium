# I/O Module API Reference

The `ununennium.io` module provides high-performance, cloud-native access to geospatial raster data. It abstracts GDAL complexity and optimizes for deep learning throughput.

## 1. Readers

### `read_geotiff`

Reads a GeoTIFF file (local or S3) into a `GeoTensor`.

```python
def read_geotiff(
    path: str | Path,
    bands: list[int] | None = None,
    window: Window | None = None,
    out_shape: tuple[int, int] | None = None,
    resampling: Resampling = Resampling.bilinear
) -> GeoTensor
```

**Parameters:**
*   `path` *(str)*: Path to file. Supports VSI (e.g., `/vsis3/bucket/key`).
*   `bands` *(list[int])*: 1-based indices of bands to read. Defaults to all.
*   `window` *(Window)*: Rasterio window for lazy reading of subsets.
*   `out_shape` *(tuple)*: Target $(H, W)$ for on-the-fly resizing.
*   `resampling`: Kernel to use if `out_shape` differs from window size.

**Performance Note:**
Uses `rasterio.Env` with `GDAL_DISABLE_READDIR_ON_OPEN` for optimized cloud performance.

---

### `read_stac`

Resolves and reads assets from a SpatioTemporal Asset Catalog (STAC) Item.

```python
def read_stac(
    item: pystac.Item,
    assets: list[str] = ["red", "green", "blue"],
    resolution: float | None = None
) -> GeoTensor
```

**Features:**
*   **Automatic Resampling**: If assets have different native resolutions (e.g., Sentinel-2 10m/20m bands), they are upsampled to the finest resolution automatically.
*   **Stacking**: Returns a single multi-channel tensor.

---

### `read_zarr`

Reads N-dimensional array data from a Zarr data store.

```python
def read_zarr(
    store: str | zarr.Store,
    selection: tuple[slice, ...] | None = None
) -> GeoTensor
```

**Use Case:**
Ideal for datacube access (Time Series) where temporal slicing is efficient.

---

## 2. Writers

### `write_geotiff`

Persists a `GeoTensor` to disk.

```python
def write_geotiff(
    path: str | Path,
    tensor: GeoTensor,
    profile: dict | None = None,
    tiled: bool = True,
    compress: str = "deflate"
) -> None
```

**Parameters:**
*   `path`: Destination path.
*   `tensor`: Data to write. Must have `crs` and `transform` attributes.
*   `profile`: Rasterio creation options (e.g., non-default block sizes).
*   `tiled` *(bool)*: Whether to create internal tiles (512x512). **Critical for performance** in subsequent reads.
*   `compress` *(str)*: Compression algorithm (`lzw`, `deflate`, `zstd`).

---

## 3. Exception Handling

The module defines specific exceptions for robust error handling in pipelines.

*   `RasterNotFoundError`: File does not exist or is inaccessible.
*   `CRSMismatchError`: Attempting to stack bands with different projections.
*   `BandIndexError`: Requesting band N in an M-band image (where N > M).
