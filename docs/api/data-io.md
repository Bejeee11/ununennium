# Data I/O API

## Reading Data

### `read_window`

```python
ununennium.io.read_window(path: str, window: Window) -> GeoTensor
```

Read a specific window from a raster file.

### `stream_from_cog`

```python
ununennium.io.stream_from_cog(url: str, chunk_size: int = 512) -> Iterator[GeoTensor]
```

Stream tiles from a Cloud Optimized GeoTIFF.

## STAC Integration

Utilities for interacting with Spatio-Temporal Asset Catalogs.

### `STACReader`

```python
class STACReader:
    def search(bbox, datetime, collections) -> ItemCollection
    def load_item(item, bands) -> GeoTensor
```

## Writing Data

### `write_geotiff`

```python
ununennium.io.write_geotiff(
    path: str, 
    tensor: GeoTensor, 
    profile: dict = None
)
```

Write a GeoTensor to disk as a GeoTIFF.
