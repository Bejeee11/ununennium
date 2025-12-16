# Data I/O API

The I/O module provides readers and writers for geospatial data formats.

---

## GeoTIFF Operations

### read_geotiff

Read a GeoTIFF file as a GeoTensor.

```python
def read_geotiff(
    path: str | Path,
    bands: list[int] | None = None,
    window: BoundingBox | None = None,
    overview_level: int | None = None,
) -> GeoTensor:
    """Read GeoTIFF file.
    
    Parameters
    ----------
    path : str | Path
        Path to GeoTIFF file or URL (http://, s3://).
    bands : list[int] | None, optional
        1-indexed band numbers to read. Default: all bands.
    window : BoundingBox | None, optional
        Spatial window to read. Default: entire image.
    overview_level : int | None, optional
        Overview level (0=full res). Default: auto-select.
    
    Returns
    -------
    GeoTensor
        Loaded data with metadata.
    
    Examples
    --------
    >>> from ununennium.io import read_geotiff
    >>> tensor = read_geotiff("sentinel2.tif")
    >>> tensor.shape
    (12, 10980, 10980)
    """
```

### write_geotiff

Write a GeoTensor to GeoTIFF.

```python
def write_geotiff(
    tensor: GeoTensor,
    path: str | Path,
    compress: str = "lzw",
    tiled: bool = True,
    blocksize: int = 512,
    cog: bool = True,
) -> None:
    """Write GeoTensor to GeoTIFF.
    
    Parameters
    ----------
    tensor : GeoTensor
        Data to write.
    path : str | Path
        Output path.
    compress : str, optional
        Compression method. Default: "lzw".
    tiled : bool, optional
        Write as tiled TIFF. Default: True.
    blocksize : int, optional
        Tile size. Default: 512.
    cog : bool, optional
        Write as Cloud Optimized GeoTIFF. Default: True.
    """
```

---

## STAC Client

Query SpatioTemporal Asset Catalog (STAC) APIs.

### Class Definition

```python
class STACClient:
    """Client for STAC API queries.
    
    Parameters
    ----------
    url : str
        STAC API endpoint URL.
    headers : dict | None, optional
        Additional HTTP headers.
    """
```

### Methods

#### `search`

```python
def search(
    self,
    bbox: tuple[float, float, float, float] | None = None,
    datetime: str | None = None,
    collections: list[str] | None = None,
    query: dict | None = None,
    limit: int = 100,
) -> Iterator[Item]:
    """Search STAC catalog.
    
    Parameters
    ----------
    bbox : tuple, optional
        Bounding box (west, south, east, north).
    datetime : str, optional
        Temporal query (e.g., "2023-01-01/2023-12-31").
    collections : list[str], optional
        Collection IDs to search.
    query : dict, optional
        Additional query parameters.
    limit : int, optional
        Maximum items per page. Default: 100.
    
    Yields
    ------
    Item
        STAC items matching query.
    """
```

### Example

```python
from ununennium.io import STACClient

client = STACClient("https://earth-search.aws.element84.com/v1")

items = client.search(
    bbox=(9.0, 45.0, 10.0, 46.0),
    datetime="2023-06-01/2023-06-30",
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 20}},
)

for item in items:
    tensor = client.read_item(item, assets=["B04", "B03", "B02"])
    print(f"Loaded: {tensor.shape}")
```

---

## Zarr Operations

### read_zarr

```python
def read_zarr(
    path: str | Path,
    chunks: tuple[int, ...] | None = None,
) -> GeoTensor:
    """Read Zarr store as GeoTensor.
    
    Parameters
    ----------
    path : str | Path
        Path to Zarr store.
    chunks : tuple, optional
        Chunk size for lazy loading.
    
    Returns
    -------
    GeoTensor
        Lazy-loaded data.
    """
```

### write_zarr

```python
def write_zarr(
    tensor: GeoTensor,
    path: str | Path,
    chunks: tuple[int, ...] = (1, 512, 512),
    compressor: str = "zstd",
) -> None:
    """Write GeoTensor to Zarr store.
    
    Parameters
    ----------
    tensor : GeoTensor
        Data to write.
    path : str | Path
        Output path.
    chunks : tuple, optional
        Chunk size. Default: (1, 512, 512).
    compressor : str, optional
        Compression codec. Default: "zstd".
    """
```

---

## Streaming Operations

### stream_geotiff

```python
def stream_geotiff(
    path: str | Path,
    chunk_size: int = 1024,
    overlap: int = 0,
) -> Iterator[GeoTensor]:
    """Stream large GeoTIFF in chunks.
    
    Parameters
    ----------
    path : str | Path
        Path to GeoTIFF.
    chunk_size : int, optional
        Chunk size in pixels. Default: 1024.
    overlap : int, optional
        Overlap between chunks. Default: 0.
    
    Yields
    ------
    GeoTensor
        Chunk with metadata.
    """
```

### Example

```python
from ununennium.io import stream_geotiff

for chunk in stream_geotiff("large_raster.tif", chunk_size=512):
    # Process chunk
    result = model(chunk.data.unsqueeze(0))
```

---

## Format Support

| Format | Read | Write | Streaming | Cloud-Native |
|--------|------|-------|-----------|--------------|
| GeoTIFF | Yes | Yes | Yes | Partial |
| COG | Yes | Yes | Yes | Yes |
| Zarr | Yes | Yes | Yes | Yes |
| NetCDF | Yes | No | Partial | No |
| HDF5 | Yes | No | Partial | No |
