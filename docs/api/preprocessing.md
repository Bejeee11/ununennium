# Preprocessing API

The preprocessing module provides spectral index computation, normalization, and radiometric operations.

---

## Spectral Indices

### Common Indices

| Index | Formula | Description |
|-------|---------|-------------|
| NDVI | $\frac{NIR - Red}{NIR + Red}$ | Vegetation index |
| NDWI | $\frac{Green - NIR}{Green + NIR}$ | Water index |
| NBR | $\frac{NIR - SWIR}{NIR + SWIR}$ | Burn ratio |
| EVI | $2.5 \cdot \frac{NIR - Red}{NIR + 6 \cdot Red - 7.5 \cdot Blue + 1}$ | Enhanced vegetation |

### compute_index

```python
def compute_index(
    data: torch.Tensor | GeoTensor,
    index: str,
    bands: dict[str, int],
    epsilon: float = 1e-8,
) -> torch.Tensor | GeoTensor:
    """Compute spectral index.
    
    Parameters
    ----------
    data : torch.Tensor | GeoTensor
        Input data with shape (C, H, W) or (B, C, H, W).
    index : str
        Index name: "ndvi", "ndwi", "nbr", "evi", "savi".
    bands : dict[str, int]
        Mapping of band names to indices (0-indexed).
    epsilon : float, optional
        Small value to prevent division by zero.
    
    Returns
    -------
    torch.Tensor | GeoTensor
        Computed index with shape (H, W) or (B, H, W).
    """
```

### Example

```python
from ununennium.preprocessing import compute_index

# Sentinel-2 band mapping
bands = {"red": 3, "nir": 7, "green": 2, "blue": 1, "swir": 11}

ndvi = compute_index(tensor, "ndvi", bands)
print(f"NDVI range: [{ndvi.min():.2f}, {ndvi.max():.2f}]")
```

---

## Normalization

### normalize

```python
def normalize(
    data: torch.Tensor | GeoTensor,
    method: str = "minmax",
    per_channel: bool = True,
    **kwargs,
) -> torch.Tensor | GeoTensor:
    """Normalize data.
    
    Parameters
    ----------
    data : torch.Tensor | GeoTensor
        Input data.
    method : str, optional
        Normalization method. Default: "minmax".
    per_channel : bool, optional
        Normalize each channel independently. Default: True.
    **kwargs
        Method-specific parameters.
    
    Methods
    -------
    minmax : Scale to [0, 1]
    zscore : Standardize to mean=0, std=1
    percentile : Scale using percentiles (robust to outliers)
    """
```

### Normalization Methods

| Method | Formula | Parameters |
|--------|---------|------------|
| `minmax` | $\frac{x - \min}{\max - \min}$ | None |
| `zscore` | $\frac{x - \mu}{\sigma}$ | None |
| `percentile` | $\frac{x - p_{low}}{p_{high} - p_{low}}$ | `p=(2, 98)` |

### Example

```python
from ununennium.preprocessing import normalize

# Robust normalization
normalized = normalize(
    tensor,
    method="percentile",
    p=(2, 98),
    per_channel=True,
)
```

---

## Standardization

### standardize

```python
def standardize(
    data: torch.Tensor | GeoTensor,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    sensor: str | None = None,
) -> torch.Tensor | GeoTensor:
    """Standardize data with mean and std.
    
    Parameters
    ----------
    data : torch.Tensor | GeoTensor
        Input data.
    mean : list[float] | None, optional
        Per-channel means.
    std : list[float] | None, optional
        Per-channel stds.
    sensor : str | None, optional
        Sensor name for predefined statistics.
    """
```

### Predefined Statistics

```python
# Use sensor-specific statistics
standardized = standardize(tensor, sensor="sentinel2_l2a")
```

| Sensor | Bands | Source |
|--------|-------|--------|
| `sentinel2_l2a` | 12 | ESA Level-2A |
| `landsat8_c2` | 7 | USGS Collection 2 |
| `modis_mod09ga` | 7 | MODIS daily |

---

## Cloud Masking

### apply_cloud_mask

```python
def apply_cloud_mask(
    data: torch.Tensor | GeoTensor,
    mask: torch.Tensor,
    fill_value: float = float("nan"),
) -> torch.Tensor | GeoTensor:
    """Apply cloud mask to data.
    
    Parameters
    ----------
    data : torch.Tensor | GeoTensor
        Input data.
    mask : torch.Tensor
        Boolean mask (True = cloud).
    fill_value : float, optional
        Value for masked pixels. Default: NaN.
    """
```

### compute_cloud_mask

```python
def compute_cloud_mask(
    data: torch.Tensor | GeoTensor,
    sensor: str,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute cloud mask using sensor-specific algorithm.
    
    Parameters
    ----------
    data : torch.Tensor | GeoTensor
        Input data with required bands.
    sensor : str
        Sensor identifier.
    threshold : float, optional
        Cloud probability threshold. Default: 0.5.
    """
```

---

## Harmonization

### harmonize

```python
def harmonize(
    data: torch.Tensor | GeoTensor,
    source: str,
    target: str,
) -> torch.Tensor | GeoTensor:
    """Harmonize data between sensors.
    
    Parameters
    ----------
    data : torch.Tensor | GeoTensor
        Input data from source sensor.
    source : str
        Source sensor identifier.
    target : str
        Target sensor identifier.
    
    Returns
    -------
    torch.Tensor | GeoTensor
        Harmonized data matching target sensor.
    """
```

### Example

```python
from ununennium.preprocessing import harmonize

# Convert Landsat-8 to Sentinel-2 spectral response
s2_like = harmonize(landsat_tensor, source="landsat8", target="sentinel2")
```
