# API Reference: Preprocessing

Data preprocessing and normalization utilities for satellite imagery.

---

## Spectral Indices

### NDVI (Normalized Difference Vegetation Index)

$$NDVI = \frac{NIR - Red}{NIR + Red}$$

```python
from ununennium.preprocessing import compute_ndvi

ndvi = compute_ndvi(
    tensor,
    nir_band=7,  # Sentinel-2 B08
    red_band=3,  # Sentinel-2 B04
)
```

### EVI (Enhanced Vegetation Index)

$$EVI = G \cdot \frac{NIR - Red}{NIR + C_1 \cdot Red - C_2 \cdot Blue + L}$$

```python
from ununennium.preprocessing import compute_evi

evi = compute_evi(
    tensor,
    nir_band=7,
    red_band=3,
    blue_band=1,
)
```

### NDWI (Normalized Difference Water Index)

$$NDWI = \frac{Green - NIR}{Green + NIR}$$

```python
from ununennium.preprocessing import compute_ndwi

ndwi = compute_ndwi(tensor, green_band=2, nir_band=7)
```

### SAVI (Soil Adjusted Vegetation Index)

$$SAVI = \frac{(NIR - Red)(1 + L)}{NIR + Red + L}$$

```python
from ununennium.preprocessing import compute_savi

savi = compute_savi(tensor, nir_band=7, red_band=3, L=0.5)
```

---

## Normalization

### MinMaxNormalization

Scale values to [0, 1] range.

```python
from ununennium.preprocessing import MinMaxNormalization

normalizer = MinMaxNormalization(
    min_val=0,
    max_val=10000,  # Sentinel-2 reflectance range
)
normalized = normalizer(tensor)
```

### StandardNormalization

Z-score normalization with per-channel statistics.

```python
from ununennium.preprocessing import StandardNormalization

normalizer = StandardNormalization(
    mean=[0.485, 0.456, 0.406],  # ImageNet stats
    std=[0.229, 0.224, 0.225],
)
normalized = normalizer(tensor)
```

### SensorNormalization

Sensor-specific normalization using predefined statistics.

```python
from ununennium.preprocessing import SensorNormalization

normalizer = SensorNormalization(
    sensor="sentinel2",  # or "landsat8", "modis"
    bands="all",
)
normalized = normalizer(tensor)
```

---

## Cloud Masking

### CloudMask

Detect and mask cloudy pixels.

```python
from ununennium.preprocessing import CloudMask

masker = CloudMask(
    method="fmask",  # or "sen2cor", "threshold"
    cloud_prob_threshold=0.3,
)
mask = masker(tensor)
clean_tensor = tensor * (1 - mask)
```

---

## Atmospheric Correction

### TOAtoSR

Convert Top-of-Atmosphere reflectance to Surface Reflectance.

```python
from ununennium.preprocessing import TOAtoSR

corrector = TOAtoSR(
    method="dos",  # Dark Object Subtraction
    sensor="sentinel2",
)
sr = corrector(toa_tensor, metadata)
```

---

## Compositing

### TemporalComposite

Create cloud-free composites from time series.

```python
from ununennium.preprocessing import TemporalComposite

compositor = TemporalComposite(
    method="median",  # or "max_ndvi", "best_pixel"
)
composite = compositor(time_series_stack)
```

---

## API Reference

::: ununennium.preprocessing
