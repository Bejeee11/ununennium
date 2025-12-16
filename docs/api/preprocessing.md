# Preprocessing API

## Spectral Indices

### `compute_ndvi`

```python
def compute_ndvi(nir: torch.Tensor, red: torch.Tensor) -> torch.Tensor
```

Computes Normalized Difference Vegetation Index.

```math
\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}
```

### `compute_ndwi`

```python
def compute_ndwi(green: torch.Tensor, nir: torch.Tensor) -> torch.Tensor
```

Computes Normalized Difference Water Index.

```math
\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}
```

## Normalization

### `min_max_scale`

```python
def min_max_scale(
    tensor: torch.Tensor, 
    min_val: float, 
    max_val: float
) -> torch.Tensor
```

Scale values to [0, 1].

### `standardize`

```python
def standardize(
    tensor: torch.Tensor, 
    mean: list[float], 
    std: list[float]
) -> torch.Tensor
```

Z-score normalization.

```math
z = \frac{x - \mu}{\sigma}
```
