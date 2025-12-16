# Train/Val/Test Splitting

Spatial autocorrelation invalidates random shuffling for splits.

## Spatial Cross-Validation

We must split data by contiguous blocks to ensure independence.

```python
from ununennium.data import SpatialKFold

splitter = SpatialKFold(n_splits=5, buffer_radius=100)

for train_idx, val_idx in splitter.split(coords):
    # Train coordinates: coords[train_idx]
    # Val coordinates: coords[val_idx]
```

### Buffer Zones

To strictly prevent leakage, we define a buffer zone between train and validation sets.

```math
\text{dist}(S_{train}, S_{val}) > R_{corr}
```

where $R_{corr}$ is the spatial autocorrelation range (e.g., from Variogram analysis).
