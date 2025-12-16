# Datasets and Splits

This guide covers best practices for dataset management and train/val/test splitting in geospatial ML.

---

## Spatial Autocorrelation

Nearby pixels are correlated. Random splits cause data leakage.

### Correlation Range

| Land Cover | Typical Range |
|------------|---------------|
| Urban | 100-500 m |
| Agriculture | 500m-2 km |
| Forest | 1-10 km |

### Block Size Rule

$$
\text{block\_size} \geq 2 \times \text{correlation\_range}
$$

---

## Splitting Strategies

### Block-Based Splitting

```python
from ununennium.datasets import BlockSplitter

splitter = BlockSplitter(block_size=(5000, 5000))
train, val, test = splitter.split(bounds, ratios=[0.7, 0.15, 0.15])
```

### Spatial K-Fold

```python
from ununennium.datasets import SpatialKFold

cv = SpatialKFold(n_folds=5, buffer=1000)
for train_idx, val_idx in cv.split(data):
    # Train and evaluate
    pass
```

---

## Class Imbalance

Use stratified spatial sampling:

```python
from ununennium.datasets import StratifiedBlockSampler

sampler = StratifiedBlockSampler(
    class_weights={0: 1.0, 1: 5.0, 2: 10.0},
    block_size=(1000, 1000),
)
```

---

## Temporal Considerations

For change detection, ensure temporal separation:

```python
# Wrong: Same scene in train and val (different tiles)
# Right: Different acquisition dates in train and val
```

---

## See Also

- [Tutorial 02: Train/Val/Test](../tutorials/02_train_val_test.md)
- [Spatial Autocorrelation Theory](../research/math-foundations.md)
