# API Reference: Evaluation

Tools for model evaluation with proper spatial validation methodology.

---

## Spatial Validation

### SpatialKFold

K-Fold cross-validation with spatial blocking to prevent data leakage.

```python
from ununennium.evaluation import SpatialKFold

splitter = SpatialKFold(
    n_splits=5,
    block_size=1000,  # meters
    buffer_distance=500,  # buffer between folds
)

for train_idx, val_idx in splitter.split(dataset):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    # Train and evaluate
```

### BufferedSplit

Create train/val/test splits with spatial buffers.

```python
from ununennium.evaluation import BufferedSplit

splitter = BufferedSplit(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    buffer_distance=1000,
)

train_idx, val_idx, test_idx = splitter.split(dataset)
```

---

## Metrics Aggregation

### MetricCollection

Compute multiple metrics in a single pass.

```python
from ununennium.evaluation import MetricCollection
from ununennium.metrics import IoU, DiceCoefficient, PixelAccuracy

metrics = MetricCollection([
    IoU(num_classes=10),
    DiceCoefficient(num_classes=10),
    PixelAccuracy(),
])

# Update with batches
for pred, target in zip(predictions, targets):
    metrics.update(pred, target)

# Compute all metrics
results = metrics.compute()
# {"IoU": tensor, "Dice": tensor, "PixelAccuracy": float}
```

---

## Benchmark Suite

### run_benchmark

Comprehensive model benchmarking.

```python
from ununennium.evaluation import run_benchmark

results = run_benchmark(
    model=model,
    test_loader=test_loader,
    metrics=["mIoU", "Dice", "PixelAccuracy"],
    device="cuda",
    num_runs=3,  # For timing stability
)

print(f"mIoU: {results['mIoU']:.4f}")
print(f"Throughput: {results['throughput']:.1f} img/s")
print(f"Memory: {results['memory_mb']:.1f} MB")
```

---

## Reporting

### generate_report

Create evaluation reports with visualizations.

```python
from ununennium.evaluation import generate_report

report = generate_report(
    model=model,
    test_loader=test_loader,
    class_names=["Water", "Forest", "Urban"],
    output_dir="reports/",
    include_confusion_matrix=True,
    include_per_class_metrics=True,
    include_sample_predictions=10,
)
```

---

## API Reference

::: ununennium.evaluation
