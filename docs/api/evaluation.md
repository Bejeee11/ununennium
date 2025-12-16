# Evaluation API

The evaluation module provides metrics for model assessment and calibration analysis.

---

## Segmentation Metrics

### IoU (Intersection over Union)

```python
class IoU(Metric):
    """Intersection over Union metric.
    
    Parameters
    ----------
    num_classes : int
        Number of classes.
    ignore_index : int | None, optional
        Class index to ignore. Default: None.
    average : str, optional
        Averaging mode: "micro", "macro", "weighted". Default: "macro".
    """
```

$$
\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}
$$

### Dice Coefficient

```python
class Dice(Metric):
    """Dice coefficient (F1 score).
    
    Parameters
    ----------
    num_classes : int
        Number of classes.
    smooth : float, optional
        Smoothing factor. Default: 1e-6.
    """
```

$$
\text{Dice}_c = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}
$$

### Example

```python
from ununennium.metrics import IoU, Dice

iou = IoU(num_classes=10)
dice = Dice(num_classes=10)

# Update with predictions
iou.update(predictions, targets)
dice.update(predictions, targets)

# Compute final metrics
print(f"mIoU: {iou.compute():.4f}")
print(f"mDice: {dice.compute():.4f}")
```

---

## Classification Metrics

### Accuracy

```python
class Accuracy(Metric):
    """Classification accuracy.
    
    Parameters
    ----------
    task : str
        Task type: "binary", "multiclass", "multilabel".
    num_classes : int | None, optional
        Number of classes (required for multiclass).
    """
```

### F1Score

```python
class F1Score(Metric):
    """F1 score.
    
    Parameters
    ----------
    num_classes : int
        Number of classes.
    average : str, optional
        Averaging: "micro", "macro", "weighted". Default: "macro".
    """
```

---

## Calibration Metrics

### Expected Calibration Error (ECE)

```python
class ECE(Metric):
    """Expected Calibration Error.
    
    Parameters
    ----------
    n_bins : int, optional
        Number of confidence bins. Default: 15.
    """
```

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

where $B_m$ is the set of samples in bin $m$.

### Brier Score

```python
class BrierScore(Metric):
    """Brier score for probabilistic predictions.
    
    Parameters
    ----------
    num_classes : int
        Number of classes.
    """
```

$$
\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} (p_{ic} - y_{ic})^2
$$

### Example

```python
from ununennium.metrics import ECE, BrierScore

ece = ECE(n_bins=15)
brier = BrierScore(num_classes=10)

# probabilities: (N, C) - class probabilities
# targets: (N,) - ground truth class indices
ece.update(probabilities, targets)
brier.update(probabilities, targets)

print(f"ECE: {ece.compute():.4f}")
print(f"Brier: {brier.compute():.4f}")
```

---

## Detection Metrics

### Mean Average Precision (mAP)

```python
class MeanAP(Metric):
    """Mean Average Precision.
    
    Parameters
    ----------
    iou_thresholds : list[float], optional
        IoU thresholds. Default: [0.5, 0.75].
    """
```

---

## Change Detection Metrics

### Kappa Coefficient

```python
class KappaCoefficient(Metric):
    """Cohen's Kappa for change detection.
    
    Parameters
    ----------
    num_classes : int
        Number of classes (including no-change).
    """
```

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where $p_o$ is observed agreement and $p_e$ is expected agreement.

---

## Super-Resolution Metrics

### PSNR

```python
class PSNR(Metric):
    """Peak Signal-to-Noise Ratio."""
```

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)
$$

### SSIM

```python
class SSIM(Metric):
    """Structural Similarity Index."""
```

---

## Metric Aggregation

### MetricCollection

```python
from ununennium.metrics import MetricCollection, IoU, Dice

metrics = MetricCollection([
    IoU(num_classes=10),
    Dice(num_classes=10),
])

# Update all metrics at once
metrics.update(predictions, targets)

# Compute all metrics
results = metrics.compute()
# {"IoU": 0.78, "Dice": 0.85}
```

---

## Bootstrap Confidence Intervals

```python
from ununennium.metrics import bootstrap_ci

# Compute 95% confidence interval
mean, (low, high) = bootstrap_ci(iou, n_bootstrap=1000, confidence=0.95)
print(f"mIoU: {mean:.4f} [{low:.4f}, {high:.4f}]")
```
