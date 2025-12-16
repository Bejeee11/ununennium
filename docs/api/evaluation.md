# Evaluation API

## Metrics

### `IoU` (Intersection over Union)

```python
class IoU(Metric):
    def update(self, pred, target)
    def compute() -> float
```

Formula:

```math
\text{IoU} = \frac{\text{Intersection}}{\text{Union}}
```

### `F1Score` (Dice Coefficient)

Harmonic mean of precision and recall.

```math
F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
```

### `ConfusionMatrix`

Computes the $C \times C$ confusion matrix for $C$ classes.

## Uncertainty

### `ExpectedCalibrationError`

Measures how well predicted probabilities match accuracy.

```math
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|
```
