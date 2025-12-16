# Uncertainty and Calibration

This guide covers uncertainty quantification and calibration for geospatial models.

---

## Types of Uncertainty

### Aleatoric Uncertainty

Irreducible uncertainty from data noise.

### Epistemic Uncertainty

Model uncertainty, reducible with more data.

---

## Calibration

A calibrated model satisfies:

$$
P(Y = y | \hat{p} = p) = p
$$

### Reliability Diagram

```python
from ununennium.visualization import reliability_diagram

reliability_diagram(probabilities, targets, n_bins=10)
plt.savefig("reliability.png")
```

### Temperature Scaling

```python
from ununennium.calibration import TemperatureScaling

calibrator = TemperatureScaling()
calibrator.fit(val_logits, val_targets)

calibrated_probs = calibrator(test_logits)
```

---

## Metrics

### ECE

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

### Brier Score

$$
\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2
$$

---

## MC Dropout

```python
model.train()  # Enable dropout

predictions = []
for _ in range(100):
    with torch.no_grad():
        pred = model(x)
        predictions.append(pred)

mean = torch.stack(predictions).mean(0)
uncertainty = torch.stack(predictions).std(0)
```

---

## See Also

- [Evaluation API](../api/evaluation.md)
- [Metrics Guide](metrics.md)
