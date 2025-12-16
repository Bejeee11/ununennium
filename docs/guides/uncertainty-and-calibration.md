# Uncertainty and Calibration

Models should know when they don't know.

## Probability Calibration

A calibrated model's confidence matches its accuracy. If a model says "80% confidence", it should be correct 80% of the time.

### Reliability Diagrams

We visualize calibration using reliability diagrams (Confidence vs. Accuracy).

![Reliability Diagram](../assets/plots/reliability_diagram.png)

## Temperature Scaling

A post-processing technique to calibrate predictions.

```math
\hat{p} = \text{softmax}(\frac{\text{logits}}{T})
```

where $T$ is a learned scalar parameter.
