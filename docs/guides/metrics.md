# Metrics Guide

## Segmentation

### IoU (Jaccard Index)

```math
\text{IoU} = \frac{\text{Intersection}}{\text{Union}}
```

### Dice Coefficient

```math
\text{Dice} = \frac{2 \cdot \text{Intersection}}{\text{Sum of Areas}}
```

## Change Detection

### Precision & Recall

```math
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}
```

## Calibration

### ECE (Expected Calibration Error)

```math
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|
```
