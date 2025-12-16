# Metrics

This guide covers metric selection and interpretation for Earth observation tasks.

---

## Segmentation Metrics

### IoU (Intersection over Union)

$$
\text{IoU} = \frac{TP}{TP + FP + FN}
$$

- Range: [0, 1], higher is better
- Penalizes both false positives and false negatives

### Dice Coefficient

$$
\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

- Equivalent to F1 score
- More tolerant of small objects

### When to Use

| Metric | Best For |
|--------|----------|
| IoU | General evaluation |
| Dice | Class imbalance |
| Pixel Accuracy | Gross overview |

---

## Change Detection Metrics

### Overall Accuracy

$$
OA = \frac{TP + TN}{TP + TN + FP + FN}
$$

### Kappa Coefficient

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

Accounts for chance agreement.

| Kappa | Interpretation |
|-------|----------------|
| < 0 | Worse than chance |
| 0-0.2 | Slight |
| 0.2-0.6 | Fair to moderate |
| 0.6-0.8 | Substantial |
| > 0.8 | Almost perfect |

---

## Super-Resolution Metrics

### PSNR

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)
$$

Higher is better. Typical range: 20-40 dB.

### SSIM

Measures structural similarity. Range [0, 1].

---

## Calibration Metrics

### ECE (Expected Calibration Error)

Measures alignment between confidence and accuracy.

$$
\text{ECE} = \sum_m \frac{|B_m|}{n} \cdot |\text{acc}(B_m) - \text{conf}(B_m)|
$$

Lower is better.

---

## See Also

- [Evaluation API](../api/evaluation.md)
- [Uncertainty and Calibration](uncertainty-and-calibration.md)
