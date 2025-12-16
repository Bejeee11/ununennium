# API Reference: Metrics

Evaluation metrics for geospatial machine learning with proper handling of spatial data.

---

## Segmentation Metrics

### Intersection over Union (IoU / Jaccard Index)

$$IoU = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$

```python
from ununennium.metrics import IoU

metric = IoU(num_classes=10, ignore_index=255)
metric.update(predictions, targets)
iou_per_class = metric.compute()
mean_iou = iou_per_class.mean()
```

### Dice Coefficient (F1 Score)

$$Dice = \frac{2|A \cap B|}{|A| + |B|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

```python
from ununennium.metrics import DiceCoefficient

metric = DiceCoefficient(num_classes=10)
metric.update(predictions, targets)
dice_scores = metric.compute()
```

### Pixel Accuracy

$$PA = \frac{\sum_i n_{ii}}{\sum_i t_i}$$

Where $n_{ii}$ is the number of correctly classified pixels and $t_i$ is the total pixels of class $i$.

```python
from ununennium.metrics import PixelAccuracy

metric = PixelAccuracy()
accuracy = metric(predictions, targets)
```

---

## Detection Metrics

### Mean Average Precision (mAP)

$$mAP = \frac{1}{|C|} \sum_{c \in C} AP_c$$

```python
from ununennium.metrics import MeanAveragePrecision

metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
metric.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
results = metric.compute()
# results["mAP@0.5"], results["mAP@0.75"]
```

---

## Classification Metrics

### Confusion Matrix

```python
from ununennium.metrics import ConfusionMatrix

metric = ConfusionMatrix(num_classes=10, normalize="true")
cm = metric(predictions, targets)
```

### Cohen's Kappa

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where $p_o$ is observed agreement and $p_e$ is expected agreement by chance.

```python
from ununennium.metrics import CohenKappa

metric = CohenKappa(num_classes=10)
kappa = metric(predictions, targets)
```

---

## Calibration Metrics

### Expected Calibration Error (ECE)

$$ECE = \sum_{m=1}^M \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$$

```python
from ununennium.metrics import ExpectedCalibrationError

metric = ExpectedCalibrationError(n_bins=15)
ece = metric(probabilities, targets)
```

---

## Usage with Trainer

```python
from ununennium.training import Trainer
from ununennium.metrics import IoU, DiceCoefficient

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    metrics=[IoU(num_classes=10), DiceCoefficient(num_classes=10)],
)
```

---

## API Reference

::: ununennium.metrics.segmentation
