# API Reference: Visualization

Tools for visualizing geospatial data and model outputs.

---

## Image Visualization

### plot_rgb

Display satellite imagery as RGB composite.

```python
from ununennium.utils.visualization import plot_rgb

# Sentinel-2: B4 (Red), B3 (Green), B2 (Blue)
plot_rgb(
    tensor,
    bands=[3, 2, 1],  # 0-indexed
    title="Sentinel-2 True Color",
    figsize=(10, 10),
    percentile_clip=(2, 98),  # Contrast stretch
)
```

### plot_false_color

Create false color composites for vegetation or urban analysis.

```python
from ununennium.utils.visualization import plot_false_color

# NIR-Red-Green composite for vegetation
plot_false_color(
    tensor,
    bands={"R": 7, "G": 3, "B": 2},  # NIR, Red, Green
    title="False Color (Vegetation)",
)
```

---

## Segmentation Visualization

### plot_segmentation

Overlay segmentation masks on imagery.

```python
from ununennium.utils.visualization import plot_segmentation

plot_segmentation(
    image=tensor,
    mask=prediction,
    class_names=["Water", "Forest", "Urban", "Agriculture"],
    alpha=0.5,
    colormap="tab10",
)
```

### plot_confusion_matrix

Display confusion matrix with class labels.

```python
from ununennium.utils.visualization import plot_confusion_matrix

plot_confusion_matrix(
    cm=confusion_matrix,
    class_names=class_names,
    normalize=True,
    figsize=(12, 10),
    cmap="Blues",
)
```

---

## Prediction Comparison

### plot_comparison

Side-by-side comparison of input, ground truth, and prediction.

```python
from ununennium.utils.visualization import plot_comparison

plot_comparison(
    image=input_tensor,
    ground_truth=gt_mask,
    prediction=pred_mask,
    class_names=class_names,
    figsize=(15, 5),
)
```

---

## GAN Outputs

### plot_translation

Visualize image translation results.

```python
from ununennium.utils.visualization import plot_translation

plot_translation(
    source=sar_image,
    generated=fake_optical,
    target=real_optical,  # Optional
    titles=["SAR Input", "Generated Optical", "Real Optical"],
)
```

---

## Uncertainty Visualization

### plot_uncertainty

Visualize model uncertainty maps.

```python
from ununennium.utils.visualization import plot_uncertainty

plot_uncertainty(
    mean_prediction=mean_pred,
    uncertainty_map=entropy_map,
    colormap="viridis",
    title="Prediction Uncertainty",
)
```

---

## GeoTensor Utilities

### export_geotiff

Save predictions as GeoTIFF with proper CRS.

```python
from ununennium.utils.visualization import export_geotiff

export_geotiff(
    tensor=prediction,
    path="output.tif",
    crs=input_tensor.crs,
    transform=input_tensor.transform,
)
```

---

## API Reference

::: ununennium.utils.visualization
