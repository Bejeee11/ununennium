# Ununennium: The Geospatial Deep Learning Standard

**Ununennium** is a production-grade, theoretically rigorous, and high-performance Python library designed for the era of Petabyte-scale Earth Observation.

It bridges the gap between **Geospatial Engineering** (GDAL, PROJ, OGC standards) and **Deep Learning** (PyTorch, Autograd, CUDA), providing a unified API for training state-of-the-art models on satellite imagery.

---

## The Philosophy of Rigor

Most "satellite-ai" libraries treat geospatial rasters as mere pictures (JPEGs). This approach fails in scientific applications because it ignores:
1.  **Metric Distortion:** The Earth is curved; pixels have variable physical areas.
2.  **Spectral Radiometry:** "Color" is a physical measurement of photons, not a 0-255 integer.
3.  **Spatial Autocorrelation:** The i.i.d. assumption of standard SGD is violated in geography.

**Ununennium** treats these as first-class citizens.
*   **GeoTensor:** Every tensor knows its CRS ($ReferenceSystem$) and physical location ($Transform$).
*   **Physics-Aware Models:** Loss functions respect physical laws (PINNs) and radiometric bounds.
*   **Sampling Theory:** Data loaders implement stratified spatial sampling to respect geostatistics.

---

## Core Capabilities

### 1. Cloud-Native I/O
Streaming directly from S3/GCS using COG (Cloud Optimized GeoTIFF) and Zarr logic. No more "download-then-train"; we train directly on the data lake.

```python
# Streaming access to 10TB dataset
ds = ununennium.io.read_stac(
    "s3://sentinel-cogs/T32VNM/...",
    resolution=10.0,
    resampling="average" # Radiometrically correct downsampling
)
```

### 2. The Model Zoo
15+ Architectures optimized for EO.
*   **Segmentation:** U-Net (ResNet/EffNet), DeepLabV3+.
*   **Generative:** Pix2Pix (SAR-to-Optical), CycleGAN (Season Transfer).
*   **Physics:** PDEs solved via Neural Networks (PINNs).

### 3. Training at Scale
Mixed Precision (AMP), Distributed Data Parallel (DDP), and Gradient Accumulation tailored for massive 12+ channel satellite inputs.

---

## Navigate the Documentation

### [Theory Guides](theory/index.md)
Master the mathematics behind the code.
*   [Spatial Autocorrelation](theory/spatial_autocorrelation.md)
*   [Geodesic Calculations](theory/geodesic_calculations.md)
*   [Sampling Theory](theory/sampling_theory.md)
*   [Uncertainty Quantification](theory/uncertainty.md)

### [Feature Catalog](models/model_zoo.md)
Detailed architectural diagrams and formulaic definitions of available models.

### [API Reference](api/core.md)
Strict typing notation and docstrings for every class and function.

### [Tutorials](tutorials/quickstart.md)
Step-by-step notebooks taking you from raw GeoTIFFs to deployed Onnx models.

---

## Provenance

**Ununennium** is developed by the DeepMind Advanced Agentic Coding Team as an exemplar of "Software 2.0" for the physical sciences.
