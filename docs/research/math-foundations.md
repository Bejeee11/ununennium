# Mathematical Foundations

This document establishes the mathematical foundations for Earth observation machine learning.

---

## Coordinate Reference Systems

### Geodetic Datum

The World Geodetic System 1984 (WGS84) defines the reference ellipsoid:

```math
\frac{x^2 + y^2}{a^2} + \frac{z^2}{b^2} = 1
```

where a = 6378137 m (semi-major axis) and b = 6356752.314 m (semi-minor axis).

**Reference**: National Imagery and Mapping Agency (2000). *Department of Defense World Geodetic System 1984*. NIMA TR8350.2.

### Universal Transverse Mercator (UTM)

UTM divides Earth into 60 zones, each 6 degrees wide:

```math
\text{zone} = \left\lfloor \frac{\lambda + 180}{6} \right\rfloor + 1
```

**Reference**: Snyder, J.P. (1987). *Map Projections: A Working Manual*. USGS Professional Paper 1395. [DOI: 10.3133/pp1395](https://doi.org/10.3133/pp1395)

---

## Spectral Indices

### NDVI (Normalized Difference Vegetation Index)

```math
\text{NDVI} = \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + \rho_{Red}}
```

NDVI ranges from -1 to +1, with healthy vegetation typically 0.2-0.9.

**Reference**: Tucker, C.J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment*, 8(2), 127-150. [DOI: 10.1016/0034-4257(79)90013-0](https://doi.org/10.1016/0034-4257(79)90013-0)

### EVI (Enhanced Vegetation Index)

```math
\text{EVI} = G \cdot \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + C_1 \cdot \rho_{Red} - C_2 \cdot \rho_{Blue} + L}
```

with G = 2.5, C₁ = 6, C₂ = 7.5, and L = 1.

**Reference**: Huete, A., et al. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. *Remote Sensing of Environment*, 83(1-2), 195-213. [DOI: 10.1016/S0034-4257(02)00096-2](https://doi.org/10.1016/S0034-4257(02)00096-2)

### NDWI (Normalized Difference Water Index)

```math
\text{NDWI} = \frac{\rho_{Green} - \rho_{NIR}}{\rho_{Green} + \rho_{NIR}}
```

**Reference**: McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI). *International Journal of Remote Sensing*, 17(7), 1425-1432. [DOI: 10.1080/01431169608948714](https://doi.org/10.1080/01431169608948714)

---

## Spatial Statistics

### Moran's I

Global measure of spatial autocorrelation:

```math
I = \frac{n}{\sum_i \sum_j w_{ij}} \cdot \frac{\sum_i \sum_j w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}
```

| Value | Interpretation |
|-------|---------------|
| I > 0 | Positive autocorrelation (clustering) |
| I ≈ 0 | Random pattern |
| I < 0 | Negative autocorrelation (dispersion) |

**Reference**: Moran, P.A.P. (1950). Notes on Continuous Stochastic Phenomena. *Biometrika*, 37(1/2), 17-23. [DOI: 10.2307/2332142](https://doi.org/10.2307/2332142)

### Semivariogram

```math
\gamma(h) = \frac{1}{2|N(h)|} \sum_{(i,j) \in N(h)} (z_i - z_j)^2
```

**Reference**: Matheron, G. (1963). Principles of geostatistics. *Economic Geology*, 58(8), 1246-1266. [DOI: 10.2113/gsecongeo.58.8.1246](https://doi.org/10.2113/gsecongeo.58.8.1246)

---

## Deep Learning Foundations

### Convolution

```math
(I * K)[i,j] = \sum_m \sum_n I[i+m, j+n] \cdot K[m, n]
```

**Reference**: LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444. [DOI: 10.1038/nature14539](https://doi.org/10.1038/nature14539)

### Attention Mechanism

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

**Reference**: Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*, 30. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## Loss Functions

### Cross-Entropy Loss

```math
\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
```

### Dice Loss

```math
\mathcal{L}_{Dice} = 1 - \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}
```

**Reference**: Milletari, F., et al. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV*. [DOI: 10.1109/3DV.2016.79](https://doi.org/10.1109/3DV.2016.79)

---

## See Also

- [GAN Theory](gan-theory.md)
- [PINN Theory](pinn-theory.md)
- [Bibliography](bibliography.md)
