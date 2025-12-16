# Mathematical Foundations

This document establishes the mathematical foundations for Earth observation machine learning, covering coordinate systems, spectral indices, spatial statistics, and deep learning fundamentals.

---

## Coordinate Reference Systems

### Geodetic Datum

The World Geodetic System 1984 (WGS84) defines the reference ellipsoid:

$$
\frac{x^2 + y^2}{a^2} + \frac{z^2}{b^2} = 1
$$

where $a = 6378137$ m (semi-major axis) and $b = 6356752.314$ m (semi-minor axis).

**Reference**: National Imagery and Mapping Agency (2000). *Department of Defense World Geodetic System 1984*. NIMA TR8350.2. [Link](https://earth-info.nga.mil/php/download.php?file=coord-wgs84)

### Map Projections

#### Universal Transverse Mercator (UTM)

UTM divides Earth into 60 zones, each 6 degrees wide:

$$
\text{zone} = \left\lfloor \frac{\lambda + 180}{6} \right\rfloor + 1
$$

The forward transformation from geodetic coordinates $(\phi, \lambda)$ to UTM $(E, N)$:

$$
E = E_0 + k_0 \nu \left( A + \frac{(1-T+C)A^3}{6} + \frac{(5-18T+T^2+72C-58e'^2)A^5}{120} \right)
$$

where $\nu = a/\sqrt{1 - e^2\sin^2\phi}$, $T = \tan^2\phi$, and $k_0 = 0.9996$.

**Reference**: Snyder, J.P. (1987). *Map Projections: A Working Manual*. USGS Professional Paper 1395. [DOI: 10.3133/pp1395](https://doi.org/10.3133/pp1395)

---

## Resampling Theory

### Interpolation Kernels

Image resampling requires interpolation from discrete samples:

$$
f(x) = \sum_{k} f[k] \cdot h(x - k)
$$

#### Bicubic Interpolation

The Keys cubic convolution kernel:

$$
h(x) = \begin{cases}
(a+2)|x|^3 - (a+3)|x|^2 + 1 & |x| \leq 1 \\
a|x|^3 - 5a|x|^2 + 8a|x| - 4a & 1 < |x| < 2 \\
0 & \text{otherwise}
\end{cases}
$$

with $a = -0.5$ for optimal approximation.

**Reference**: Keys, R. (1981). Cubic convolution interpolation for digital image processing. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 29(6), 1153-1160. [DOI: 10.1109/TASSP.1981.1163711](https://doi.org/10.1109/TASSP.1981.1163711)

---

## Spectral Indices

### Normalized Difference Formulation

The general normalized difference index:

$$
\text{NDI}_{i,j} = \frac{\rho_i - \rho_j}{\rho_i + \rho_j}
$$

### Vegetation Indices

#### NDVI (Normalized Difference Vegetation Index)

$$
\text{NDVI} = \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + \rho_{Red}}
$$

NDVI ranges from -1 to +1, with healthy vegetation typically 0.2-0.9.

**Reference**: Tucker, C.J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment*, 8(2), 127-150. [DOI: 10.1016/0034-4257(79)90013-0](https://doi.org/10.1016/0034-4257(79)90013-0)

#### EVI (Enhanced Vegetation Index)

$$
\text{EVI} = G \cdot \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + C_1 \cdot \rho_{Red} - C_2 \cdot \rho_{Blue} + L}
$$

with $G = 2.5$, $C_1 = 6$, $C_2 = 7.5$, and $L = 1$.

**Reference**: Huete, A., et al. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. *Remote Sensing of Environment*, 83(1-2), 195-213. [DOI: 10.1016/S0034-4257(02)00096-2](https://doi.org/10.1016/S0034-4257(02)00096-2)

### Water Indices

#### NDWI (Normalized Difference Water Index)

$$
\text{NDWI} = \frac{\rho_{Green} - \rho_{NIR}}{\rho_{Green} + \rho_{NIR}}
$$

**Reference**: McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425-1432. [DOI: 10.1080/01431169608948714](https://doi.org/10.1080/01431169608948714)

---

## Spatial Statistics

### Spatial Autocorrelation

#### Moran's I

Global measure of spatial autocorrelation:

$$
I = \frac{n}{\sum_i \sum_j w_{ij}} \cdot \frac{\sum_i \sum_j w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}
$$

| Value | Interpretation |
|-------|---------------|
| $I > 0$ | Positive autocorrelation (clustering) |
| $I \approx 0$ | Random pattern |
| $I < 0$ | Negative autocorrelation (dispersion) |

**Reference**: Moran, P.A.P. (1950). Notes on Continuous Stochastic Phenomena. *Biometrika*, 37(1/2), 17-23. [DOI: 10.2307/2332142](https://doi.org/10.2307/2332142)

#### Semivariogram

The semivariance function:

$$
\gamma(h) = \frac{1}{2|N(h)|} \sum_{(i,j) \in N(h)} (z_i - z_j)^2
$$

Components:
- **Nugget** ($c_0$): Variance at zero distance
- **Sill** ($c_0 + c$): Total variance
- **Range** ($a$): Distance where autocorrelation ceases

**Reference**: Matheron, G. (1963). Principles of geostatistics. *Economic Geology*, 58(8), 1246-1266. [DOI: 10.2113/gsecongeo.58.8.1246](https://doi.org/10.2113/gsecongeo.58.8.1246)

---

## Deep Learning Foundations

### Convolutional Neural Networks

The discrete 2D convolution:

$$
(I * K)[i,j] = \sum_m \sum_n I[i+m, j+n] \cdot K[m, n]
$$

**Reference**: LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444. [DOI: 10.1038/nature14539](https://doi.org/10.1038/nature14539)

### Attention Mechanism

Scaled dot-product attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**Reference**: Vaswani, A., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## Loss Functions

### Cross-Entropy Loss

For classification:

$$
\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

### Dice Loss

For segmentation with class imbalance:

$$
\mathcal{L}_{Dice} = 1 - \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}
$$

**Reference**: Milletari, F., Navab, N., & Ahmadi, S.A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV 2016*. [DOI: 10.1109/3DV.2016.79](https://doi.org/10.1109/3DV.2016.79)

---

## See Also

- [GAN Theory](gan-theory.md)
- [PINN Theory](pinn-theory.md)
- [Bibliography](bibliography.md)
