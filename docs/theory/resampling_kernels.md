# Theory of Resampling Kernels in Remote Sensing

## 1. Introduction to Digital Image Resampling

Resampling is the process of transforming a discrete image from one coordinate system to another, or changing its spatial resolution. In remote sensing, this is mathematically equivalent to **signal reconstruction** followed by **sampling** at new grid points.

Generally, the continuous signal $f(x, y)$ is reconstructed from discrete samples $f[n_x, n_y]$ via convolution with a reconstruction kernel $h$:

$$ \hat{f}(x, y) = \sum_{n_x} \sum_{n_y} f[n_x, n_y] \cdot h(x - n_x) \cdot h(y - n_y) $$

The choice of kernel $h(t)$ dictates the spectral preservation and geometric fidelity of the result.

---

## 2. Kernel Taxonomy and Frequency Domain Analysis

### 2.1 Nearest Neighbor (Box Kernel)

The simplest interpolation, equivalent to a convolution with a rectangle function ($\text{rect}$).

$$ h(t) = \begin{cases} 1 & |t| < 0.5 \\ 0 & \text{otherwise} \end{cases} $$

*   **Frequency Response:** $\text{sinc}(f)$. It has infinite support in the frequency domain, causing significant **aliasing** (jagged edges).
*   **Pros:** Preserves original digital numbers (DN). Essential for **categorical data** (classification masks).
*   **Cons:** Maximum geometric distortion (up to 0.5 pixel shift).

### 2.2 Bilinear Interpolation (Triangle Kernel)

Convolution with a triangle function ($\text{tri} = \text{rect} * \text{rect}$).

$$ h(t) = \begin{cases} 1 - |t| & |t| < 1 \\ 0 & \text{otherwise} \end{cases} $$

*   **Frequency Response:** $\text{sinc}^2(f)$. Better suppression of high frequencies (aliasing) than Nearest Neighbor, but acts as a **low-pass filter**, blurring the image.
*   **Pros:** Continuous signal, computationally cheap.
*   **Cons:** Blurs edges, modifies radiometric values (not suitable for raw spectral analysis).

### 2.3 Cubic Convolution (Approximated Sinc)

Approximates the ideal $\text{sinc}$ filter using piecewise cubic polynomials. The general form depends on a parameter $\alpha$ (usually -0.5 or -0.75).

$$ h(t) = \begin{cases} (\alpha+2)|t|^3 - (\alpha+3)|t|^2 + 1 & |t| < 1 \\ \alpha|t|^3 - 5\alpha|t|^2 + 8\alpha|t| - 4\alpha & 1 \le |t| < 2 \\ 0 & \text{otherwise} \end{cases} $$

*   **Frequency Response:** Closer to the ideal brick-wall filter.
*   **Pros:** Preserves mean and standard deviation better than bilinear. Sharper edges.
*   **Cons:** Can introduce **Gibbs phenomenon** (ringing/overshoot) at sharp contrast boundaries (e.g., water/land interfaces).

### 2.4 Lanczos Resampling (Windowed Sinc)

Uses a $\text{sinc}$ kernel windowed by a larger $\text{sinc}$ envelope. Considered the gold standard for downsampling.

$$ L(x) = \begin{cases} \text{sinc}(x) \text{sinc}(x/a) & |x| < a \\ 0 & \text{otherwise} \end{cases} $$

where typically $a=2$ or $a=3$.

---

## 3. Aliasing and Moiré Patterns

When downsampling an image (decimation) without adequate pre-filtering, high-frequency components "fold over" into low frequencies, creating false patterns (aliasing).

**Nyquist-Shannon Sampling Theorem:**
To reconstruct a signal, one must sample at a rate $f_s > 2B$, where $B$ is the signal bandwidth.

Most satellite sensors (Sentinel-2, Landsat) have a Modulation Transfer Function (MTF) that allows some aliasing to maintain sharpness. When users naively resize these images using Nearest Neighbor, aliasing is exacerbated, potentially confusing Convolutional Neural Networks (CNNs) with texture artifacts.

> **Ununennium Standard:** We implement **Average Pooling** or **Lanczos** for downsampling satellite imagery to simulate sensor integration, strictly avoiding Nearest Neighbor for continuous spectral bands.

---

## 4. Kernel Selection Matrix

Ununennium's `GeoTensor.resample()` module selects kernels based on data type and analysis goal:

| Data Type | Goal | Recommended Kernel | Rationale |
|-----------|------|--------------------|-----------|
| **Discrete (Masks)** | Reprojection | **Nearest Neighbor** | Must strictly preserve class integers. |
| **Continuous (RGB)** | Visualization | **Bicubic / Lanczos** | Smoothest visual appearance, sharp edges. |
| **Continuous (Science)** | Spectral Analysis | **Bilinear / Average** | Minimized "ringing" artifacts; conserves energy. |
| **Radar (SAR)** | Speckle Reduction | **Average / Box** | Reduces noise variance. |
| **Elevation (DEM)** | Terrain Analysis | **Bicubic** | Ensures continuous derivatives for slope/aspect calc. |

---

## 5. Mathematical Impact on Deep Learning Features

Using the wrong resampling kernel injects inductive biases into the neural network:

1.  **Artifact Injection:** Cubic ringing creates "halos" around objects. A U-Net might learn to detect buildings by their ringing halos rather than their spectral signature.
2.  **Spectral Mixing:** Bilinear interpolation creates "mixed pixels" that never existed in reality. A generated value of 0.5 (between forest 0.2 and water 0.8) might be interpreted as "urban" (0.5), leading to classification errors.
3.  **Positional Shift:** Nearest Neighbor introduces spatial jitter up to $\pm \frac{1}{2}$ pixel. In multi-temporal change detection, this creates false change along all edges.

**Ununennium Policy:**
*   **Training:** All augmentations involving rotation/scale use **Bilinear** (images) and **Nearest** (masks).
*   **Preprocessing:** L1C to L2A conversions or reprojections utilize **Average** resampling to strictly preserve photon counts (radiometric integrity).
