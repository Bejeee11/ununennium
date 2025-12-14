# Theory of Error Propagation in Geospatial Pipelines

## 1. The Cascade of Uncertainty

In a typical EO pipeline, errors are not additive; they propagate and compound non-linearly through transformations.

$$ \text{Source} \to \text{Registration} \to \text{Resampling} \to \text{Classification} \to \text{Vectorization} $$

Let $y = f(x_1, x_2, ...)$. The variance of the output $\sigma_y^2$ is approximations by Taylor series expansion:

$$ \sigma_y^2 \approx \sum_{i} \left( \frac{\partial f}{\partial x_i} \right)^2 \sigma_{x_i}^2 + \sum_{i \neq j} \left( \frac{\partial f}{\partial x_i} \right) \left( \frac{\partial f}{\partial x_j} \right) \sigma_{ij} $$

---

## 2. Positional Error Propagation

### 2.1 Registration Error (RMSE)
If a satellite image has a geolocation RMSE of 10m ($1\sigma$), and we perform change detection with a second image (RMSE 10m), the relative registration error is:

$$ \sigma_{rel} = \sqrt{10^2 + 10^2} \approx 14.1\text{m} $$

For a Change Detection model, any change smaller than $2\sigma_{rel}$ (approx 28m) is statistically indistinguishable from misalignments.
*   **Implication:** Do not train building footprint segmentation on poorly registered data. The model will learn to predict "fuzzy" boundaries to minimize loss, effectively learning the registration noise.

---

## 3. Classification Error Propagation

### 3.1 The Confusion Matrix
For a classifier $C$, let $P_{ij}$ be the probability of predicting class $j$ given true class $i$.

### 3.2 Area Estimation Bias (Olofsson et al., 2014)
Counting pixels to estimate area (e.g., "Deforestation Area") is biased if the classifier is imperfect.
*   **Mapped Area ($A_m$):** Raw pixel count.
*   **Estimated Area ($\hat{A}$):** Corrected using error matrix.

$$ \hat{p}_k = \sum_{i=1}^q W_i \frac{n_{ik}}{n_{i.}} $$
(Stratified estimator of area proportion)

Ununennium's `metrics` module automatically calculates **Confidence Intervals** for area estimates, not just raw pixel counts.

---

## 4. Vectorization Error

Deep Learning produces raster masks. Converting these to vectors (polygons) introduces:
1.  **Staircase Effect:** Raster aliases the true boundary.
2.  **Simplification Error:** Douglas-Peucker algorithms remove vertices.

**Topo-Regularization:**
We employ regularization losses (e.g., Active Contour / Level Set loss) during training so the raster output is topologically homeomorphic to the target vector, minimizing conversion error.
