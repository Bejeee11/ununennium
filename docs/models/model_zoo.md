# Ununennium Model Zoo & Architecture Theory

## 1. Design Philosophy

The Ununennium Model Zoo is not merely a collection of weights; it is a repository of **Production-Grade**, **Geospatially-Aware** neural architectures. Each model is curated to handle the specific challenges of Earth Observation:
*   **Multi-Modal Inputs:** Native support for $N$-channel inputs (Optical + SAR + DEM).
*   **Scale Invariance:** Architectures designed to handle GSD (Ground Sampling Distance) variations.
*   **Rotation Equivariance:** Recognition that "up" on a map is arbitrary.

---

## 2. Segmentation Architectures

### 2.1 The U-Net Family (Standard, ResNet, EfficientNet)

The workhorse of semantic segmentation.

**Mathematical Formulation:**
Let $E$ be the encoder function and $D$ be the decoder.
$$ z_l = E_l(z_{l-1}) \quad \text{(Downsampling)} $$
$$ x_l = D_l(x_{l+1}, z_l) \quad \text{(Upsampling + Skip)} $$

The skip connection $z_l$ is critical in EO because determining boundaries (e.g., road vs. building) requires high-frequency spatial details that are lost in the bottleneck.

**Ununennium Variations:**
*   **`unet_resnet50`**: Deep residual backbone. Best for large datasets.
*   **`unet_efficientnet_b4`**: Compound scaling. Best accuracy/FLOPS trade-off.
*   **Dynamic Upsampling:** We use standard `bilinear` interpolation rather than `ConvTranspose` to avoid checkerboard artifacts (Odena et al., 2016), which are detrimental for geospatial vectorization.

### 2.2 DeepLabV3+ (Atrous Convolution)

Designed to capture multi-scale context without losing spatial resolution.

**Atrous (Dilated) Convolution:**
$$ y[i] = \sum_k x[i + r \cdot k] w[k] $$
Where $r$ is the dilation rate. This expands the Receptive Field (RF) exponentially without increasing parameters.

**ASPP (Atrous Spatial Pyramid Pooling):**
Parallel branches with rates $r=\{6, 12, 18\}$.
*   *EO Application:* Detecting large objects (lakes) and small objects (cars) simultaneously in the same scene.

---

## 3. Generative Adversarial Networks (GANs)

### 3.1 Pix2Pix (Conditional GAN)

Used for paired translation (e.g., SAR $\to$ Optical, Cloud $\to$ Clear).

**cGAN Objective:**
$$ \mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{x,z}[\log(1 - D(x, G(x, z)))] $$
**L1 Term:**
$$ \mathcal{L}_{L1}(G) = \mathbb{E}_{x,y,z}[\| y - G(x, z) \|_1] $$

We use the **PatchGAN** discriminator, which classifies $N \times N$ patches as real/fake. This enforces high-frequency texture correctness (texture consistency) critical for visual interpretation.

### 3.2 CycleGAN (Unpaired Translation)

Used for domain adaptation (e.g., Summer $\to$ Winter) where paired data is impossible.

**Cycle Consistency Loss:**
$$ x \to G(x) \to F(G(x)) \approx x $$
$$ \| F(G(x)) - x \|_1 $$

This constraint prevents "mode collapse" and ensures the semantic content (geometry of roads/buildings) is preserved while the style (season/sensor) changes.

---

## 4. Physics-Informed Neural Networks (PINNs)

PINNs integrate differential equations directly into the loss function.

**Total Loss:**
$$ \mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{PDE} $$

Where $\mathcal{L}_{PDE}$ is the residual of the governing equation (e.g., Diffusion, Navier-Stokes).
*   *Application:* Modeling atmospheric dispersion, groundwater flow, or sea surface temperature interpolation.
*   *Advantage:* Generalizes outside the training data because it obeys physical laws.

---

## 5. Vision Transformers (ViT)

Moving beyond convolutions.

**Self-Attention Mechanism:**
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

In EO, ViTs excel at **Global Context**. A pixel depends on the entire scene (e.g., identifying an airport requires seeing the runway arrangement, not just local asphalt texture).
*   **Mae (Masked Autoencoders):** We support MAE pre-training on unlabeled satellite imagery to learn robust representation foundations.
