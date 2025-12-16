# GAN Theory

Theoretical foundations of Generative Adversarial Networks for Earth observation image translation and enhancement.

---

## Adversarial Learning Framework

### Minimax Game

GANs consist of two networks in a zero-sum game:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

where:
- $G$: Generator network
- $D$: Discriminator network
- $p_{data}$: Real data distribution
- $p_z$: Prior noise distribution

**Reference**: Goodfellow, I., et al. (2014). Generative Adversarial Nets. *Advances in Neural Information Processing Systems*, 27. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

### Nash Equilibrium

At optimal convergence:
- $p_G = p_{data}$ (generator perfectly matches data)
- $D(x) = 0.5$ for all $x$ (discriminator cannot distinguish)

---

## Loss Functions

### Vanilla GAN Loss

**Discriminator**:
$$
\mathcal{L}_D = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

**Generator**:
$$
\mathcal{L}_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]
$$

### Least Squares GAN (LSGAN)

More stable training using squared error:

$$
\mathcal{L}_D = \frac{1}{2}\mathbb{E}_{x}[(D(x) - 1)^2] + \frac{1}{2}\mathbb{E}_{z}[D(G(z))^2]
$$

$$
\mathcal{L}_G = \frac{1}{2}\mathbb{E}_{z}[(D(G(z)) - 1)^2]
$$

**Reference**: Mao, X., et al. (2017). Least Squares Generative Adversarial Networks. *ICCV*. [DOI: 10.1109/ICCV.2017.304](https://doi.org/10.1109/ICCV.2017.304)

### Wasserstein GAN (WGAN)

Uses Earth Mover distance for improved gradients:

$$
\mathcal{L}_D = \mathbb{E}_{z}[D(G(z))] - \mathbb{E}_{x}[D(x)]
$$

with Lipschitz constraint on $D$.

**Reference**: Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. *ICML*. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)

### Hinge Loss

$$
\mathcal{L}_D = \mathbb{E}_{x}[\max(0, 1 - D(x))] + \mathbb{E}_{z}[\max(0, 1 + D(G(z)))]
$$

$$
\mathcal{L}_G = -\mathbb{E}_{z}[D(G(z))]
$$

**Reference**: Lim, J.H., & Ye, J.C. (2017). Geometric GAN. [arXiv:1705.02894](https://arxiv.org/abs/1705.02894)

---

## Image Translation Architectures

### Pix2Pix (Paired Translation)

Conditional GAN with L1 reconstruction loss:

$$
\mathcal{L} = \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)
$$

where:
$$
\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y}[\|y - G(x)\|_1]
$$

**Architecture**:
- Generator: U-Net with skip connections
- Discriminator: PatchGAN (classifies 70x70 patches)

**Reference**: Isola, P., et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR*. [DOI: 10.1109/CVPR.2017.632](https://doi.org/10.1109/CVPR.2017.632)

### CycleGAN (Unpaired Translation)

For unpaired training using cycle consistency:

$$
\mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y}[\|G(F(y)) - y\|_1]
$$

**Total Loss**:
$$
\mathcal{L} = \mathcal{L}_{GAN}(G, D_Y) + \mathcal{L}_{GAN}(F, D_X) + \lambda_{cyc}\mathcal{L}_{cyc} + \lambda_{id}\mathcal{L}_{id}
$$

**Reference**: Zhu, J.Y., et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. *ICCV*. [DOI: 10.1109/ICCV.2017.244](https://doi.org/10.1109/ICCV.2017.244)

---

## Super-Resolution GANs

### SRGAN

Perceptual loss for natural textures:

$$
\mathcal{L}_{percep} = \sum_{l} \frac{1}{C_l H_l W_l} \|\phi_l(I_{HR}) - \phi_l(G(I_{LR}))\|_2^2
$$

where $\phi_l$ are VGG19 feature maps.

**Reference**: Ledig, C., et al. (2017). Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. *CVPR*. [DOI: 10.1109/CVPR.2017.19](https://doi.org/10.1109/CVPR.2017.19)

### ESRGAN

Enhanced version with:
- RRDB (Residual-in-Residual Dense Block)
- Relativistic discriminator
- Perceptual loss before activation

**Reference**: Wang, X., et al. (2018). ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. *ECCV Workshops*. [arXiv:1809.00219](https://arxiv.org/abs/1809.00219)

---

## Training Stability

### Spectral Normalization

Normalizes weight matrices by spectral norm:

$$
\bar{W} = \frac{W}{\sigma(W)}
$$

where $\sigma(W)$ is the largest singular value.

**Reference**: Miyato, T., et al. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)

### Gradient Penalty (WGAN-GP)

$$
\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
$$

where $\hat{x}$ is interpolated between real and fake.

**Reference**: Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs. *NeurIPS*. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)

---

## Earth Observation Applications

| Application | Architecture | Input | Output |
|-------------|--------------|-------|--------|
| SAR-to-Optical | Pix2Pix | SAR | RGB |
| Cloud Removal | CycleGAN | Cloudy | Clear |
| Pan-sharpening | SRGAN | MS + PAN | HR MS |
| Resolution Enhancement | ESRGAN | 10m | 2.5m |

---

## See Also

- [Math Foundations](math-foundations.md)
- [PINN Theory](pinn-theory.md)
- [GAN API Reference](../api/gan.md)
