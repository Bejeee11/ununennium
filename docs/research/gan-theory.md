# GAN Theory

Theoretical foundations of Generative Adversarial Networks for Earth observation.

---

## Adversarial Learning

### Minimax Objective

```math
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
```

**Reference**: Goodfellow, I., et al. (2014). Generative Adversarial Nets. *NeurIPS*. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

---

## Loss Functions

### LSGAN (Least Squares)

```math
\mathcal{L}_D = \frac{1}{2}\mathbb{E}_{x}[(D(x) - 1)^2] + \frac{1}{2}\mathbb{E}_{z}[D(G(z))^2]
```

**Reference**: Mao, X., et al. (2017). Least Squares Generative Adversarial Networks. *ICCV*. [DOI: 10.1109/ICCV.2017.304](https://doi.org/10.1109/ICCV.2017.304)

### WGAN

```math
\mathcal{L}_D = \mathbb{E}_{z}[D(G(z))] - \mathbb{E}_{x}[D(x)]
```

**Reference**: Arjovsky, M., et al. (2017). Wasserstein GAN. *ICML*. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)

---

## Image Translation

### Pix2Pix

```math
\mathcal{L} = \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)
```

**Reference**: Isola, P., et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR*. [DOI: 10.1109/CVPR.2017.632](https://doi.org/10.1109/CVPR.2017.632)

### CycleGAN

Cycle consistency loss:

```math
\mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y}[\|G(F(y)) - y\|_1]
```

**Reference**: Zhu, J.Y., et al. (2017). Unpaired Image-to-Image Translation. *ICCV*. [DOI: 10.1109/ICCV.2017.244](https://doi.org/10.1109/ICCV.2017.244)

---

## Super-Resolution

### Perceptual Loss

```math
\mathcal{L}_{percep} = \sum_{l} \frac{1}{C_l H_l W_l} \|\phi_l(I_{HR}) - \phi_l(G(I_{LR}))\|_2^2
```

**Reference**: Ledig, C., et al. (2017). Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. *CVPR*. [DOI: 10.1109/CVPR.2017.19](https://doi.org/10.1109/CVPR.2017.19)

---

## Training Stability

### Spectral Normalization

```math
\bar{W} = \frac{W}{\sigma(W)}
```

**Reference**: Miyato, T., et al. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)

### Gradient Penalty

```math
\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
```

**Reference**: Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs. *NeurIPS*. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)

---

## See Also

- [Math Foundations](math-foundations.md)
- [PINN Theory](pinn-theory.md)
