# PINN Theory

Physics-Informed Neural Networks for scientific computing in Earth observation.

---

## Overview

PINNs incorporate physical laws as soft constraints via the loss function.

**Reference**: Raissi, M., et al. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707. [DOI: 10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

---

## Formulation

### General PDE

```math
\mathcal{N}[u](x, t) = 0, \quad x \in \Omega
```

### Composite Loss

```math
\mathcal{L}(\theta) = \lambda_d \mathcal{L}_{data} + \lambda_p \mathcal{L}_{pde} + \lambda_b \mathcal{L}_{bc}
```

where:

```math
\mathcal{L}_{data} = \frac{1}{N_d}\sum_{i=1}^{N_d}\|u_\theta(x_i) - u_i^{obs}\|^2
```

```math
\mathcal{L}_{pde} = \frac{1}{N_c}\sum_{j=1}^{N_c}\|\mathcal{N}[u_\theta](x_j)\|^2
```

---

## Common PDEs

### Heat/Diffusion Equation

```math
\frac{\partial u}{\partial t} = D \nabla^2 u
```

### Advection-Diffusion

```math
\frac{\partial u}{\partial t} + \mathbf{v} \cdot \nabla u = D \nabla^2 u
```

**Reference**: Okubo, A. (1971). Oceanic diffusion diagrams. *Deep Sea Research*, 18(8), 789-802. [DOI: 10.1016/0011-7471(71)90046-5](https://doi.org/10.1016/0011-7471(71)90046-5)

---

## Network Architectures

### Fourier Features

```math
\gamma(x) = [\cos(2\pi B x), \sin(2\pi B x)]^T
```

**Reference**: Tancik, M., et al. (2020). Fourier Features Let Networks Learn High Frequency Functions. *NeurIPS*. [arXiv:2006.10739](https://arxiv.org/abs/2006.10739)

---

## Collocation Strategies

**Reference**: Lu, L., et al. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208-228. [DOI: 10.1137/19M1274067](https://doi.org/10.1137/19M1274067)

---

## EO Applications

| Application | PDE | Observable |
|-------------|-----|------------|
| SST Interpolation | Advection-Diffusion | Temperature |
| Pollution Dispersion | Transport | Concentration |
| Groundwater Flow | Darcy | Hydraulic head |

---

## See Also

- [Math Foundations](math-foundations.md)
- [GAN Theory](gan-theory.md)
