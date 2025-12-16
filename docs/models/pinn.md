# Physics-Informed Neural Networks (PINNs)

Neural networks constrained by physical laws for scientific applications.

---

## Overview

PINNs integrate differential equations directly into the loss function, enabling:
- **Physically consistent predictions** that obey conservation laws
- **Generalization beyond training data** by encoding domain knowledge
- **Data-efficient learning** when physics is well-understood

---

## Mathematical Foundation

### Total Loss

$$\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{PDE}$$

Where:
- $\mathcal{L}_{data}$: Data fidelity term (MSE to observations)
- $\mathcal{L}_{PDE}$: Physics residual at collocation points
- $\lambda$: Weighting factor for physics constraint

### PDE Residual

For a PDE of the form $\mathcal{N}[u](x) = f(x)$:

$$\mathcal{L}_{PDE} = \frac{1}{N_c} \sum_{i=1}^{N_c} \left| \mathcal{N}[\hat{u}](x_i) - f(x_i) \right|^2$$

Where $N_c$ collocation points are sampled in the domain.

---

## Usage

### Basic PINN

```python
from ununennium.models.pinn import PINN, DiffusionEquation, MLP

# Define the governing equation
equation = DiffusionEquation(diffusivity=0.1)

# Create network architecture
network = MLP(
    layers=[2, 128, 128, 128, 1],  # 2D input → 1D output
    activation="tanh",
)

# Create PINN
pinn = PINN(
    network=network,
    equation=equation,
    lambda_pde=10.0,  # Physics weight
)

# Training
for epoch in range(epochs):
    # Sample collocation points
    x_collocation = sampler.sample(n_points=1000)
    
    # Compute combined loss
    loss_dict = pinn.compute_loss(
        x_data=x_train,
        u_data=u_train,
        x_collocation=x_collocation,
    )
    
    total_loss = loss_dict["data"] + loss_dict["pde"]
    total_loss.backward()
    optimizer.step()
```

---

## Available Equations

### DiffusionEquation

$$\frac{\partial u}{\partial t} = D \nabla^2 u$$

```python
from ununennium.models.pinn import DiffusionEquation

equation = DiffusionEquation(diffusivity=0.1)
```

**Applications**: Heat transfer, pollutant dispersion, groundwater flow

### AdvectionEquation

$$\frac{\partial u}{\partial t} + \mathbf{v} \cdot \nabla u = 0$$

```python
from ununennium.models.pinn import AdvectionEquation

equation = AdvectionEquation(velocity=[1.0, 0.5])
```

**Applications**: Transport phenomena, tracer movement

### AdvectionDiffusionEquation

$$\frac{\partial u}{\partial t} + \mathbf{v} \cdot \nabla u = D \nabla^2 u$$

```python
from ununennium.models.pinn import AdvectionDiffusionEquation

equation = AdvectionDiffusionEquation(
    velocity=[1.0, 0.5],
    diffusivity=0.01,
)
```

---

## Collocation Sampling

### Uniform Sampler

```python
from ununennium.models.pinn import UniformSampler

sampler = UniformSampler(
    bounds=[(0, 1), (0, 1), (0, 10)],  # x, y, t
    n_points=10000,
)
```

### Adaptive Sampler

Focus points where PDE residual is high:

```python
from ununennium.models.pinn import AdaptiveSampler

sampler = AdaptiveSampler(
    bounds=bounds,
    residual_fn=pinn.pde_residual,
    n_points=5000,
    refinement_ratio=0.3,
)
```

---

## Boundary Conditions

### Dirichlet (Fixed Value)

```python
pinn.add_boundary_condition(
    type="dirichlet",
    location=lambda x: x[:, 0] == 0,  # Left boundary
    value=1.0,
)
```

### Neumann (Fixed Gradient)

```python
pinn.add_boundary_condition(
    type="neumann",
    location=lambda x: x[:, 0] == 1,  # Right boundary
    gradient=0.0,  # No-flux
)
```

---

## Custom Equations

Define your own PDE:

```python
from ununennium.models.pinn import PDEEquation

class NavierStokes2D(PDEEquation):
    def __init__(self, viscosity):
        self.nu = viscosity
    
    def residual(self, x, u, grads):
        # u = [u, v, p] velocity and pressure
        # grads contains automatic derivatives
        u_t = grads["u_t"]
        u_x, u_y = grads["u_x"], grads["u_y"]
        u_xx, u_yy = grads["u_xx"], grads["u_yy"]
        p_x = grads["p_x"]
        
        # Momentum equation (x)
        res_u = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        
        return res_u  # + other residuals
```

---

## Earth Observation Applications

### Sea Surface Temperature Interpolation

```python
# SST follows advection-diffusion
equation = AdvectionDiffusionEquation(
    velocity=ocean_current_field,
    diffusivity=thermal_diffusivity,
)

pinn = PINN(network=network, equation=equation)
# Train on sparse buoy observations
# Predict dense SST field respecting physics
```

### Atmospheric Dispersion Modeling

```python
# Pollutant dispersion
equation = AdvectionDiffusionEquation(
    velocity=wind_field,
    diffusivity=eddy_diffusivity,
)
```

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks*. Journal of Computational Physics.
2. Karniadakis, G. E., et al. (2021). *Physics-informed machine learning*. Nature Reviews Physics.
