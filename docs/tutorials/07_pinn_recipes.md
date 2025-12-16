# Tutorial 07: PINN Recipes

This tutorial provides practical recipes for Physics-Informed Neural Networks in Earth observation.

---

## Recipe 1: Heat Diffusion (Basic PINN)

Model thermal diffusion on land surface.

### Problem Setup

Heat equation:

$$
\frac{\partial u}{\partial t} = D \nabla^2 u
$$

### Implementation

```python
from ununennium.models.pinn import PINN, DiffusionEquation, MLP, UniformSampler

# Define PDE
equation = DiffusionEquation(diffusivity=0.1)

# Create network
network = MLP(
    layers=[3, 128, 128, 128, 1],  # (x, y, t) -> temperature
    activation="tanh",
)

# Create PINN
pinn = PINN(
    network=network,
    equation=equation,
    lambda_pde=10.0,
)

# Sample collocation points
sampler = UniformSampler(
    bounds=[(0, 1), (0, 1), (0, 1)],  # x, y, t
    n_points=10000,
)
x_collocation = sampler.sample()

# Training
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

for epoch in range(10000):
    optimizer.zero_grad()
    
    losses = pinn.compute_loss(x_data, u_data, x_collocation)
    losses["total"].backward()
    
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: data={losses['data']:.4e}, pde={losses['pde']:.4e}")
```

---

## Recipe 2: Advection-Diffusion (SST Interpolation)

Interpolate Sea Surface Temperature using physics constraints.

### Problem Setup

$$
\frac{\partial T}{\partial t} + \mathbf{v} \cdot \nabla T = D \nabla^2 T
$$

### Implementation

```python
from ununennium.models.pinn import (
    PINN, AdvectionDiffusionEquation, FourierMLP, LatinHypercubeSampler
)

# Physical parameters from oceanographic data
equation = AdvectionDiffusionEquation(
    diffusivity=100.0,       # m^2/s (horizontal eddy diffusivity)
    velocity=(0.1, 0.05),    # m/s (mean current)
)

# Use Fourier features for high-frequency patterns
network = FourierMLP(
    in_features=3,           # x, y, t
    layers=[256, 256, 256, 1],
    sigma=10.0,
    n_frequencies=256,
)

pinn = PINN(
    network=network,
    equation=equation,
    lambda_pde=1.0,
)

# Better space coverage with LHS
sampler = LatinHypercubeSampler(
    bounds=[(0, 100000), (0, 100000), (0, 86400)],  # 100km x 100km x 1 day
    n_points=5000,
)
```

### Training with Sparse SST Data

```python
# Load sparse SST observations
sst_data = load_sst_observations("sst_buoys.csv")
x_data = sst_data[["x", "y", "t"]].values
u_data = sst_data["temperature"].values

# Collocation points for physics constraint
x_collocation = sampler.sample()

# Train PINN
for epoch in range(20000):
    losses = pinn.compute_loss(x_data, u_data, x_collocation)
    # ...

# Predict on dense grid
x_grid = create_grid(100, 100, 24)  # 100x100 spatial, 24 time steps
sst_prediction = pinn.network(x_grid)
```

---

## Recipe 3: Boundary-Constrained PINN

Enforce boundary conditions for pollution dispersion.

### Setup

```python
from ununennium.models.pinn import PINN, AdvectionDiffusionEquation, DirichletBC

equation = AdvectionDiffusionEquation(
    diffusivity=10.0,
    velocity=(1.0, 0.0),  # Wind direction
)

# Boundary conditions
bc_inlet = DirichletBC(value=1.0)   # Pollution source
bc_walls = DirichletBC(value=0.0)   # No flux

pinn = PINN(
    network=network,
    equation=equation,
    lambda_pde=1.0,
    lambda_bc=10.0,
)

# Sample boundary points
x_bc_inlet = sample_boundary(side="left", n=500)
u_bc_inlet = torch.ones(500, 1)

x_bc_walls = sample_boundary(sides=["top", "bottom"], n=1000)
u_bc_walls = torch.zeros(1000, 1)
```

---

## Recipe 4: Multi-Output PINN

Simultaneously predict temperature and concentration.

```python
class CoupledNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = MLP([3, 128, 128], activation="tanh")
        self.temp_head = nn.Linear(128, 1)
        self.conc_head = nn.Linear(128, 1)
    
    def forward(self, x):
        features = self.shared(x)
        temperature = self.temp_head(features)
        concentration = self.conc_head(features)
        return torch.cat([temperature, concentration], dim=-1)

# Coupled PDEs
class CoupledEquation(PDEEquation):
    def residual(self, x, u):
        T, C = u[:, 0:1], u[:, 1:2]
        
        # Temperature equation
        T_t = grad(T, x, dim=2)
        T_xx = laplacian(T, x, dims=[0, 1])
        res_T = T_t - 0.1 * T_xx
        
        # Concentration equation (depends on temperature)
        C_t = grad(C, x, dim=2)
        C_xx = laplacian(C, x, dims=[0, 1])
        reaction = k(T) * C  # Temperature-dependent reaction
        res_C = C_t - 0.05 * C_xx + reaction
        
        return torch.cat([res_T, res_C], dim=-1)
```

---

## Training Tips

### Learning Rate Schedule

```python
# Warmup + cosine decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=20000,
    pct_start=0.1,
)
```

### Loss Weighting

| Phase | Data | PDE | BC |
|-------|------|-----|-----|
| Early (0-5k) | 1.0 | 0.1 | 1.0 |
| Middle (5k-15k) | 1.0 | 1.0 | 1.0 |
| Late (15k+) | 1.0 | 10.0 | 1.0 |

### Adaptive Collocation

```python
# Resample where residual is high
residuals = pinn.compute_residual(x_collocation)
high_residual_idx = residuals.abs() > threshold
x_collocation_new = importance_sample(high_residual_idx)
```

---

## Visualization

```python
import matplotlib.pyplot as plt

# Predict on grid
x_grid = create_meshgrid(100, 100)
u_pred = pinn.network(x_grid).reshape(100, 100)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(x_data[:, 0], x_data[:, 1], c=u_data, s=10)
axes[0].set_title("Observations")

axes[1].contourf(x_grid[..., 0], x_grid[..., 1], u_pred.detach())
axes[1].set_title("PINN Prediction")

residual = pinn.compute_residual(x_grid)
axes[2].contourf(x_grid[..., 0], x_grid[..., 1], residual.abs())
axes[2].set_title("PDE Residual")

plt.savefig("pinn_results.png")
```

---

## Next Steps

- [PINN Theory](../research/pinn-theory.md)
- [PINN API](../api/pinn.md)
