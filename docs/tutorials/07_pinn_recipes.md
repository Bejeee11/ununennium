# PINN Recipes

Physics-Informed Neural Networks examples.

## Advection-Diffusion

Modeling pollutant dispersion.

```math
\frac{\partial u}{\partial t} + v \frac{\partial u}{\partial x} = D \frac{\partial^2 u}{\partial x^2}
```

### Implementation

```python
class DispersionPINN(PINNModule):
    def __init__(self, v=1.0, D=0.1):
        super().__init__()
        self.net = MLP(input_dim=2, output_dim=1)
        self.v = v
        self.D = D

    def pde_residual(self, x, t, u):
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        return u_t + self.v * u_x - self.D * u_xx
```

### Training

Sample points from domain and train.

```python
sampler = LatinHypercubeSampler(domain_bounds)
points = sampler.sample(1000)
loss = model.compute_loss(points)
```
