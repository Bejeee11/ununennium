# PINN API

## Physics-Informed Neural Networks

### `PINNModule`

Base class for physics-informed models. Defines the `pde_residual` method.

```python
class HeatEquationPINN(PINNModule):
    def pde_residual(self, x, t, u):
        u_t = grad(u, t)
        u_xx = grad(grad(u, x), x)
        return u_t - self.D * u_xx
```

## Samplers

### `CollocationSampler`

Samples points in the domain $\Omega \times [0, T]$ for PDE loss computation.

- `UniformSampler`: Uniform grid.
- `LatinHypercubeSampler`: LHS sampling for better coverage.

## Boundary Conditions

### `DirichletBC`

Fixed value boundary condition: $u(x) = g(x)$.

### `NeumannBC`

Fixed derivative boundary condition: $\frac{\partial u}{\partial n} = h(x)$.
