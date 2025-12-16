# GAN API

## Models

### `Pix2Pix`

Paired image-to-image translation.

```python
model = Pix2Pix(in_channels=3, out_channels=3)
```

### `CycleGAN`

Unpaired translation between domains A and B.

```python
model = CycleGAN(in_channels_a=3, in_channels_b=3)
```

### `ESRGAN`

Super-resolution model.

```python
model = ESRGAN(scale_factor=4)
```

## Losses

### `AdversarialLoss`

Generic GAN loss interface (BCE, MSE, Wasserstein).

### `PerceptualLoss`

VGG-based feature matching loss.

```math
\mathcal{L}_{p} = \| \phi(x) - \phi(y) \|_2^2
```
