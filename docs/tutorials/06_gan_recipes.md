# Tutorial 06: GAN Recipes

This tutorial provides practical recipes for training GANs for Earth observation applications.

---

## Recipe 1: SAR-to-Optical Translation (Pix2Pix)

Translate SAR imagery to optical RGB.

### Setup

```python
from ununennium.models.gan import Pix2Pix
from ununennium.training import GANTrainer

model = Pix2Pix(
    in_channels=2,       # VV, VH
    out_channels=3,      # RGB
    ngf=64,
    ndf=64,
    lambda_l1=100.0,
)
```

### Dataset

```python
from ununennium.datasets import PairedImageDataset

dataset = PairedImageDataset(
    source_paths=["sar/*.tif"],
    target_paths=["optical/*.tif"],
    tile_size=256,
    augment=True,
)
```

### Training

```python
trainer = GANTrainer(
    generator=model.generator,
    discriminator=model.discriminator,
    g_optimizer=torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
    d_optimizer=torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
    n_critic=1,
)

history = trainer.fit(train_loader, epochs=200)
```

---

## Recipe 2: Cloud Removal (CycleGAN)

Remove clouds using unpaired clean/cloudy images.

### Setup

```python
from ununennium.models.gan import CycleGAN

model = CycleGAN(
    in_channels_a=3,         # Cloudy RGB
    in_channels_b=3,         # Clean RGB
    ngf=64,
    n_residual=9,
    lambda_cycle=10.0,
    lambda_identity=0.5,
)
```

### Dataset

```python
from ununennium.datasets import UnpairedImageDataset

# No need for paired data
dataset_cloudy = UnpairedImageDataset(paths=["cloudy/*.tif"], tile_size=256)
dataset_clean = UnpairedImageDataset(paths=["clean/*.tif"], tile_size=256)
```

### Training Loop

```python
for epoch in range(200):
    for cloudy, clean in zip(loader_cloudy, loader_clean):
        # Generator update
        g_loss = model.generator_step(cloudy, clean)
        
        # Discriminator update
        d_loss = model.discriminator_step(cloudy, clean)
    
    print(f"Epoch {epoch}: G={g_loss:.4f}, D={d_loss:.4f}")
```

---

## Recipe 3: Super-Resolution (ESRGAN)

4x resolution enhancement.

### Setup

```python
from ununennium.models.gan import ESRGAN

model = ESRGAN(
    in_channels=12,
    scale_factor=4,
    n_rrdb=23,
)
```

### Loss Weights

| Loss | Weight | Purpose |
|------|--------|---------|
| L1 Pixel | 1.0 | Reconstruction |
| Perceptual | 0.1 | Texture quality |
| Adversarial | 0.01 | Sharpness |

```python
from ununennium.losses import PerceptualLoss, AdversarialLoss

perceptual = PerceptualLoss(layers=["conv4_4"])
adversarial = AdversarialLoss(mode="hinge")

def g_loss(fake, real, d_fake):
    return (
        1.0 * F.l1_loss(fake, real) +
        0.1 * perceptual(fake, real) +
        0.01 * adversarial.g_loss(d_fake)
    )
```

---

## Recipe 4: Semantic-Guided Translation

Use segmentation masks to guide translation.

```python
from ununennium.models.gan import SemanticPix2Pix

model = SemanticPix2Pix(
    in_channels=3,
    out_channels=3,
    num_classes=10,      # Semantic classes
    use_instance=True,   # Instance normalization
)

# Input: image + one-hot semantic map
fake = model.generator(image, semantic_map)
```

---

## Training Tips

### Learning Rate

| Phase | G LR | D LR |
|-------|------|------|
| Warmup (epochs 1-10) | 1e-4 | 1e-4 |
| Main (epochs 10-150) | 2e-4 | 2e-4 |
| Decay (epochs 150+) | Linear to 0 | Linear to 0 |

### Discriminator Regularization

```python
# Gradient penalty for WGAN-GP
def gradient_penalty(discriminator, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    
    d_out = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True,
    )[0]
    
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
```

### Mode Collapse Prevention

- Use spectral normalization in discriminator
- Label smoothing (real=0.9, fake=0.1)
- Feature matching loss
- Progressive growing

---

## Evaluation

### FID (Frechet Inception Distance)

```python
from ununennium.metrics import FID

fid = FID()
fid.update(real_images, fake_images)
print(f"FID: {fid.compute():.2f}")  # Lower is better
```

### Visual Quality Assessment

```python
# Generate comparison grid
from ununennium.visualization import make_comparison_grid

grid = make_comparison_grid(
    input_images,
    generated_images,
    target_images,
    labels=["Input", "Generated", "Target"],
)
grid.save("comparison.png")
```

---

## Next Steps

- [Tutorial 07: PINN Recipes](07_pinn_recipes.md)
- [GAN Theory](../research/gan-theory.md)
- [GAN API](../api/gan.md)
