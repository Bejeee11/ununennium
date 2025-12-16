# Generative Adversarial Networks (GANs)

Image-to-image translation models for Earth Observation applications.

---

## Overview

GANs enable:
- **SAR → Optical**: Generate optical-like images from radar
- **Cloud Removal**: Reconstruct cloud-covered regions
- **Super-Resolution**: Enhance spatial resolution
- **Domain Adaptation**: Transfer between sensors/seasons

---

## Pix2Pix (Conditional GAN)

Paired image-to-image translation requiring matched input-output pairs.

### Architecture

- **Generator**: U-Net with skip connections
- **Discriminator**: PatchGAN (classifies N×N patches)

### Objective Function

**Adversarial Loss:**
$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{x,z}[\log(1 - D(x, G(x, z)))]$$

**L1 Reconstruction:**
$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y,z}[\| y - G(x, z) \|_1]$$

**Total Loss:**
$$G^* = \arg\min_G \max_D \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

### Usage

```python
from ununennium.models.gan import Pix2Pix

model = Pix2Pix(
    in_channels=1,       # SAR input
    out_channels=3,      # RGB output
    ngf=64,              # Generator filters
    ndf=64,              # Discriminator filters
    n_layers=3,          # PatchGAN depth
)

# Training step
fake_optical = model.generator(sar_input)
d_real = model.discriminator(sar_input, real_optical)
d_fake = model.discriminator(sar_input, fake_optical.detach())

# Compute losses
g_loss, d_loss = model.compute_loss(sar_input, real_optical)
```

### Pretrained Models

| Task | Input | Output | FID | Download |
|------|-------|--------|-----|----------|
| SAR→Optical | S1 VV/VH | RGB | 42.3 | [Download](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/pix2pix_sar2optical.pt) |
| Cloud Removal | Cloudy RGB | Clear RGB | 38.7 | [Download](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/pix2pix_cloud_removal.pt) |

---

## CycleGAN (Unpaired Translation)

Domain adaptation without paired training data.

### Architecture

- **Two Generators**: $G: A \to B$, $F: B \to A$
- **Two Discriminators**: $D_A$, $D_B$

### Cycle Consistency

The key insight is that translation should be reversible:

$$x \to G(x) \to F(G(x)) \approx x$$
$$y \to F(y) \to G(F(y)) \approx y$$

**Cycle Loss:**
$$\mathcal{L}_{cyc}(G, F) = \mathbb{E}_x[\|F(G(x)) - x\|_1] + \mathbb{E}_y[\|G(F(y)) - y\|_1]$$

**Identity Loss (optional):**
$$\mathcal{L}_{id}(G, F) = \mathbb{E}_y[\|G(y) - y\|_1] + \mathbb{E}_x[\|F(x) - x\|_1]$$

### Usage

```python
from ununennium.models.gan import CycleGAN

model = CycleGAN(
    in_channels_a=1,     # Domain A (SAR)
    in_channels_b=3,     # Domain B (Optical)
    ngf=64,
    n_residual_blocks=9,
)

# Forward translation (A → B)
fake_optical = model(sar_image, direction="A2B")

# Backward translation (B → A)
fake_sar = model(optical_image, direction="B2A")

# Full cycle
reconstructed = model(fake_optical, direction="B2A")
```

### Pretrained Models

| Task | Domain A | Domain B | FID | Download |
|------|----------|----------|-----|----------|
| Season Transfer | Summer | Winter | 65.2 | [Download](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/cyclegan_summer2winter.pt) |
| Sensor Adaptation | Sentinel-1 | Sentinel-2 | 58.4 | [Download](https://github.com/olaflaitinen/ununennium/releases/download/v1.0.5/cyclegan_s1_to_s2.pt) |

---

## Training Tips

### Learning Rate Schedule
```python
# Linear decay after 100 epochs
scheduler = LinearDecayLR(optimizer, start_epoch=100, decay_epochs=100)
```

### Label Smoothing
```python
# Prevent discriminator overconfidence
real_label = 0.9  # instead of 1.0
fake_label = 0.1  # instead of 0.0
```

### Buffer for History
```python
# Store generated images for discriminator training
from ununennium.models.gan import ImageBuffer
buffer = ImageBuffer(pool_size=50)
```

---

## GAN Disclosure

Generated images should be marked for transparency:

```python
from ununennium.models.gan import add_disclosure_stamp

stamped = add_disclosure_stamp(
    generated_image,
    text="AI Generated - Ununennium",
    position="bottom-right",
)
```
