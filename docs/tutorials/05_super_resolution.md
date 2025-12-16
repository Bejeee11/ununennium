# Tutorial 05: Super-Resolution

This tutorial covers super-resolution techniques to enhance the spatial resolution of satellite imagery.

---

## Super-Resolution Overview

Super-resolution reconstructs high-resolution images from low-resolution inputs.

$$ I_{HR} = f_\theta(I_{LR}) $$

### Applications

| Application | Scale | Benefit |
|-------------|-------|---------|
| Sentinel-2 enhancement | 2-4x | 10m to 2.5-5m |
| Pan-sharpening | 4x | Fuse multispectral with panchromatic |
| Historical imagery | 4-8x | Improve archive data |

---

## Single-Image Super-Resolution

### ESRGAN Model

```python
from ununennium.models.gan import ESRGAN

model = ESRGAN(
    in_channels=12,     # Multispectral bands
    scale_factor=4,     # 4x upscaling
    n_rrdb=23,          # RRDB blocks
)
```

### Training

```python
from ununennium.losses import PerceptualLoss, AdversarialLoss
from ununennium.training import GANTrainer

# Loss components
perceptual_loss = PerceptualLoss(layers=["conv3_4", "conv4_4"])
adversarial_loss = AdversarialLoss(mode="hinge")
l1_loss = nn.L1Loss()

def generator_loss(fake, real, d_fake):
    return (
        1.0 * l1_loss(fake, real) +
        0.1 * perceptual_loss(fake, real) +
        0.01 * adversarial_loss.generator_loss(d_fake)
    )

trainer = GANTrainer(
    generator=model.generator,
    discriminator=model.discriminator,
    g_loss_fn=generator_loss,
    d_loss_fn=adversarial_loss.discriminator_loss,
)

history = trainer.fit(train_loader, epochs=100)
```

---

## Data Preparation

### Creating LR-HR Pairs

```python
from ununennium.preprocessing import downsample

def create_pair(hr_image, scale=4):
    """Create low-resolution input from high-resolution target."""
    lr_image = downsample(
        hr_image,
        scale=scale,
        method="bicubic",
        antialias=True,
    )
    return lr_image, hr_image
```

### Dataset

```python
from ununennium.datasets import SuperResolutionDataset

dataset = SuperResolutionDataset(
    hr_paths=["high_res/*.tif"],
    scale_factor=4,
    patch_size=256,  # HR patch size
    augment=True,
)

sample = dataset[0]
print(f"LR shape: {sample['lr'].shape}")   # (12, 64, 64)
print(f"HR shape: {sample['hr'].shape}")   # (12, 256, 256)
```

---

## Metrics

### PSNR (Peak Signal-to-Noise Ratio)

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)
$$

### SSIM (Structural Similarity)

$$
\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
$$

```python
from ununennium.metrics import PSNR, SSIM

psnr = PSNR()
ssim = SSIM()

psnr.update(sr_images, hr_images)
ssim.update(sr_images, hr_images)

print(f"PSNR: {psnr.compute():.2f} dB")
print(f"SSIM: {ssim.compute():.4f}")
```

---

## Real-World Super-Resolution

### Degradation Modeling

Real images have complex degradations beyond bicubic downsampling:

```python
from ununennium.preprocessing import RealDegradation

degradation = RealDegradation(
    blur_kernels=["gaussian", "motion", "defocus"],
    noise_level=(0.01, 0.05),
    jpeg_quality=(30, 95),
    scale=4,
)

lr_image = degradation(hr_image)
```

### Real-ESRGAN Training

```python
from ununennium.models.gan import RealESRGAN

model = RealESRGAN(
    in_channels=3,
    scale_factor=4,
    degradation=degradation,
)
```

---

## Pan-Sharpening

Fuse low-resolution multispectral with high-resolution panchromatic:

```python
from ununennium.models import create_model

# Pan-sharpening model
model = create_model(
    "pansharpening_resnet",
    ms_channels=4,   # Multispectral bands
    pan_channels=1,  # Panchromatic band
    out_channels=4,  # High-res multispectral
    scale_factor=4,
)

# Input: (B, 4, H, W) MS + (B, 1, 4H, 4W) PAN
# Output: (B, 4, 4H, 4W) High-res MS
```

---

## Inference

### Single Image

```python
# Load low-resolution image
lr_image = uu.io.read_geotiff("sentinel2_10m.tif")

# Super-resolve
model.eval()
with torch.no_grad():
    sr_image = model(lr_image.data.unsqueeze(0).cuda())

# Save with updated resolution
sr_geotensor = GeoTensor(
    data=sr_image[0].cpu(),
    crs=lr_image.crs,
    transform=lr_image.transform * Affine.scale(0.25),  # 4x finer
)
sr_geotensor.save("sentinel2_2.5m.tif")
```

---

## Next Steps

- [Tutorial 06: GAN Recipes](06_gan_recipes.md)
- [GAN API Reference](../api/gan.md)
