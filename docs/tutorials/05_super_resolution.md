# Super-Resolution Tutorial

Enhancing visual quality of satellite imagery with ESRGAN.

## Concept

Upscale low-resolution (LR) inputs to high-resolution (HR) outputs.

```math
I_{HR} = G(I_{LR})
```

## Training

### Generator Loss

Combination of pixel loss, perceptual loss, and adversarial loss.

```math
\mathcal{L}_G = \lambda_1 \mathcal{L}_{L1} + \lambda_2 \mathcal{L}_{VGG} + \lambda_3 \mathcal{L}_{Adv}
```

### Code Example

```python
from ununennium.models.gan import ESRGAN
from ununennium.models.gan.losses import PerceptualLoss

generator = ESRGAN(scale_factor=4)
perceptual_criterion = PerceptualLoss()

# In training loop
fake_hr = generator(lr_img)
loss_p = perceptual_criterion(fake_hr, real_hr)
```
