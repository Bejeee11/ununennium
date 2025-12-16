# GAN Recipes

Common workflows for Generative Adversarial Networks.

## Cloud Removal (CycleGAN)

Translate from "Cloudy" domain to "Clear" domain without paired data.

```python
from ununennium.models.gan import CycleGAN

model = CycleGAN(in_channels_a=4, in_channels_b=4)
# Domain A: Cloudy Sentinel-2
# Domain B: Clear Sentinel-2
```

## SAR-to-Optical (Pix2Pix)

Translate paired SAR images to Optical.

```python
from ununennium.models.gan import Pix2Pix

model = Pix2Pix(in_channels=2, out_channels=3)
# Input: VV, VH (2 channels)
# Output: RGB (3 channels)
```
