# API Reference: Losses

Loss functions optimized for geospatial machine learning tasks.

---

## Segmentation Losses

### DiceLoss

Region-based loss that maximizes overlap between prediction and target.

$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

Where $p_i$ is the predicted probability and $g_i$ is the ground truth.

```python
from ununennium.losses import DiceLoss

loss_fn = DiceLoss(smooth=1.0)
loss = loss_fn(predictions, targets)
```

### FocalLoss

Addresses class imbalance by down-weighting easy examples.

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- $\gamma$: Focusing parameter (default: 2.0)
- $\alpha$: Class balancing weight

```python
from ununennium.losses import FocalLoss

loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
loss = loss_fn(predictions, targets)
```

### CombinedLoss

Weighted combination of multiple losses.

$$\mathcal{L} = \alpha \mathcal{L}_{Dice} + \beta \mathcal{L}_{CE} + \gamma \mathcal{L}_{Focal}$$

```python
from ununennium.losses import CombinedLoss, DiceLoss, FocalLoss
import torch.nn as nn

loss_fn = CombinedLoss([
    (DiceLoss(), 0.5),
    (nn.CrossEntropyLoss(), 0.3),
    (FocalLoss(), 0.2),
])
```

---

## GAN Losses

### AdversarialLoss

Standard GAN objective with optional label smoothing.

```python
from ununennium.models.gan.losses import AdversarialLoss

adv_loss = AdversarialLoss(mode="vanilla")  # or "lsgan", "wgan"
```

### PerceptualLoss

Feature-matching loss using VGG features.

$$\mathcal{L}_{perc} = \sum_l \| \phi_l(y) - \phi_l(\hat{y}) \|_2^2$$

```python
from ununennium.models.gan.losses import PerceptualLoss

perc_loss = PerceptualLoss(layers=["relu2_2", "relu3_3"])
```

---

## Physics-Informed Losses

### PDEResidualLoss

Computes the residual of a PDE at collocation points.

$$\mathcal{L}_{PDE} = \frac{1}{N} \sum_{i=1}^N \left| \mathcal{N}[u](x_i) - f(x_i) \right|^2$$

```python
from ununennium.models.pinn import PINN, DiffusionEquation

equation = DiffusionEquation(diffusivity=0.1)
pinn = PINN(network=model, equation=equation, lambda_pde=10.0)
```

---

## API Reference

::: ununennium.losses.segmentation
