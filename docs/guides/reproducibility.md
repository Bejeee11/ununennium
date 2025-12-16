# Reproducibility

This guide covers practices for reproducible geospatial ML experiments.

---

## Random Seeds

### Setting Seeds

```python
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Per-Experiment Seeds

```yaml
experiment:
  seed: 42
  data_seed: 123
  model_seed: 456
```

---

## Configuration Management

### YAML Configuration

```yaml
model:
  name: unet_resnet50
  in_channels: 12
  num_classes: 10

training:
  epochs: 100
  batch_size: 16
  learning_rate: 1e-4

data:
  tile_size: 256
  overlap: 0.25
```

### Logging Configuration

```python
from ununennium.utils import save_config

save_config(config, "experiment_001/config.yaml")
```

---

## Checkpointing

```python
from ununennium.training import CheckpointCallback

callback = CheckpointCallback(
    path="checkpoints/",
    save_optimizer=True,
    save_config=True,
)
```

---

## Version Tracking

```python
import ununennium

print(f"ununennium: {ununennium.__version__}")
print(f"torch: {torch.__version__}")
print(f"numpy: {np.__version__}")
```

---

## See Also

- [Configuration Guide](configuration.md)
- [CONTRIBUTING.md](../../CONTRIBUTING.md)
