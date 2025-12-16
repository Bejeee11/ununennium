# Configuration

This guide covers the configuration system in Ununennium.

---

## Configuration Files

### YAML Format

```yaml
# config.yaml
model:
  name: unet_resnet50
  in_channels: 12
  num_classes: 10
  pretrained: true

training:
  epochs: 100
  batch_size: 16
  learning_rate: 1e-4
  weight_decay: 1e-5
  mixed_precision: true
  gradient_accumulation_steps: 4

data:
  train_paths: ["data/train/*.tif"]
  val_paths: ["data/val/*.tif"]
  tile_size: 256
  overlap: 0.25
  num_workers: 4

callbacks:
  - type: checkpoint
    path: checkpoints/
    monitor: val_iou
  - type: early_stopping
    patience: 10
```

### Loading Configuration

```python
from ununennium.utils import load_config

config = load_config("config.yaml")
model = create_model(**config["model"])
```

---

## Environment Variables

```bash
# Override config values
export UU_BATCH_SIZE=32
export UU_DEVICE=cuda:1
```

```python
import os
batch_size = int(os.environ.get("UU_BATCH_SIZE", config["training"]["batch_size"]))
```

---

## Experiment Organization

```
experiments/
├── exp_001/
│   ├── config.yaml
│   ├── checkpoints/
│   ├── logs/
│   └── results/
└── exp_002/
    └── ...
```

---

## See Also

- [Reproducibility](reproducibility.md)
