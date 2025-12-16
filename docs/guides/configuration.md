# Configuration

Manage complex experiments with YAML config files.

## Structure

```yaml
experiment_name: "unet_sentinel2_v1"

data:
  path: "data/train.csv"
  batch_size: 32
  num_workers: 4

model:
  name: "unet"
  encoder: "resnet34"
  pretrained: true

optimizer:
  name: "adamw"
  lr: 1e-4
  weight_decay: 1e-2
```

## Loading

```python
from ununennium.config import load_config

cfg = load_config("config.yaml")
```
