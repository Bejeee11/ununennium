# Training API

The training module provides a flexible trainer with callbacks and distributed training support.

---

## Trainer

Main training loop abstraction.

### Class Definition

```python
class Trainer:
    """Training loop manager.
    
    Parameters
    ----------
    model : nn.Module
        Model to train.
    optimizer : Optimizer
        PyTorch optimizer.
    loss_fn : nn.Module | Callable
        Loss function.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader | None, optional
        Validation data loader.
    callbacks : list[Callback] | None, optional
        Training callbacks.
    mixed_precision : bool, optional
        Enable AMP. Default: False.
    gradient_accumulation_steps : int, optional
        Accumulation steps. Default: 1.
    gradient_clip_norm : float | None, optional
        Gradient clipping norm.
    distributed : bool, optional
        Enable DDP. Default: False.
    device : str | torch.device, optional
        Training device. Default: auto-detect.
    """
```

### Methods

#### `fit`

```python
def fit(
    self,
    epochs: int,
    start_epoch: int = 0,
) -> dict[str, list[float]]:
    """Train the model.
    
    Parameters
    ----------
    epochs : int
        Number of epochs to train.
    start_epoch : int, optional
        Starting epoch (for resumption). Default: 0.
    
    Returns
    -------
    dict[str, list[float]]
        Training history with metrics.
    """
```

#### `validate`

```python
def validate(self) -> dict[str, float]:
    """Run validation loop.
    
    Returns
    -------
    dict[str, float]
        Validation metrics.
    """
```

### Example

```python
from ununennium.training import Trainer, CheckpointCallback
from ununennium.models import create_model
from ununennium.losses import DiceLoss
import torch

model = create_model("unet_resnet50", in_channels=12, num_classes=10)

trainer = Trainer(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    loss_fn=DiceLoss(),
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=[
        CheckpointCallback("checkpoints/", monitor="val_iou"),
    ],
    mixed_precision=True,
    gradient_accumulation_steps=4,
)

history = trainer.fit(epochs=100)
```

---

## Callbacks

### Base Callback

```python
class Callback:
    """Base callback class.
    
    Methods
    -------
    on_train_start(trainer)
    on_train_end(trainer)
    on_epoch_start(trainer, epoch)
    on_epoch_end(trainer, epoch, logs)
    on_batch_start(trainer, batch_idx)
    on_batch_end(trainer, batch_idx, logs)
    """
```

### CheckpointCallback

```python
class CheckpointCallback(Callback):
    """Save model checkpoints.
    
    Parameters
    ----------
    path : str | Path
        Directory to save checkpoints.
    monitor : str, optional
        Metric to monitor. Default: "val_loss".
    mode : str, optional
        "min" or "max". Default: "min".
    save_best_only : bool, optional
        Only save best model. Default: True.
    save_last : bool, optional
        Always save latest. Default: True.
    """
```

### EarlyStoppingCallback

```python
class EarlyStoppingCallback(Callback):
    """Stop training when metric stops improving.
    
    Parameters
    ----------
    monitor : str, optional
        Metric to monitor. Default: "val_loss".
    patience : int, optional
        Epochs to wait. Default: 10.
    mode : str, optional
        "min" or "max". Default: "min".
    min_delta : float, optional
        Minimum change threshold. Default: 1e-4.
    """
```

### LRSchedulerCallback

```python
class LRSchedulerCallback(Callback):
    """Learning rate scheduler.
    
    Parameters
    ----------
    scheduler : LRScheduler
        PyTorch scheduler instance.
    step_on : str, optional
        When to step: "epoch" or "batch". Default: "epoch".
    """
```

### Example: Custom Callback

```python
from ununennium.training import Callback

class LoggingCallback(Callback):
    def on_epoch_end(self, trainer, epoch, logs):
        print(f"Epoch {epoch}: loss={logs['train_loss']:.4f}")
```

---

## Learning Rate Schedulers

Common scheduler configurations:

| Scheduler | Description |
|-----------|-------------|
| `CosineAnnealingLR` | Cosine decay |
| `OneCycleLR` | Super-convergence |
| `ReduceLROnPlateau` | Reduce on plateau |

### Example

```python
from ununennium.training import LRSchedulerCallback
import torch.optim.lr_scheduler as sched

scheduler = sched.CosineAnnealingLR(optimizer, T_max=100)
callback = LRSchedulerCallback(scheduler)
```

---

## Distributed Training

### Multi-GPU Setup

```python
# Launch: torchrun --nproc_per_node=4 train.py

trainer = Trainer(
    model=model,
    distributed=True,
    # ...
)
```

### Distributed Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | "nccl" | Communication backend |
| `find_unused_params` | False | DDP parameter |

---

## Metrics Integration

```python
from ununennium.metrics import IoU, Dice

trainer = Trainer(
    # ...
    metrics=[
        IoU(num_classes=10),
        Dice(num_classes=10),
    ],
)
```
