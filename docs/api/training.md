# Training Module API Reference

The `ununennium.training` module implements a highly customizable, PyTorch-native training loop geared towards Earth Observation tasks. It supports mixed precision, gradient accumulation, and callbacks.

## 1. Trainer

**`ununennium.training.trainer.Trainer`**

The orchestrator of the training process.

### Signature

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        scheduler: LRScheduler | None = None,
        callbacks: list[Callback] | None = None,
        config: TrainerConfig | None = None
    )
```

### Configuration (`TrainerConfig`)

| Field | Default | Description |
|-------|---------|-------------|
| `epochs` | `100` | Max training epochs. |
| `device` | `'cuda'` | Computation device. |
| `mixed_precision` | `True` | Enable FP16 (AMP) for 2x speedup. |
| `accumulation_steps` | `1` | Gradient accumulation for large batch simulation. |
| `gradient_clip` | `1.0` | Max norm for gradient clipping. |

### Methods

#### `fit`

Starts the training loop.

```python
def fit(self, epochs: int | None = None) -> dict[str, list[float]]
```

**Returns:**
A dictionary containing the history of all logged metrics (e.g., `train_loss`, `val_IoU`).

---

## 2. Callbacks

Callbacks allow hooking into the training lifecycle. Base class: `ununennium.training.callbacks.Callback`.

### Built-in Callbacks

### `CheckpointCallback`

Saves model weights based on metric monitoring.

```python
class CheckpointCallback(Callback):
    def __init__(self, 
                 dirpath: str, 
                 monitor: str = "val_loss", 
                 mode: str = "min", 
                 save_top_k: int = 1)
```

*   **save_top_k**: Keep only the best $K$ models.
*   **mode**: `min` for loss, `max` for accuracy/IoU.

### `EarlyStopping`

Halts training if the monitored metric stops improving.

```python
class EarlyStopping(Callback):
    def __init__(self, 
                 monitor: str = "val_loss", 
                 patience: int = 10, 
                 min_delta: float = 0.0)
```

---

## 3. Distributed Training (DDP)

The `Trainer` is compatible with `torch.distributed`.

**Usage Pattern:**

```python
# Launch with torchrun
# torchrun --nproc_per_node=4 train.py

def main():
    dist.init_process_group(backend="nccl")
    model = DDP(model.to(rank), device_ids=[rank])
    # Trainer handles the rest implicitly via device placement
```

> [!NOTE]
> Ensure your `DataLoader` uses `DistributedSampler` when running in DDP mode to prevent workers from seeing the same data.
