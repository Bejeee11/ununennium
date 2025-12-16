# Ingest and Tiling

Processing large satellite imagery requires efficient tiling strategies.

## Tiling Strategy

We use overlap-tile processing to avoid edge effects.

```python
from ununennium.tiling import SlidingWindowTiler

tiler = SlidingWindowTiler(
    tensor_shape=(10000, 10000),
    tile_size=512,
    overlap=64
)

for window in tiler:
    # window is ((row_start, row_end), (col_start, col_end))
    patch = read_window(path, window)
```

## Creating a Dataset

```python
from torch.utils.data import Dataset

class SentinelDataset(Dataset):
    def __init__(self, tiler, image_path):
        self.tiler = tiler
        self.path = image_path
        
    def __len__(self):
        return len(self.tiler)
        
    def __getitem__(self, idx):
        window = self.tiler[idx]
        return read_window(self.path, window)
```
