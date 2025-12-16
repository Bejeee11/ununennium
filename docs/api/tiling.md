# API Reference: Tiling

Spatial sampling and tiling utilities for large-scale geospatial data.

---

## Tiling Strategies

### GridTiler

Extract tiles on a regular grid with configurable overlap.

```python
from ununennium.tiling import GridTiler

tiler = GridTiler(
    tile_size=(512, 512),
    stride=(256, 256),  # 50% overlap
    padding_mode="reflect",
)

tiles = tiler.tile(large_image)  # List of tiles
reconstructed = tiler.untile(tiles, original_shape)
```

### AdaptiveTiler

Adjust tiling based on content complexity.

```python
from ununennium.tiling import AdaptiveTiler

tiler = AdaptiveTiler(
    min_tile_size=(128, 128),
    max_tile_size=(512, 512),
    complexity_threshold=0.3,
)

tiles = tiler.tile(image)
```

---

## Spatial Sampling

### RandomGeoSampler

Random sampling with spatial constraints.

```python
from ununennium.tiling import RandomGeoSampler

sampler = RandomGeoSampler(
    dataset=geo_dataset,
    size=(256, 256),
    length=1000,  # samples per epoch
    roi=region_of_interest,
)

loader = DataLoader(dataset, sampler=sampler, batch_size=16)
```

### StratifiedGeoSampler

Class-balanced spatial sampling.

```python
from ununennium.tiling import StratifiedGeoSampler

sampler = StratifiedGeoSampler(
    dataset=geo_dataset,
    size=(256, 256),
    class_weights={0: 0.1, 1: 0.3, 2: 0.6},  # Oversample rare classes
)
```

### ImportanceSampler

Sample based on difficulty/uncertainty maps.

$$P(x, y) = \alpha P_{freq} + \beta P_{edge} + \gamma P_{error}$$

```python
from ununennium.tiling import ImportanceSampler

sampler = ImportanceSampler(
    dataset=geo_dataset,
    importance_map=uncertainty_map,
    temperature=1.0,
)
```

---

## Overlap Handling

### GaussianMerger

Merge overlapping predictions with Gaussian weighting.

$$P_{final}(i, j) = \frac{\sum_k w_k(i,j) P_k(i,j)}{\sum_k w_k(i,j)}$$

```python
from ununennium.tiling import GaussianMerger

merger = GaussianMerger(
    tile_size=(512, 512),
    sigma=0.25,  # Gaussian std relative to tile size
)

# During inference
for tile, coords in tiles:
    pred = model(tile)
    merger.add(pred, coords)

final_prediction = merger.merge()
```

### LinearMerger

Linear blending in overlap regions.

```python
from ununennium.tiling import LinearMerger

merger = LinearMerger(overlap=128)
```

---

## Inference Pipeline

### TiledInference

End-to-end tiled inference with automatic stitching.

```python
from ununennium.tiling import TiledInference

inferencer = TiledInference(
    model=model,
    tile_size=(512, 512),
    overlap=128,
    batch_size=8,
    device="cuda",
)

# Process entire large image
full_prediction = inferencer(large_geotiff_path)
```

---

## API Reference

::: ununennium.tiling
