# Pipelines

This document describes the data flow pipelines in Ununennium, from raw data ingestion through model training and inference.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Training Pipeline](#training-pipeline)
3. [Inference Pipeline](#inference-pipeline)
4. [Streaming Architecture](#streaming-architecture)

---

## Pipeline Overview

```mermaid
graph TB
    subgraph "Data Sources"
        S3[Cloud Storage]
        LOCAL[Local Files]
        STAC[STAC Catalog]
    end
    
    subgraph "Ingestion"
        READ[Reader Pool]
        CACHE[Tile Cache]
    end
    
    subgraph "Processing"
        TILE[Tiling]
        PREP[Preprocessing]
        AUG[Augmentation]
    end
    
    subgraph "Execution"
        TRAIN[Training]
        INFER[Inference]
    end
    
    S3 --> READ
    LOCAL --> READ
    STAC --> READ
    READ --> CACHE
    CACHE --> TILE
    TILE --> PREP
    PREP --> AUG
    AUG --> TRAIN
    AUG --> INFER
```

---

## Training Pipeline

### Data Loading Flow

```mermaid
sequenceDiagram
    participant DS as Dataset
    participant TL as Tiler
    participant PP as Preprocessor
    participant AU as Augmenter
    participant DL as DataLoader
    participant GPU as GPU
    
    loop Each Epoch
        DL->>DS: get_sample(idx)
        DS->>TL: extract_tile(bounds)
        TL-->>DS: tile_data
        DS->>PP: preprocess(tile)
        PP-->>DS: normalized_tile
        DS->>AU: augment(tile)
        AU-->>DS: augmented_tile
        DS-->>DL: sample
        DL->>GPU: batch.to(device)
    end
```

### Tiling Strategy

Large rasters are divided into overlapping patches for training:

```mermaid
graph TB
    subgraph "Source Raster"
        SR[10980 x 10980 pixels]
    end
    
    subgraph "Tiling Parameters"
        SIZE[Tile Size: 512 x 512]
        STRIDE[Stride: 256 x 256]
        OVERLAP[Overlap: 50%]
    end
    
    subgraph "Output Tiles"
        T1[Tile 1]
        T2[Tile 2]
        TN[Tile N...]
    end
    
    SR --> SIZE
    SIZE --> STRIDE
    STRIDE --> OVERLAP
    OVERLAP --> T1
    OVERLAP --> T2
    OVERLAP --> TN
```

The overlap ratio $r$ determines the stride:

$$
s = \lfloor w \cdot (1 - r) \rfloor
$$

where $w$ is the tile width and $s$ is the stride.

### Spatial Sampling

To avoid spatial autocorrelation in train/validation splits:

```mermaid
graph LR
    subgraph "Block-based CV"
        B1[Block 1<br/>Train]
        B2[Block 2<br/>Val]
        B3[Block 3<br/>Train]
        B4[Block 4<br/>Test]
    end
```

Block size should exceed the spatial autocorrelation range:

$$
\text{block\_size} \geq 2 \cdot \text{correlation\_range}
$$

---

## Inference Pipeline

### Sliding Window Inference

```mermaid
graph TB
    subgraph "Input"
        IMG[Large Raster]
    end
    
    subgraph "Windowing"
        W1[Window 1]
        W2[Window 2]
        WN[Window N]
    end
    
    subgraph "Inference"
        MODEL[Model]
        P1[Pred 1]
        P2[Pred 2]
        PN[Pred N]
    end
    
    subgraph "Assembly"
        STITCH[Stitcher]
        OUT[Output Raster]
    end
    
    IMG --> W1
    IMG --> W2
    IMG --> WN
    W1 --> MODEL
    W2 --> MODEL
    WN --> MODEL
    MODEL --> P1
    MODEL --> P2
    MODEL --> PN
    P1 --> STITCH
    P2 --> STITCH
    PN --> STITCH
    STITCH --> OUT
```

### Overlap Blending

To avoid seam artifacts, overlapping predictions are blended using cosine weights:

$$
w(i, j) = \frac{1}{2}\left(1 - \cos\left(\frac{\pi \cdot \min(i, W-i)}{m}\right)\right) \cdot \frac{1}{2}\left(1 - \cos\left(\frac{\pi \cdot \min(j, H-j)}{m}\right)\right)
$$

where $m$ is the margin width.

```python
from ununennium.tiling import SlidingWindowInference

inferencer = SlidingWindowInference(
    model=model,
    tile_size=(512, 512),
    overlap=0.5,
    blend_mode="cosine",  # or "linear", "none"
)

output = inferencer.predict("input_raster.tif")
```

---

## Streaming Architecture

### COG Streaming

Cloud Optimized GeoTIFFs enable efficient partial reads:

```mermaid
graph LR
    subgraph "COG Structure"
        META[Metadata]
        OV1[Overview 1]
        OV2[Overview 2]
        FULL[Full Resolution]
    end
    
    subgraph "HTTP Range Requests"
        REQ1[Fetch Header]
        REQ2[Fetch Tile]
    end
    
    META --> REQ1
    FULL --> REQ2
```

### Memory Budget

For streaming large datasets:

| Parameter | Formula |
|-----------|---------|
| Tiles in memory | $\frac{\text{memory\_budget}}{\text{tile\_size}^2 \cdot \text{channels} \cdot \text{dtype\_size}}$ |
| Prefetch queue | $\text{batch\_size} \cdot \text{prefetch\_factor}$ |
| Cache size | $\frac{\text{cache\_budget}}{\text{avg\_tile\_bytes}}$ |

```python
from ununennium.io import StreamingDataset

dataset = StreamingDataset(
    urls=["s3://bucket/image1.tif", ...],
    tile_size=512,
    cache_size="2GB",
    prefetch=4,
)
```

---

## Pipeline Configuration

### YAML Configuration

```yaml
pipeline:
  data:
    sources:
      - type: stac
        url: "https://earth-search.aws.element84.com/v1"
        collection: sentinel-2-l2a
        
  tiling:
    size: [512, 512]
    overlap: 0.25
    
  preprocessing:
    - type: normalize
      method: percentile
      p: [2, 98]
    - type: indices
      compute: [ndvi, ndwi]
      
  augmentation:
    - type: rotate
      degrees: [-15, 15]
    - type: flip
      horizontal: true
      vertical: true
```

---

## Next Steps

- [Performance](performance.md) - Optimization strategies
- [API Reference: Data I/O](../api/data-io.md) - I/O API documentation
