# System Architecture Overview

This document provides a comprehensive overview of Ununennium's architecture, design principles, and component interactions.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [System Overview](#system-overview)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Extension Points](#extension-points)

---

## Design Principles

Ununennium is built on the following architectural principles:

| Principle | Description |
|-----------|-------------|
| **CRS Preservation** | Coordinate Reference Systems are tracked through the entire pipeline |
| **Lazy Evaluation** | Data is loaded on-demand to handle datasets larger than memory |
| **Modularity** | Components are loosely coupled and independently testable |
| **Type Safety** | Strong typing with runtime validation for geospatial constraints |
| **Reproducibility** | Deterministic operations with explicit random state management |

---

## System Overview

```mermaid
graph TB
    subgraph "Data Layer"
        COG[COG Reader]
        STAC[STAC Client]
        ZARR[Zarr Store]
    end
    
    subgraph "Core Layer"
        GT[GeoTensor]
        GB[GeoBatch]
        TILE[Tiler]
        SAMP[Sampler]
    end
    
    subgraph "Processing Layer"
        PREP[Preprocessing]
        AUG[Augmentation]
        IDX[Spectral Indices]
    end
    
    subgraph "Model Layer"
        REG[Model Registry]
        BACK[Backbones]
        HEAD[Task Heads]
        ARCH[Architectures]
    end
    
    subgraph "Training Layer"
        TRAIN[Trainer]
        CB[Callbacks]
        OPT[Optimizers]
        LOSS[Loss Functions]
    end
    
    subgraph "Evaluation Layer"
        MET[Metrics]
        CAL[Calibration]
        VIS[Visualization]
    end
    
    subgraph "Export Layer"
        ONNX[ONNX Export]
        TS[TorchScript]
        QUANT[Quantization]
    end
    
    COG --> GT
    STAC --> GT
    ZARR --> GT
    GT --> TILE
    TILE --> SAMP
    SAMP --> GB
    GB --> PREP
    PREP --> AUG
    AUG --> IDX
    IDX --> ARCH
    REG --> ARCH
    BACK --> ARCH
    HEAD --> ARCH
    ARCH --> TRAIN
    CB --> TRAIN
    OPT --> TRAIN
    LOSS --> TRAIN
    TRAIN --> MET
    MET --> CAL
    TRAIN --> ONNX
    TRAIN --> TS
```

---

## Component Architecture

### Data Layer

The data layer provides cloud-native access to geospatial imagery.

```mermaid
classDiagram
    class GeoReader {
        <<interface>>
        +read(path: str) GeoTensor
        +read_window(path: str, bounds: BoundingBox) GeoTensor
    }
    
    class COGReader {
        +read(path: str) GeoTensor
        +read_window(path: str, bounds: BoundingBox) GeoTensor
        +get_overview_level(scale: float) int
    }
    
    class STACClient {
        +search(bbox: tuple, datetime: str) Iterator
        +read_item(item: Item) GeoTensor
    }
    
    class ZarrReader {
        +read(path: str) GeoTensor
        +read_chunk(path: str, chunk_id: tuple) GeoTensor
    }
    
    GeoReader <|-- COGReader
    GeoReader <|-- STACClient
    GeoReader <|-- ZarrReader
```

### Core Layer

The core layer provides CRS-aware tensor abstractions.

| Component | Responsibility |
|-----------|---------------|
| `GeoTensor` | Single CRS-aware tensor with metadata |
| `GeoBatch` | Batch of GeoTensors with consistent CRS |
| `Tiler` | Patch extraction with overlap handling |
| `Sampler` | Spatial sampling strategies |

#### GeoTensor Structure

```
GeoTensor
├── data: torch.Tensor          # Shape: (C, H, W)
├── crs: CRS                    # Coordinate Reference System
├── transform: Affine           # Pixel-to-world transform
├── bounds: BoundingBox         # Geographic extent
├── resolution: tuple[float, float]  # Pixel size in CRS units
└── nodata: float | None        # NoData value
```

### Model Layer

The model layer follows a registry pattern for architecture composition.

```mermaid
graph LR
    subgraph "Registry Pattern"
        REG[ModelRegistry]
        
        subgraph "Backbones"
            RN[ResNet]
            EN[EfficientNet]
            VIT[ViT]
            SWIN[Swin]
        end
        
        subgraph "Heads"
            SEG[SegmentationHead]
            CLS[ClassificationHead]
            DET[DetectionHead]
        end
        
        subgraph "Architectures"
            UNET[U-Net]
            DL[DeepLabV3+]
            FPN[FPN]
        end
    end
    
    REG --> RN
    REG --> EN
    REG --> VIT
    REG --> SWIN
    REG --> SEG
    REG --> CLS
    REG --> DET
    REG --> UNET
    REG --> DL
    REG --> FPN
```

### Training Layer

The training layer provides a flexible trainer with callback hooks.

```mermaid
sequenceDiagram
    participant User
    participant Trainer
    participant Callbacks
    participant Model
    participant DataLoader
    
    User->>Trainer: fit(epochs)
    loop Each Epoch
        Trainer->>Callbacks: on_epoch_start()
        loop Each Batch
            DataLoader->>Trainer: batch
            Trainer->>Callbacks: on_batch_start()
            Trainer->>Model: forward(batch)
            Model-->>Trainer: predictions
            Trainer->>Trainer: compute_loss()
            Trainer->>Model: backward()
            Trainer->>Callbacks: on_batch_end()
        end
        Trainer->>Callbacks: on_epoch_end()
    end
    Trainer-->>User: history
```

---

## Data Flow

### Training Pipeline

```mermaid
graph LR
    subgraph "Input"
        S3[(Cloud Storage)]
        FS[(Local Files)]
    end
    
    subgraph "Loading"
        STREAM[Streaming Reader]
        CACHE[(Tile Cache)]
    end
    
    subgraph "Preprocessing"
        NORM[Normalize]
        MASK[Cloud Mask]
        IDX[Indices]
    end
    
    subgraph "Batching"
        SAMP[Sampler]
        LOAD[DataLoader]
    end
    
    subgraph "Training"
        GPU[GPU]
        MODEL[Model]
    end
    
    S3 --> STREAM
    FS --> STREAM
    STREAM --> CACHE
    CACHE --> NORM
    NORM --> MASK
    MASK --> IDX
    IDX --> SAMP
    SAMP --> LOAD
    LOAD --> GPU
    GPU --> MODEL
```

### Inference Pipeline

```mermaid
graph LR
    subgraph "Input"
        IMG[Large Raster]
    end
    
    subgraph "Tiling"
        TILE[Tiler]
        WIN[Sliding Window]
    end
    
    subgraph "Inference"
        BATCH[Batch]
        MODEL[Model]
        PRED[Predictions]
    end
    
    subgraph "Assembly"
        STITCH[Stitcher]
        BLEND[Overlap Blend]
    end
    
    subgraph "Output"
        OUT[Output Raster]
    end
    
    IMG --> TILE
    TILE --> WIN
    WIN --> BATCH
    BATCH --> MODEL
    MODEL --> PRED
    PRED --> STITCH
    STITCH --> BLEND
    BLEND --> OUT
```

---

## Extension Points

Ununennium provides several extension points for customization:

| Extension Point | Interface | Purpose |
|-----------------|-----------|---------|
| Custom Readers | `GeoReader` | New data formats |
| Custom Backbones | `BackboneRegistry` | New architectures |
| Custom Losses | `LossFunction` | Domain-specific objectives |
| Custom Callbacks | `Callback` | Training customization |
| Custom Metrics | `Metric` | Evaluation criteria |

### Registering Custom Components

```python
from ununennium.models import register_backbone

@register_backbone("my_backbone")
class MyBackbone(nn.Module):
    """Custom backbone implementation."""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Implementation
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Return multi-scale features
        pass
```

---

## Next Steps

- [Data Model](data-model.md) - Deep dive into GeoTensor
- [Pipelines](pipelines.md) - Detailed pipeline architecture
- [Performance](performance.md) - Optimization guide
