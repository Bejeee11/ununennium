# System Architecture

Ununennium is designed as a layered framework for high-performance geospatial machine learning.

```mermaid
graph TD
    User[User API] --> High[High-Level API]
    User --> Core[Core Primitives]
    
    subgraph "High-Level API"
        Models[Model Registry]
        Trainer[Trainer / Callbacks]
        Pipe[Pipelines]
    end
    
    subgraph "Core Primitives"
        GT[GeoTensor]
        GB[GeoBatch]
        IO[Streaming I/O]
    end
    
    subgraph "Backend"
        Torch[PyTorch]
        GDAL[GDAL/Rasterio]
        CUDA[CUDA Kernels]
    end
    
    High --> Core
    Core --> Backend
```

## Core Components

### 1. GeoTensor
The fundamental data structure extending `torch.Tensor` with geospatial metadata:
- **Affine Transform**: Maps pixel coordinates to map coordinates.
- **CRS**: Coordinate Reference System (WKT/EPSG).
- **Bounding Box**: Spatial extent management.

```math
\text{Pixel}(col, row) \xrightarrow{M} \text{Map}(x, y)
```

### 2. Streaming I/O
Lazy loading architecture for files larger than memory (e.g., 100GB+ Sentinel-2 scenes). Uses windowed reading with caching.

### 3. Model Registry
Standardized interface for 15+ models (CNN, ViT, GAN, PINN), supporting:
- Automatic weight downloading
- Config-driven instantiation
- Standardized forward/predict signatures

## Data Flow

```mermaid
sequenceDiagram
    participant D as Data Storage (S3/Local)
    participant I as IO Layer
    participant T as Tiler
    participant M as Model
    
    D->>I: Stream Window
    I->>T: Raw Patch
    T->>T: Augment & Normalize
    T->>M: Forward Pass
    M->>T: Prediction
    T->>I: Write Result
```
