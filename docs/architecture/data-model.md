# Data Model

## GeoTensor

The `GeoTensor` is the primary citizen of Ununennium. It wraps a PyTorch tensor and ensures that geospatial operations are mathematically consistent.

### Affine Representations

The affine transform matrix $M$ defines the relationship between pixel space $(u, v)$ and projected map space $(x, y)$:

```math
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
```

Where:
- $a$: pixel width
- $e$: pixel height (usually negative)
- $c, f$: top-left origin coordinates
- $b, d$: rotation terms (usually 0 for north-up images)

### Coordinate Propagation

When performing operations like pooling or strided convolution, the affine transform is updated automatically.

For a stride $s$:

```math
M_{new} = M_{old} \times \begin{bmatrix} s & 0 & 0 \\ 0 & s & 0 \\ 0 & 0 & 1 \end{bmatrix}
```

## GeoBatch

A `GeoBatch` is a collection of `GeoTensor`s. It supports stacking tensors even if they are from different non-overlapping regions, tracking their individual metadata for reconstruction.

## Bounding Boxes

Bounding boxes are represented as `(minx, miny, maxx, maxy)` tuples, automatically handled in CRS operations.
