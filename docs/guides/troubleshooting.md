# Troubleshooting

Common issues and solutions.

---

## Installation Issues

### GDAL Not Found

```bash
# Ubuntu/Debian
sudo apt-get install libgdal-dev

# macOS
brew install gdal

# Then install
pip install "ununennium[geo]"
```

### CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Data Loading Issues

### CRS Error

```
CRSError: Invalid CRS specification
```

**Solution**: Ensure your GeoTIFF has a valid CRS:

```python
import rasterio
with rasterio.open("image.tif") as src:
    print(src.crs)  # Should not be None
```

### Memory Error

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size
2. Use smaller tiles
3. Enable gradient checkpointing
4. Use mixed precision

---

## Training Issues

### NaN Loss

**Causes**:
- Learning rate too high
- Division by zero in indices

**Solutions**:
```python
# Lower learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Use epsilon in computations
ndvi = (nir - red) / (nir + red + 1e-8)
```

### Validation Better Than Test

**Cause**: Spatial leakage from random splitting.

**Solution**: Use block-based splitting.

---

## Inference Issues

### Seam Artifacts

**Solution**: Increase overlap and use blending:

```python
inferencer = SlidingWindowInference(
    model=model,
    overlap=0.5,
    blend_mode="cosine",
)
```

---

## Getting Help

- [GitHub Issues](https://github.com/olaflaitinen/ununennium/issues)
- [SUPPORT.md](../../SUPPORT.md)
