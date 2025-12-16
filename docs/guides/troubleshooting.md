# Troubleshooting

Common issues and solutions.

## CUDA OOM (Out of Memory)

**Symptom**: `RuntimeError: CUDA out of memory`

**Fixes**:
1. Reduce `batch_size`.
2. Enable `gradient_checkpointing` for the model.
3. Use mixed precision (`fp16`).
4. Clear cache: `torch.cuda.empty_cache()` (use sparingly).

## GDAL/Rasterio Installation

**Symptom**: `ImportError: /usr/lib/libgdal.so...`

**Fixes**:
- Use `conda` for easiest geospatial dependency management.
- Or use `pip install ununennium[geo]` which attempts to pull binary wheels.

## NaN Loss

**Symptom**: Loss becomes `nan`.

**Fixes**:
- Check learning rate (too high?).
- Check for division by zero in custom losses/metrics.
- Enable `detect_anomaly=True` in PyTorch for traceback.
