# Inference at Scale

Running inference on terabyte-scale imagery.

## Model Export

First, export the model to ONNX or TorchScript for speed.

```python
from ununennium.export import to_torchscript

scripted_model = to_torchscript(model, example_input)
scripted_model.save("model.pt")
```

## Parallel Inference

Use `Writer` to stitch tiles asynchronously.

```python
from ununennium.io import GeoTIFFWriter

writer = GeoTIFFWriter("output.tif", profile=profile)

with writer:
    for batch_windows, batch_preds in inference_loader:
        writer.write_batch(batch_windows, batch_preds)
```
