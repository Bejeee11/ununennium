# Benchmarking

Measuring system performance.

## Throughput

Measure images per second.

```math
\text{Throughput} = \frac{\text{Batch Size} \times \text{Batches}}{\text{Time (s)}}
```

## Profiling

Use PyTorch Profiler to identify bottlenecks.

```python
with torch.profiler.profile(...) as prof:
    model(input)
print(prof.key_averages().table())
```

See [Performance Architecture](../architecture/performance.md) for optimization tips.
