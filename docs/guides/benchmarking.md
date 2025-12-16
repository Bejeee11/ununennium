# Benchmarking

This guide covers performance benchmarking for Ununennium models.

---

## Throughput Benchmarking

### Measure Training Throughput

```python
import time
import torch

def benchmark_training(model, loader, n_batches=100):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Warmup
    for batch in loader:
        break
    _ = model(batch["image"].cuda())
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        
        optimizer.zero_grad()
        output = model(batch["image"].cuda())
        loss = output.mean()
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    throughput = n_batches * loader.batch_size / elapsed
    print(f"Throughput: {throughput:.1f} img/s")
    return throughput
```

### Measure Inference Throughput

```python
def benchmark_inference(model, input_shape, n_iterations=100):
    model.eval()
    x = torch.randn(input_shape).cuda()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    throughput = n_iterations * input_shape[0] / elapsed
    latency = elapsed / n_iterations * 1000
    print(f"Throughput: {throughput:.1f} img/s, Latency: {latency:.2f} ms")
```

---

## Memory Profiling

```python
def measure_memory(model, input_shape):
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(input_shape).cuda()
    
    # Forward
    output = model(x)
    fwd_mem = torch.cuda.max_memory_allocated() / 1e9
    
    # Backward
    output.mean().backward()
    total_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"Forward: {fwd_mem:.2f} GB, Total: {total_mem:.2f} GB")
```

---

## Benchmark Suite

```python
from ununennium.benchmarks import ModelBenchmark

benchmark = ModelBenchmark(
    model=model,
    input_shapes=[(1, 12, 256, 256), (1, 12, 512, 512)],
    batch_sizes=[1, 4, 8, 16],
)

results = benchmark.run()
results.to_csv("benchmark_results.csv")
```

---

## See Also

- [Performance Guide](../architecture/performance.md)
