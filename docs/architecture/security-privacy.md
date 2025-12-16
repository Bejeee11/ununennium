# Security and Privacy

## Data Security

- **Encryption**: Support for reading analytical data from encrypted S3 buckets.
- **Sanitization**: Metadata scrubbing for exported assets.

## Geospatial Privacy

High-resolution imagery can reveal sensitive information.

### Geometric Obfuscation
Random jittering of coordinates for anonymized public releases.

```math
x' = x + \epsilon_x, \quad \epsilon_x \sim \mathcal{N}(0, \sigma^2)
```

### Resolution Downgrading
Automatic resampling to lower resolution for privacy-sensitive areas.

## Model Security

### Adversarial Defense
Models can be robustified against adversarial attacks using adversarial training.

```math
\min_\theta \mathbb{E}_{(x,y)} \left[ \max_{\delta \in S} \mathcal{L}(f_\theta(x+\delta), y) \right]
```

where $S$ is the allowed perturbation set (e.g., $\ell_\infty$ ball).
