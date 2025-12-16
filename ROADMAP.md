# Roadmap

This document outlines the development roadmap for Ununennium.

---

## Vision

Ununennium aims to be the definitive open-source framework for Earth observation machine learning, providing production-grade tools that bridge the gap between research and deployment.

---

## Current Status

**Version**: 1.0.5 (Stable)

The following core capabilities are available:

| Module | Status |
|--------|--------|
| Core (GeoTensor, GeoBatch) | Stable |
| I/O (COG, STAC, Zarr) | Stable |
| Models (CNN, GAN, PINN) | Stable |
| Training (Trainer, DDP) | Stable |
| Export (ONNX, TorchScript) | Stable |
| Metrics and Evaluation | Stable |

---

## Milestones

### v1.1.0 - Enhanced Foundation Models

**Target**: Q2 2025

| Feature | Description | Status |
|---------|-------------|--------|
| SAM Integration | Segment Anything Model for EO | Planned |
| SSL Pretraining | Self-supervised pretraining recipes | Planned |
| Model Zoo Expansion | 10+ pretrained models | Planned |
| Benchmark Suite | Standardized benchmark datasets | Planned |

### v1.2.0 - Advanced Change Detection

**Target**: Q3 2025

| Feature | Description | Status |
|---------|-------------|--------|
| Temporal Attention | Attention-based change detection | Planned |
| Multi-temporal Fusion | Variable-length sequence handling | Planned |
| Anomaly Detection | Unsupervised change detection | Planned |

### v1.3.0 - Cloud-Native Deployment

**Target**: Q4 2025

| Feature | Description | Status |
|---------|-------------|--------|
| Ray Integration | Distributed inference with Ray | Planned |
| Kubernetes Operator | K8s deployment operator | Planned |
| Streaming Inference | Real-time processing pipeline | Planned |

---

## Long-Term Goals

### Research Integration

- Rapid prototyping of new architectures
- Reproducibility tooling for publications
- Benchmark leaderboards

### Enterprise Features

- Multi-tenancy support
- Access control and audit logging
- SLA monitoring

### Community Growth

- Plugin ecosystem
- Third-party model marketplace
- Educational resources

---

## Non-Goals

The following are explicitly out of scope:

| Non-Goal | Rationale |
|----------|-----------|
| Web-based GUI | Focus on programmatic API |
| Data marketplace | Focus on processing, not data hosting |
| Desktop application | Library-first design |
| Non-Python clients | Python ecosystem focus |

---

## Contributing to the Roadmap

Roadmap priorities are influenced by community feedback. To suggest features:

1. Open a [GitHub Discussion](https://github.com/olaflaitinen/ununennium/discussions) with the "Feature Request" label
2. Provide use case and motivation
3. Participate in prioritization discussions

---

## Changelog

For a history of changes, see [CHANGELOG.md](CHANGELOG.md).
