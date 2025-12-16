# Contributing to Ununennium

Thank you for your interest in contributing to Ununennium. This document provides comprehensive guidelines and standards for contributing to the project.

---

## Table of Contents

1. [Development Workflow](#development-workflow)
2. [Code Style](#code-style)
3. [Type Annotations](#type-annotations)
4. [Testing](#testing)
5. [Documentation](#documentation)
6. [Commit Messages](#commit-messages)
7. [Pull Request Process](#pull-request-process)

---

## Development Workflow

### Getting Started

1. **Fork the repository** on GitHub

2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/ununennium.git
   cd ununennium
   ```

3. **Set up development environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

### Making Changes

1. Write code following the style guidelines below
2. Add or update tests as appropriate
3. Update documentation if your change affects public APIs
4. Run the test suite and linting checks
5. Commit changes with clear, descriptive messages

### Submitting Changes

1. Push your branch to your fork
2. Open a Pull Request against `main`
3. Fill out the PR template completely
4. Wait for review and address any feedback

---

## Code Style

### Python Style

Ununennium uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check src/

# Auto-fix issues
ruff check src/ --fix

# Format code
ruff format src/
```

### Style Guidelines

| Aspect | Requirement |
|--------|-------------|
| Line length | 100 characters maximum |
| Imports | Sorted, grouped (stdlib, third-party, local) |
| Docstrings | NumPy style for all public APIs |
| Type hints | Required for all public function signatures |
| Naming | snake_case for functions/variables, PascalCase for classes |

### Import Organization

```python
# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import torch
import torch.nn as nn

# Local
from ununennium.core import GeoTensor
from ununennium.models import create_model
```

---

## Type Annotations

All public APIs must include type annotations:

```python
def compute_ndvi(
    nir: torch.Tensor,
    red: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute Normalized Difference Vegetation Index.

    Parameters
    ----------
    nir : torch.Tensor
        Near-infrared reflectance values.
    red : torch.Tensor
        Red reflectance values.
    epsilon : float, optional
        Small value to prevent division by zero. Default: 1e-8.

    Returns
    -------
    torch.Tensor
        NDVI values in range [-1, 1].

    Examples
    --------
    >>> nir = torch.tensor([0.5, 0.6, 0.7])
    >>> red = torch.tensor([0.1, 0.2, 0.3])
    >>> ndvi = compute_ndvi(nir, red)
    >>> ndvi.shape
    torch.Size([3])
    """
    return (nir - red) / (nir + red + epsilon)
```

Run type checking with Pyright:

```bash
pyright src/
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/ununennium --cov-report=html

# Run specific test file
pytest tests/test_core.py -v

# Run tests matching pattern
pytest tests/ -v -k "test_geotensor"
```

### Writing Tests

Tests should be:

| Property | Description |
|----------|-------------|
| Focused | Test one behavior per test function |
| Independent | No dependencies between tests |
| Deterministic | Same result every run (use fixed seeds) |
| Fast | Unit tests should complete in milliseconds |

Example test structure:

```python
import pytest
import torch
from ununennium.core import GeoTensor


class TestGeoTensor:
    """Tests for GeoTensor class."""

    def test_creation_with_valid_crs(self) -> None:
        """GeoTensor should preserve CRS on creation."""
        data = torch.randn(3, 256, 256)
        tensor = GeoTensor(data, crs="EPSG:32632")

        assert tensor.crs == "EPSG:32632"
        assert tensor.shape == (3, 256, 256)

    def test_creation_with_invalid_crs_raises(self) -> None:
        """GeoTensor should raise ValueError for invalid CRS."""
        data = torch.randn(3, 256, 256)

        with pytest.raises(ValueError, match="Invalid CRS"):
            GeoTensor(data, crs="INVALID")
```

### Coverage Requirements

- New features must include tests
- Target: 80% line coverage minimum
- Critical paths: 100% coverage required

---

## Documentation

### Docstring Format

Use NumPy-style docstrings:

```python
def function_name(param1: int, param2: str = "default") -> bool:
    """Short one-line description.

    Longer description if needed, explaining the purpose and behavior
    of the function in more detail.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str, optional
        Description of param2. Default: "default".

    Returns
    -------
    bool
        Description of return value.

    Raises
    ------
    ValueError
        When param1 is negative.

    Examples
    --------
    >>> function_name(42, "custom")
    True

    Notes
    -----
    Any additional notes about implementation or usage.

    References
    ----------
    .. [1] Author, "Title", Journal, Year.
    """
```

### Markdown Documentation

Documentation files in `docs/` should:

- Use consistent heading hierarchy
- Include code examples that actually run
- Reference real code paths (not placeholders)
- Include LaTeX formulas for mathematical concepts
- Add Mermaid diagrams for complex workflows

---

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Commit Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, no logic change) |
| `refactor` | Code refactoring |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `build` | Build system or dependencies |
| `ci` | CI configuration |
| `chore` | Other changes |

### Examples

```
feat(models): add ESRGAN super-resolution architecture

fix(io): handle missing CRS in GeoTIFF files

docs(tutorials): add change detection tutorial
```

---

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code is formatted with `ruff format`
- [ ] No linting errors from `ruff check`
- [ ] Type checking passes with `pyright`
- [ ] Documentation is updated if needed
- [ ] CHANGELOG.md is updated for user-facing changes

### PR Requirements

| Requirement | Description |
|-------------|-------------|
| Title | Clear, following commit message format |
| Description | Complete explanation of changes |
| Issues | Link to related issues |
| Screenshots | Required for UI/visualization changes |
| Benchmarks | Required for optimization PRs |

### Review Process

1. Automated CI checks must pass
2. At least one maintainer approval required
3. All review comments must be resolved
4. Squash and merge preferred for clean history

---

## Issue Reporting

See [SUPPORT.md](SUPPORT.md) for detailed issue reporting guidelines.

---

## Code of Conduct

All contributors must follow the [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful, inclusive, and professional.

---

## Questions

For questions not covered here, open a [GitHub Discussion](https://github.com/olaflaitinen/ununennium/discussions).
