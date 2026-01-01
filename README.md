# AgentBible

[![CI](https://github.com/rylanmalarchick/research-code-principles/actions/workflows/ci.yml/badge.svg)](https://github.com/rylanmalarchick/research-code-principles/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade infrastructure for AI-assisted research software.**

AgentBible provides physics validation decorators, project scaffolding, provenance tracking, and testing utilities for scientific Python projects.

## Installation

```bash
pip install agentbible

# With HDF5 provenance support
pip install agentbible[hdf5]

# Full development install
pip install agentbible[all]
```

## Quick Start

### Create a New Project

```bash
bible init my-quantum-sim --template python-scientific
cd my-quantum-sim
source .venv/bin/activate
pip install -e ".[dev]"
pytest  # 28 tests pass immediately
```

### Use Physics Validators

```python
from agentbible import validate_unitary, validate_density_matrix
import numpy as np

@validate_unitary
def create_hadamard():
    """Returns Hadamard gate - validated as unitary."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

@validate_density_matrix
def create_mixed_state():
    """Returns maximally mixed state - validated as density matrix."""
    return np.eye(2, dtype=complex) / 2

# Validation happens automatically on return
H = create_hadamard()      # OK - unitary
rho = create_mixed_state() # OK - trace=1, Hermitian, positive semi-definite
```

### Validate Data Files

```bash
# Validate a numpy matrix
bible validate state.npy --check unitarity

# Validate HDF5 with all checks
bible validate results.h5 --check all

# Multiple specific checks
bible validate matrix.npy -c hermiticity -c trace -c positivity
```

### Save Data with Provenance

```python
from agentbible.provenance import save_with_metadata, load_with_metadata
import numpy as np

# Save with full reproducibility metadata
save_with_metadata(
    "results.h5",
    {"density_matrix": rho, "eigenvalues": np.linalg.eigvalsh(rho)},
    description="Ground state calculation",
)

# Load with metadata
data, metadata = load_with_metadata("results.h5")
print(metadata["git_sha"])      # "a1b2c3d..."
print(metadata["timestamp"])    # "2026-01-01T12:00:00+00:00"
print(metadata["packages"])     # {"numpy": "1.26.0", ...}
```

### Physics-Aware Testing

```python
from agentbible.testing import physics_test, deterministic_seed
import numpy as np

@physics_test(checks=["unitarity", "hermiticity"])
def test_pauli_x():
    """Automatically validates return value."""
    return np.array([[0, 1], [1, 0]], dtype=complex)

def test_reproducible(deterministic_seed):
    """Seeds are set to 42 for reproducibility."""
    random_value = np.random.rand()
    assert random_value == 0.3745401188473625  # Always the same
```

## Features

### Validators

| Decorator | Validates |
|-----------|-----------|
| `@validate_unitary` | U @ U.H = I |
| `@validate_hermitian` | A = A.H |
| `@validate_density_matrix` | Hermitian, trace=1, positive semi-definite |
| `@validate_probability` | Value in [0, 1] |
| `@validate_probabilities` | Array of probabilities |
| `@validate_normalized` | Sum or norm = 1 |
| `@validate_positive` | Value > 0 |
| `@validate_non_negative` | Value >= 0 |
| `@validate_range(min, max)` | Value in [min, max] |
| `@validate_finite` | No NaN or Inf |

### CLI Commands

```bash
bible init <name>           # Create project from template
bible validate <file>       # Validate physics constraints
bible context               # Generate AI context
bible info                  # Show installation info
```

### Provenance Metadata

`save_with_metadata()` automatically captures:
- Git SHA, branch, dirty status
- UTC timestamp
- Random seeds (numpy, python, torch)
- Hostname, platform, Python version
- Package versions (numpy, scipy, h5py, torch, etc.)

### Testing Fixtures

| Fixture | Purpose |
|---------|---------|
| `deterministic_seed` | Sets numpy/random to seed 42 |
| `tolerance` | Returns `{"rtol": 1e-10, "atol": 1e-12}` |
| `quantum_tolerance` | Returns `{"rtol": 1e-6, "atol": 1e-8}` |

## Project Templates

### Python Scientific (`python-scientific`)

Pre-configured with:
- **ruff** for linting (strict rules)
- **mypy** in strict mode
- **pytest** with 70% coverage minimum
- Physics validation helpers
- `.cursorrules` for AI agents

```bash
bible init my-project --template python-scientific
```

### C++ HPC/CUDA (`cpp-hpc-cuda`)

Pre-configured with:
- **CMake** with zero-warning policy
- **GoogleTest** for testing
- **CUDA** support (optional)
- Physical validation functions

```bash
bible init my-project --template cpp-hpc-cuda
```

## The 5 Principles

1. **Correctness First** — Physical accuracy is non-negotiable
2. **Specification Before Code** — Tests define the contract
3. **Fail Fast with Clarity** — Validate inputs, descriptive errors
4. **Simplicity by Design** — Functions ≤50 lines, single responsibility
5. **Infrastructure Enables Speed** — CI, tests, linting from day one

## Development

```bash
# Clone and setup
git clone https://github.com/rylanmalarchick/research-code-principles
cd research-code-principles
./bootstrap.sh

# Or manually
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,hdf5]"

# Run tests
pytest tests/ -v

# Lint and type check
ruff check agentbible/
mypy agentbible/
```

## Documentation

- [Philosophy](docs/philosophy.md) — Why good code matters
- [Style Guide](docs/style-guide-reference.md) — Coding conventions
- [Agent Prompts](agent_prompts/) — Modular AI context snippets
- [CHANGELOG](CHANGELOG.md) — Release history
- [SECURITY](SECURITY.md) — Security policy

## License

MIT — Use and adapt freely.

## Author

Rylan Malarchick — [rylan1012@gmail.com](mailto:rylan1012@gmail.com)

---

**v0.1.0** — Initial release (January 2026)
