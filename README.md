# AgentBible

[![CI](https://github.com/rylanmalarchick/research-code-principles/actions/workflows/ci.yml/badge.svg)](https://github.com/rylanmalarchick/research-code-principles/actions/workflows/ci.yml)
[![Docs](https://github.com/rylanmalarchick/research-code-principles/actions/workflows/docs.yml/badge.svg)](https://rylanmalarchick.github.io/research-code-principles/)
[![PyPI version](https://badge.fury.io/py/agentbible.svg)](https://pypi.org/project/agentbible/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/gh/rylanmalarchick/research-code-principles/branch/main/graph/badge.svg)](https://codecov.io/gh/rylanmalarchick/research-code-principles)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **One-shot correct infrastructure for AI agents working on scientific code.**

## Why AgentBible Exists

When AI generates scientific code, how do you know it's numerically correct?
When you run an experiment, how do you reproduce it 6 months later?

**AgentBible solves both problems:**

- **Core validators** catch numerical bugs at runtime (`@validate_finite`, `@validate_probability`, etc.)
- **Domain plugins** add field-specific validation (quantum: `@validate_unitary`, `@validate_density_matrix`)
- **Automatic provenance** captures everything needed for reproducibility (git SHA, random seeds, package versions, hardware info)

**Result:** Research code you can trust and reproduce.

## Install Now

```bash
pip install agentbible
```

```bash
# With HDF5 provenance support
pip install agentbible[hdf5]

# With semantic context retrieval
pip install agentbible[context]

# Full development install
pip install agentbible[all]
```

## The Problem AgentBible Solves

```python
# WITHOUT AgentBible - silent bug, hours of debugging
def calculate_probability(logits):
    return np.exp(logits)  # Bug: forgot normalization
    # This returns values > 1. You won't catch it until your
    # model produces nonsense results and you spend hours debugging.

probs = calculate_probability(np.array([2.0, 1.0]))  # No error - bug silently propagates
```

```python
# WITH AgentBible - catches it immediately
from agentbible import validate_finite, validate_probabilities

@validate_finite
@validate_probabilities
def calculate_probability(logits):
    return np.exp(logits)  # Same bug

probs = calculate_probability(np.array([2.0, 1.0]))
# ProbabilityBoundsError: Value not in [0, 1]
#   Expected: 0 <= p <= 1 (probability must be in unit interval)
#   Got: max = 7.389...
#   Function: calculate_probability
#
#   Reference: Kolmogorov, 'Foundations of Probability Theory'
#   Guidance: Probability must be in [0, 1]. Common causes:
#       - Forgot normalization (softmax)
#       - Numerical overflow
```

**The bug is caught immediately, with an educational error message.**

## Quick Start

### Create a New Project

```bash
bible init my-simulation --template python-scientific
cd my-simulation
source .venv/bin/activate
pip install -e ".[dev]"
pytest  # Tests pass immediately
```

### Use Core Validators (Any Scientific Code)

```python
from agentbible import (
    validate_finite,
    validate_positive,
    validate_probability,
    validate_normalized,
)
import numpy as np

@validate_finite
@validate_positive
def calculate_energy(mass: float, velocity: np.ndarray) -> float:
    """Kinetic energy must be finite and positive."""
    return 0.5 * mass * np.dot(velocity, velocity)

@validate_finite
@validate_normalized()
def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax must be finite and sum to 1."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

# Validation happens automatically on return
energy = calculate_energy(2.0, np.array([1, 2, 3]))  # OK
probs = softmax(np.array([2.0, 1.0, 0.1]))           # OK - sums to 1
```

### Use Domain Validators (Quantum Computing)

```python
# Quantum validators are in the domains subpackage
from agentbible.domains.quantum import (
    validate_unitary,
    validate_hermitian,
    validate_density_matrix,
)
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
print(metadata["pip_freeze"])   # Full pip freeze for exact reproduction
print(metadata["hardware"])     # CPU model, GPU info, memory
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

## Who This Is For

**AgentBible is for:**

- Researchers using AI agents (Claude, Copilot, Cursor) to write scientific code
- ML/Physics/HPC developers who care about correctness and reproducibility
- Quantum computing developers who need physics validation
- PhD students who want rigorous software from day one
- Anyone who has lost hours debugging a subtle numerical bug

**AgentBible is NOT for:**

- Enterprise web applications
- Frontend/GUI projects
- Code that doesn't involve numerical computation

## Features

### Core Validators (Always Available)

| Decorator | Validates |
|-----------|-----------|
| `@validate_finite` | No NaN or Inf |
| `@validate_positive` | Value > 0 |
| `@validate_non_negative` | Value >= 0 |
| `@validate_range(min, max)` | Value in [min, max] |
| `@validate_probability` | Value in [0, 1] |
| `@validate_probabilities` | Array of probabilities |
| `@validate_normalized` | Sum or norm = 1 |

### Quantum Domain (`agentbible.domains.quantum`)

| Decorator | Validates |
|-----------|-----------|
| `@validate_unitary` | U @ U.H = I |
| `@validate_hermitian` | A = A.H |
| `@validate_density_matrix` | Hermitian, trace=1, positive semi-definite |

All validators:
- Check for NaN/Inf **before** domain-specific checks
- Support both `rtol` and `atol` tolerances
- Provide educational error messages with academic references

### CLI Commands

```bash
bible init <name>           # Create project from template
bible validate <file>       # Validate physics constraints
bible context               # Generate AI context
bible info                  # Show installation info
```

### Provenance Metadata

`save_with_metadata()` automatically captures:

- Git SHA, branch, dirty status, and diff (if uncommitted changes)
- UTC timestamp (ISO 8601)
- Random seeds (numpy, python, torch)
- Hostname, platform, Python version
- Package versions (numpy, scipy, h5py, torch, etc.)
- Full `pip freeze` output for exact environment reproduction
- Hardware info (CPU model, core count, memory, GPU details)

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

1. **Correctness First** - Physical accuracy is non-negotiable
2. **Specification Before Code** - Tests define the contract
3. **Fail Fast with Clarity** - Validate inputs, descriptive errors
4. **Simplicity by Design** - Functions <= 50 lines, single responsibility
5. **Infrastructure Enables Speed** - CI, tests, linting from day one

## Status

**v0.3.0** (Alpha) - Modular architecture with core/domain separation. API stable, ready for real use.

- Core validators work for any scientific code
- Quantum domain validators available via `agentbible.domains.quantum`
- Context module for AI-assisted development
- Philosophy module with importable principles

See the full [ROADMAP.md](ROADMAP.md) for what's coming next.

## Documentation

Full documentation: [rylanmalarchick.github.io/research-code-principles](https://rylanmalarchick.github.io/research-code-principles/)

- [Getting Started](https://rylanmalarchick.github.io/research-code-principles/getting-started/installation/)
- [Validators Guide](https://rylanmalarchick.github.io/research-code-principles/guide/validators/)
- [Provenance Guide](https://rylanmalarchick.github.io/research-code-principles/guide/provenance/)
- [API Reference](https://rylanmalarchick.github.io/research-code-principles/api/validators/)
- [Philosophy](docs/philosophy.md) - Why good code matters
- [Style Guide](docs/style-guide-reference.md) - Coding conventions
- [CHANGELOG](CHANGELOG.md) - Release history

## Getting Help

- **Questions?** [Open a GitHub Issue](https://github.com/rylanmalarchick/research-code-principles/issues)
- **Found a bug?** [Report it here](https://github.com/rylanmalarchick/research-code-principles/issues/new)
- **Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Security issues?** See [SECURITY.md](SECURITY.md)
- **Email:** [rylan1012@gmail.com](mailto:rylan1012@gmail.com)

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

## License

MIT - Use and adapt freely.

## Author

Rylan Malarchick - [rylan1012@gmail.com](mailto:rylan1012@gmail.com)

---

**v0.3.0** - Modular architecture, context module, philosophy module (January 2026)
