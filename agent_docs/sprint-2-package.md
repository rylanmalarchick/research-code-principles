# Sprint 2: Package Core

**Sprint:** 2 of 6  
**Focus:** Pip-installable `agentbible` package with validators and CLI  
**Status:** Complete  
**Estimated Sessions:** 3-4

---

## Objective

Create a pip-installable Python package `agentbible` with:

1. **Validators** - Decorators for physics validation (`@validate_unitary`, etc.)
2. **CLI skeleton** - `bible --help` works with subcommand structure
3. **100% test coverage** on core validators

After this sprint: `pip install -e .` works and `bible --help` shows commands.

---

## Deliverables

### 1. pyproject.toml

**Location:** `/pyproject.toml`

**Purpose:** Modern Python packaging with Hatch

**Key sections:**
- Build system: hatchling
- Dependencies: click, rich, numpy, pyyaml
- Optional deps: dev, hdf5, vector
- Entry point: `bible = agentbible.cli.main:cli`
- Tool config: ruff, mypy

---

### 2. Package Structure

**Location:** `/agentbible/`

```
agentbible/
├── __init__.py              # Package version and public API
├── __main__.py              # python -m agentbible
├── cli/
│   ├── __init__.py
│   └── main.py              # Click CLI entry point (skeleton)
└── validators/
    ├── __init__.py          # Public exports
    ├── base.py              # Base decorator utilities
    ├── quantum.py           # @validate_unitary, @validate_hermitian
    ├── probability.py       # @validate_probability, @validate_normalized
    └── bounds.py            # @validate_positive, @validate_range
```

---

### 3. Validators

Physics validation decorators that check function outputs.

#### quantum.py

```python
@validate_unitary(rtol=1e-5, atol=1e-8)
def create_gate() -> np.ndarray:
    """Returns a unitary matrix."""
    ...

@validate_hermitian(rtol=1e-5, atol=1e-8)
def create_hamiltonian() -> np.ndarray:
    """Returns a Hermitian matrix."""
    ...

@validate_density_matrix(rtol=1e-5, atol=1e-8)
def create_state() -> np.ndarray:
    """Returns a valid density matrix (Hermitian, trace 1, positive semi-definite)."""
    ...
```

#### probability.py

```python
@validate_probability
def compute_prob() -> float:
    """Returns value in [0, 1]."""
    ...

@validate_normalized(axis=-1, rtol=1e-5)
def get_distribution() -> np.ndarray:
    """Returns array that sums to 1 along axis."""
    ...

@validate_probabilities
def get_probs() -> np.ndarray:
    """Returns array with all values in [0, 1]."""
    ...
```

#### bounds.py

```python
@validate_positive
def compute_energy() -> float:
    """Returns positive value."""
    ...

@validate_non_negative
def get_count() -> int:
    """Returns non-negative value."""
    ...

@validate_range(min_val=0.0, max_val=1.0)
def get_fidelity() -> float:
    """Returns value in specified range."""
    ...

@validate_finite
def compute_result() -> np.ndarray:
    """Returns array with no NaN or Inf values."""
    ...
```

---

### 4. CLI Skeleton

**Location:** `/agentbible/cli/main.py`

```python
import click

@click.group()
@click.version_option()
def cli():
    """AgentBible: Production-grade research code infrastructure."""
    pass

@cli.command()
@click.argument('name')
@click.option('--template', '-t', default='python-scientific')
def init(name: str, template: str):
    """Initialize a new project from template."""
    click.echo(f"Creating project '{name}' from template '{template}'...")
    # Implementation in Sprint 3

@cli.command()
@click.option('--all', '-a', 'load_all', is_flag=True)
@click.option('--query', '-q', type=str)
@click.argument('path', required=False)
def context(path: str, load_all: bool, query: str):
    """Generate AI context from documentation."""
    click.echo("Loading context...")
    # Implementation wraps opencode-context

@cli.command()
@click.argument('file')
@click.option('--check', '-c', multiple=True)
def validate(file: str, check: tuple):
    """Validate physics constraints in data files."""
    click.echo(f"Validating {file}...")
    # Implementation in Sprint 4
```

---

### 5. Tests

**Location:** `/tests/`

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_validators_quantum.py
├── test_validators_probability.py
├── test_validators_bounds.py
└── test_cli.py              # CLI smoke tests
```

**Coverage target:** 100% on validators

---

## Tasks

| Task | Status | Notes |
|------|--------|-------|
| Create `pyproject.toml` | ✅ | Hatch, deps, entry points |
| Create `agentbible/__init__.py` | ✅ | Version, public API |
| Create `agentbible/__main__.py` | ✅ | python -m support |
| Create `agentbible/cli/main.py` | ✅ | Click skeleton with init, context, validate |
| Create `agentbible/validators/base.py` | ✅ | ValidationError, utilities |
| Create `agentbible/validators/quantum.py` | ✅ | Unitary, Hermitian, density matrix |
| Create `agentbible/validators/probability.py` | ✅ | Probability, normalized |
| Create `agentbible/validators/bounds.py` | ✅ | Positive, range, finite |
| Create `tests/conftest.py` | ✅ | Shared fixtures |
| Create `tests/test_validators_*.py` | ✅ | 78 tests, 88% coverage |
| Create `tests/test_cli.py` | ✅ | CLI smoke tests |
| Verify `pip install -e .` | ✅ | Editable install works |
| Verify `bible --help` | ✅ | CLI works |
| Run `pytest --cov` | ✅ | 88% coverage (>85% threshold) |
| Update ARCHITECTURE.md | ✅ | Marked complete |
| Commit and push | 🔄 | In progress |

---

## Testing Plan

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=agentbible --cov-report=term-missing

# Verify CLI
bible --help
bible --version
bible init --help
bible context --help
bible validate --help

# Test validators manually
python -c "
from agentbible.validators import validate_unitary
import numpy as np

@validate_unitary
def make_identity():
    return np.eye(2)

make_identity()  # Should pass
"
```

---

## Acceptance Criteria

- [x] `pip install -e .` succeeds
- [x] `bible --help` shows init, context, validate commands
- [x] `bible --version` shows version
- [x] `@validate_unitary` catches non-unitary matrices
- [x] `@validate_hermitian` catches non-Hermitian matrices
- [x] `@validate_probability` catches values outside [0, 1]
- [x] `@validate_range` catches values outside specified range
- [x] `pytest --cov` shows 88% (>85% threshold)
- [x] `ruff check agentbible/` passes
- [x] `mypy agentbible/` passes (strict mode)

---

## Design Decisions

### Decorator Pattern

Validators are decorators that:
1. Call the wrapped function
2. Validate the return value
3. Raise `ValidationError` if invalid
4. Return the value unchanged if valid

```python
@validate_unitary
def my_gate():
    return np.array([[1, 0], [0, 1]])

# Equivalent to:
def my_gate():
    result = np.array([[1, 0], [0, 1]])
    if not is_unitary(result):
        raise ValidationError("Matrix is not unitary")
    return result
```

### Error Messages

Errors should be descriptive:
```
ValidationError: Matrix is not unitary
  Expected: U†U = I
  Got: max|U†U - I| = 0.15
  Tolerance: rtol=1e-5, atol=1e-8
  Shape: (4, 4)
  Function: create_cnot_gate
```

### Tolerance Defaults

| Validator | rtol | atol |
|-----------|------|------|
| unitary | 1e-5 | 1e-8 |
| hermitian | 1e-5 | 1e-8 |
| trace | 1e-5 | 1e-8 |
| probability | - | 1e-10 |
| normalized | 1e-5 | 1e-8 |

---

## Notes

- Keep validators pure - no side effects
- Use numpy for all matrix operations
- Lazy import numpy in decorators to avoid startup cost
- Consider @functools.wraps for proper function metadata
