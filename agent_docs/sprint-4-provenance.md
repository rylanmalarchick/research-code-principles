# Sprint 4: Provenance & Testing

**Sprint:** 4 of 6  
**Focus:** Research-grade data tracking and physics testing utilities  
**Status:** Complete  
**Estimated Sessions:** 2-3

---

## Objective

Implement research-critical features:

1. **HDF5 Provenance** - Automatically embed reproducibility metadata in saved files
2. **Physics Testing** - Decorators and fixtures for physics-aware tests
3. **CLI Validate** - Make `bible validate` functional for file validation

After this sprint: Researchers can save data with full provenance tracking and write physics-aware tests.

---

## Deliverables

### 1. Provenance Module

**Location:** `agentbible/provenance/`

```python
from agentbible.provenance import save_with_metadata, load_with_metadata

# Save numpy array with full provenance
save_with_metadata(
    "results.h5",
    data={"density_matrix": rho, "eigenvalues": eigs},
    description="Ground state density matrix",
)
# Automatically embeds:
# - git SHA, branch, dirty status
# - timestamp (UTC ISO 8601)
# - random seeds (numpy, python, torch if available)
# - hardware info (CPU, GPU if available)
# - package versions (numpy, scipy, etc.)

# Load with metadata
data, metadata = load_with_metadata("results.h5")
print(metadata["git_sha"])  # "a1b2c3d"
print(metadata["numpy_seed"])  # 42
```

**Metadata Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `git_sha` | str | Full 40-char commit hash |
| `git_branch` | str | Current branch name |
| `git_dirty` | bool | True if uncommitted changes |
| `timestamp` | str | ISO 8601 UTC timestamp |
| `numpy_seed` | int/None | numpy.random seed if set |
| `python_seed` | int/None | random.seed if set |
| `torch_seed` | int/None | torch seed if available |
| `hostname` | str | Machine hostname |
| `platform` | str | OS and version |
| `python_version` | str | Python version |
| `packages` | dict | Key package versions |
| `description` | str | User-provided description |

---

### 2. Testing Module

**Location:** `agentbible/testing/`

#### Fixtures (`fixtures.py`)

```python
import pytest
from agentbible.testing import deterministic_seed, tolerance

@pytest.fixture
def deterministic_seed():
    """Set all random seeds for reproducibility."""
    # Sets numpy, random, and optionally torch seeds
    yield seed_value

@pytest.fixture
def tolerance():
    """Standard tolerance for floating-point comparisons."""
    return {"rtol": 1e-10, "atol": 1e-12}

@pytest.fixture
def quantum_tolerance():
    """Relaxed tolerance for quantum calculations."""
    return {"rtol": 1e-6, "atol": 1e-8}
```

#### Decorators (`decorators.py`)

```python
from agentbible.testing import physics_test

@physics_test(checks=["unitarity", "hermiticity"])
def test_quantum_gate(gate):
    """Test automatically validates gate properties."""
    return gate  # Decorator validates returned value

@physics_test(checks=["trace_one", "positive_semidefinite"])
def test_density_matrix(rho):
    """Test validates density matrix properties."""
    return rho
```

**Available Checks:**

| Check | Validates |
|-------|-----------|
| `unitarity` | U @ U.H = I |
| `hermiticity` | A = A.H |
| `trace_one` | tr(A) = 1 |
| `positive_semidefinite` | All eigenvalues >= 0 |
| `normalization` | ||v|| = 1 |
| `probability` | All values in [0, 1] |

---

### 3. CLI Validate Command

**Location:** `agentbible/cli/validate.py`

Make the existing `bible validate` command functional:

```bash
# Validate a numpy file
bible validate state.npy --check unitarity

# Validate HDF5 dataset
bible validate results.h5 --check all

# Multiple checks
bible validate matrix.npy -c unitarity -c hermiticity

# Custom tolerance
bible validate state.npy --check normalization --rtol 1e-6
```

---

## Tasks

| Task | Status | Notes |
|------|--------|-------|
| Create `agent_docs/sprint-4-provenance.md` | ✅ | This file |
| Create `agentbible/provenance/__init__.py` | ✅ | Module init |
| Create `agentbible/provenance/hdf5.py` | ✅ | 416 lines, v2.0 with pip_freeze, hardware, git_diff |
| Create `agentbible/testing/__init__.py` | ✅ | Module init |
| Create `agentbible/testing/fixtures.py` | ✅ | 158 lines |
| Create `agentbible/testing/decorators.py` | ✅ | 171 lines, @physics_test |
| Create `agentbible/cli/validate.py` | ✅ | 220 lines |
| Update `agentbible/cli/main.py` | ✅ | Wire validate |
| Create `tests/test_provenance.py` | ✅ | Provenance tests |
| Create `tests/test_testing.py` | ✅ | Testing module tests |
| Update `pyproject.toml` | ✅ | h5py in [hdf5] extra |
| Commit and push | ⬜ | End of sprint |

---

## Testing Plan

```bash
# Test provenance
python -c "
from agentbible.provenance import save_with_metadata, load_with_metadata
import numpy as np
save_with_metadata('test.h5', {'arr': np.eye(2)}, description='test')
data, meta = load_with_metadata('test.h5')
print(meta['git_sha'])
"

# Test fixtures
pytest tests/test_testing.py -v

# Test CLI validate
bible validate tests/fixtures/unitary.npy --check unitarity
bible validate tests/fixtures/density.npy --check all
```

---

## Acceptance Criteria

- [x] `save_with_metadata()` creates HDF5 with git SHA, seeds, timestamp
- [x] `load_with_metadata()` returns data and metadata dict
- [x] `@physics_test(checks=[...])` validates returned values
- [x] `deterministic_seed` fixture sets numpy/random seeds
- [x] `bible validate file.npy --check unitarity` works
- [x] Tests pass with 90%+ coverage on new modules
- [x] Works without h5py installed (graceful degradation)

---

## Design Decisions

### HDF5 vs Other Formats

- HDF5 chosen for:
  - Native attribute storage for metadata
  - Efficient for large arrays
  - Standard in scientific computing
  - Self-describing format

### Optional Dependencies

- `h5py` is optional (in `[hdf5]` extra)
- Provenance module raises helpful error if h5py missing
- Testing module works without any optional deps

### Seed Capture

- Seeds are captured, not set
- If user hasn't set a seed, we record None
- We don't force reproducibility; we enable tracking

---

## Notes

- Consider adding `--output json` to validate command for scripting
- Future: support for zarr, netCDF formats
- Future: integration with MLflow/Weights&Biases for experiment tracking
