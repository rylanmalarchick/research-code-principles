# Quick Start

This guide gets you from zero to a validated quantum simulation in 5 minutes.

## 1. Install AgentBible

```bash
pip install agentbible[hdf5]
```

## 2. Create a New Project

```bash
bible init my-quantum-sim --template python-scientific
cd my-quantum-sim
```

This creates a project with:

- Pre-configured `pyproject.toml` (ruff, mypy, pytest)
- Physics validation helpers
- 28 example tests that pass immediately
- `.cursorrules` for AI agents

## 3. Set Up the Environment

```bash
source .venv/bin/activate  # Created by bible init
pip install -e ".[dev]"
```

## 4. Run the Tests

```bash
pytest
```

All 28 tests should pass.

## 5. Use Validators in Your Code

Create a file `src/gates.py`:

```python
from agentbible import validate_unitary, validate_hermitian
import numpy as np

@validate_unitary
def hadamard():
    """Create Hadamard gate - validated as unitary."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

@validate_hermitian
def pauli_x():
    """Create Pauli-X gate - validated as Hermitian."""
    return np.array([[0, 1], [1, 0]], dtype=complex)

@validate_unitary
def rotation_z(theta: float):
    """Create Z-rotation gate."""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)
```

## 6. Test Your Gates

Create `tests/test_gates.py`:

```python
import numpy as np
from src.gates import hadamard, pauli_x, rotation_z

def test_hadamard_properties():
    H = hadamard()
    # H^2 = I
    assert np.allclose(H @ H, np.eye(2))

def test_pauli_x_is_involution():
    X = pauli_x()
    # X^2 = I
    assert np.allclose(X @ X, np.eye(2))

def test_rotation_z_identity():
    # R_z(0) = I
    R = rotation_z(0)
    assert np.allclose(R, np.eye(2))
```

Run tests:

```bash
pytest tests/test_gates.py -v
```

## 7. Save Results with Provenance

```python
from agentbible.provenance import save_with_metadata
from src.gates import hadamard, pauli_x
import numpy as np

# Run simulation
H = hadamard()
X = pauli_x()
result = H @ X @ H  # HXH = Z

# Save with full metadata
save_with_metadata(
    "results/hxh_simulation.h5",
    {"result": result, "H": H, "X": X},
    description="HXH = Z identity verification",
)
```

The saved file contains:

- Your data
- Git SHA and branch
- Timestamp
- Random seeds
- Python and package versions
- Hardware info

## 8. Validate Data Files

```bash
# Validate unitarity
bible validate results/hxh_simulation.h5 --check unitarity

# Run all checks
bible validate results/hxh_simulation.h5 --check all
```

## Next Steps

- [Validators Guide](../guide/validators.md) - All available validators
- [Provenance Guide](../guide/provenance.md) - Full metadata tracking
- [Testing Guide](../guide/testing.md) - Physics-aware testing
- [CLI Reference](../guide/cli.md) - All CLI commands
