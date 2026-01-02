# API Reference: Testing

::: agentbible.testing

## Decorators

### physics_test

```python
from agentbible.testing import physics_test

@physics_test(
    checks: List[str],
    rtol: float = 1e-10,
    atol: float = 1e-12,
)
def test_function() -> np.ndarray:
    ...
```

Decorator that validates the return value of a test function.

**Parameters:**

- `checks` (List[str]): List of checks to perform
- `rtol` (float): Relative tolerance. Default: 1e-10
- `atol` (float): Absolute tolerance. Default: 1e-12

**Available Checks:**

| Check | Validates |
|-------|-----------|
| `"unitarity"` | U @ U.H = I |
| `"hermiticity"` | A = A.H |
| `"trace"` | Tr(A) = 1 |
| `"positivity"` | All eigenvalues ≥ 0 |
| `"normalization"` | Sum = 1 or norm = 1 |

**Example:**

```python
@physics_test(checks=["unitarity", "hermiticity"])
def test_pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

@physics_test(checks=["trace", "positivity"], rtol=1e-6)
def test_density_matrix():
    return np.eye(2, dtype=complex) / 2
```

---

## Pytest Fixtures

Import fixtures in your `conftest.py`:

```python
from agentbible.testing import (
    deterministic_seed,
    tolerance,
    quantum_tolerance,
)
```

### deterministic_seed

```python
def test_reproducible(deterministic_seed):
    value = np.random.rand()
    assert value == 0.3745401188473625
```

Sets numpy and Python random seeds to 42 for the duration of the test.

**Scope:** function (resets for each test)

---

### tolerance

```python
def test_with_tolerance(tolerance):
    A = np.random.rand(5, 5)
    A_inv = np.linalg.inv(A)
    assert np.allclose(A @ A_inv, np.eye(5), **tolerance)
```

Returns a dictionary with standard numerical tolerances:

```python
{"rtol": 1e-10, "atol": 1e-12}
```

---

### quantum_tolerance

```python
def test_quantum_gate(quantum_tolerance):
    U = create_complex_gate()
    assert np.allclose(U @ U.conj().T, np.eye(4), **quantum_tolerance)
```

Returns looser tolerances appropriate for quantum computing:

```python
{"rtol": 1e-6, "atol": 1e-8}
```

---

## Fixture Usage

### In conftest.py

```python
# tests/conftest.py
import pytest
from agentbible.testing import deterministic_seed, tolerance, quantum_tolerance

# Fixtures are automatically available to all tests
```

### In Test Files

```python
# tests/test_gates.py
import numpy as np

def test_random_unitary(deterministic_seed, quantum_tolerance):
    """Test with fixed seed and quantum tolerances."""
    # Random values are reproducible
    angles = np.random.rand(3)
    
    U = create_rotation(*angles)
    
    # Use quantum tolerance for comparison
    assert np.allclose(
        U @ U.conj().T,
        np.eye(U.shape[0]),
        **quantum_tolerance
    )
```

---

## Custom Fixtures

You can create your own fixtures based on AgentBible's:

```python
# conftest.py
import pytest
import numpy as np
from agentbible.testing import deterministic_seed

@pytest.fixture
def pauli_matrices():
    """Standard Pauli matrices."""
    return {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

@pytest.fixture
def random_state(deterministic_seed):
    """Random normalized state vector."""
    psi = np.random.rand(4) + 1j * np.random.rand(4)
    return psi / np.linalg.norm(psi)

@pytest.fixture
def random_density_matrix(deterministic_seed):
    """Random valid density matrix."""
    psi = np.random.rand(4) + 1j * np.random.rand(4)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())
```

---

## Integration with pytest

### pytest.ini / pyproject.toml

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-v", "--tb=short"]
markers = [
    "slow: marks tests as slow",
    "physics: marks physics validation tests",
]
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Only physics tests
pytest -m physics

# Exclude slow tests
pytest -m "not slow"
```
