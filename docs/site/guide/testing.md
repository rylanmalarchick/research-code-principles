# Testing Utilities

AgentBible provides physics-aware testing tools for scientific Python code.

## Physics Test Decorator

The `@physics_test` decorator combines test execution with physics validation.

### Basic Usage

```python
from agentbible.testing import physics_test
import numpy as np

@physics_test(checks=["unitarity"])
def test_hadamard_gate():
    """Returns Hadamard - automatically validated as unitary."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

@physics_test(checks=["hermiticity", "trace"])
def test_density_matrix():
    """Returns density matrix - validated for multiple properties."""
    return np.eye(2, dtype=complex) / 2
```

### Available Checks

| Check | Validates |
|-------|-----------|
| `unitarity` | U @ U.H = I |
| `hermiticity` | A = A.H |
| `trace` | Tr(A) = 1 |
| `positivity` | All eigenvalues ≥ 0 |
| `normalization` | Sum = 1 or norm = 1 |

### Multiple Checks

```python
@physics_test(checks=["hermiticity", "trace", "positivity"])
def test_valid_density_matrix():
    rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
    return rho
```

### Custom Tolerances

```python
@physics_test(checks=["unitarity"], rtol=1e-6, atol=1e-8)
def test_approximate_unitary():
    # Allow looser tolerance for numerical methods
    return computed_unitary()
```

## Pytest Fixtures

AgentBible provides fixtures for reproducible testing. Add them to your `conftest.py`:

```python
# conftest.py
from agentbible.testing import (
    deterministic_seed,
    tolerance,
    quantum_tolerance,
)
```

### `deterministic_seed`

Sets numpy and Python random seeds to 42 for reproducible tests.

```python
def test_random_matrix(deterministic_seed):
    """Random values are reproducible."""
    matrix = np.random.rand(10, 10)
    
    # This assertion always passes because seed is fixed
    assert matrix[0, 0] == 0.3745401188473625
```

### `tolerance`

Returns standard numerical tolerances.

```python
def test_matrix_inverse(tolerance):
    """Use standard tolerances for comparisons."""
    A = np.random.rand(5, 5)
    A_inv = np.linalg.inv(A)
    
    assert np.allclose(
        A @ A_inv,
        np.eye(5),
        rtol=tolerance["rtol"],
        atol=tolerance["atol"]
    )
```

Default values:
- `rtol`: 1e-10
- `atol`: 1e-12

### `quantum_tolerance`

Returns tolerances appropriate for quantum computing (slightly looser).

```python
def test_quantum_gate(quantum_tolerance):
    """Use quantum-appropriate tolerances."""
    U = create_complex_gate()
    
    assert np.allclose(
        U @ U.conj().T,
        np.eye(U.shape[0]),
        **quantum_tolerance
    )
```

Default values:
- `rtol`: 1e-6
- `atol`: 1e-8

## Test Organization

### Recommended Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_gates.py         # Quantum gate tests
├── test_states.py        # State preparation tests
├── test_measurements.py  # Measurement tests
└── test_integration.py   # End-to-end tests
```

### Example conftest.py

```python
"""Shared test fixtures and configuration."""
import pytest
import numpy as np

# Import AgentBible fixtures
from agentbible.testing import (
    deterministic_seed,
    tolerance,
    quantum_tolerance,
)

@pytest.fixture
def identity_2x2():
    """2x2 identity matrix."""
    return np.eye(2, dtype=complex)

@pytest.fixture
def pauli_matrices():
    """Pauli matrices X, Y, Z."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"X": X, "Y": Y, "Z": Z}

@pytest.fixture
def random_state(deterministic_seed):
    """Random normalized state vector."""
    psi = np.random.rand(4) + 1j * np.random.rand(4)
    return psi / np.linalg.norm(psi)
```

## Testing Best Practices

### 1. Test Physical Properties

```python
def test_pauli_anticommutation(pauli_matrices):
    """Pauli matrices anticommute: {X, Y} = 0."""
    X, Y = pauli_matrices["X"], pauli_matrices["Y"]
    assert np.allclose(X @ Y + Y @ X, np.zeros((2, 2)))
```

### 2. Test Edge Cases

```python
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_identity_is_unitary(n):
    """Identity matrix is unitary for any dimension."""
    I = np.eye(n, dtype=complex)
    assert np.allclose(I @ I.conj().T, np.eye(n))
```

### 3. Test Error Conditions

```python
def test_invalid_state_raises():
    """Non-normalized state should raise error."""
    with pytest.raises(ValueError, match="not normalized"):
        validate_state(np.array([1, 1]))  # Not normalized
```

### 4. Use Hypothesis for Property Testing

```python
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

@given(arrays(complex, (2, 2)))
def test_hermitian_conjugate_involutory(A):
    """(A†)† = A for any matrix."""
    assert np.allclose(A.conj().T.conj().T, A)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_gates.py -v

# Run tests matching pattern
pytest tests/ -k "unitary" -v

# Run excluding slow tests
pytest tests/ -m "not slow" -v
```

## Integration with CI

Example pytest configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
]
```
