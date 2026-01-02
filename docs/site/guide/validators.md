# Physics Validators

AgentBible provides decorators to validate physics and mathematical properties of function return values.

## Quantum Validators

### `@validate_unitary`

Validates that the returned matrix is unitary: U @ U.H = I

```python
from agentbible import validate_unitary
import numpy as np

@validate_unitary
def create_hadamard():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

@validate_unitary(rtol=1e-6)  # Custom tolerance
def create_rotation(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=complex)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rtol` | float | 1e-10 | Relative tolerance |
| `atol` | float | 1e-12 | Absolute tolerance |

### `@validate_hermitian`

Validates that the returned matrix equals its conjugate transpose: A = A.H

```python
from agentbible import validate_hermitian
import numpy as np

@validate_hermitian
def create_pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)
```

### `@validate_density_matrix`

Validates all properties of a valid density matrix:

- Hermitian (A = A.H)
- Trace equals 1
- Positive semi-definite (all eigenvalues ≥ 0)

```python
from agentbible import validate_density_matrix
import numpy as np

@validate_density_matrix
def create_mixed_state():
    """Maximally mixed qubit state."""
    return np.eye(2, dtype=complex) / 2

@validate_density_matrix
def create_pure_state(psi):
    """Pure state from state vector."""
    psi = np.array(psi, dtype=complex)
    return np.outer(psi, psi.conj())
```

## Probability Validators

### `@validate_probability`

Validates that the returned value is in [0, 1].

```python
from agentbible import validate_probability

@validate_probability
def compute_fidelity(rho, sigma):
    # ... computation ...
    return fidelity  # Must be in [0, 1]
```

### `@validate_probabilities`

Validates that all elements of the returned array are in [0, 1].

```python
from agentbible import validate_probabilities
import numpy as np

@validate_probabilities
def compute_eigenvalues(rho):
    """Eigenvalues of density matrix are probabilities."""
    return np.linalg.eigvalsh(rho)
```

### `@validate_normalized`

Validates that the array sums to 1 (for probability distributions) or has unit norm (for state vectors).

```python
from agentbible import validate_normalized
import numpy as np

@validate_normalized(norm_type="sum")  # Default
def probability_distribution():
    return np.array([0.25, 0.25, 0.5])

@validate_normalized(norm_type="l2")
def state_vector():
    return np.array([1, 0], dtype=complex)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `norm_type` | str | "sum" | "sum" for probability, "l2" for state vectors |
| `rtol` | float | 1e-10 | Relative tolerance |
| `atol` | float | 1e-12 | Absolute tolerance |

## Bounds Validators

### `@validate_positive`

Validates that the returned value is strictly positive (> 0).

```python
from agentbible import validate_positive

@validate_positive
def compute_energy():
    # ... computation ...
    return energy  # Must be > 0
```

### `@validate_non_negative`

Validates that the returned value is non-negative (≥ 0).

```python
from agentbible import validate_non_negative

@validate_non_negative
def compute_entropy():
    # ... computation ...
    return entropy  # Must be >= 0
```

### `@validate_range`

Validates that the returned value is within a specified range.

```python
from agentbible import validate_range

@validate_range(min_val=0, max_val=2*np.pi)
def compute_phase():
    # ... computation ...
    return phase

@validate_range(min_val=-1, max_val=1, inclusive=True)  # Default
def compute_correlation():
    return correlation
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_val` | float | None | Minimum allowed value |
| `max_val` | float | None | Maximum allowed value |
| `inclusive` | bool | True | Include boundary values |

### `@validate_finite`

Validates that the returned value contains no NaN or Inf.

```python
from agentbible import validate_finite
import numpy as np

@validate_finite
def compute_matrix():
    result = some_computation()
    return result  # Raises if contains NaN or Inf
```

## Combining Validators

You can stack multiple validators:

```python
from agentbible import validate_hermitian, validate_finite
import numpy as np

@validate_finite
@validate_hermitian
def compute_hamiltonian():
    """Returns a finite Hermitian matrix."""
    H = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
    return H
```

Validators are applied from bottom to top (inner to outer).

## Error Messages

When validation fails, you get descriptive error messages:

```python
@validate_unitary
def broken_gate():
    return np.array([[1, 0], [0, 2]], dtype=complex)  # Not unitary!

broken_gate()
# Raises: UnitaryValidationError: Matrix is not unitary.
#         Maximum deviation from identity: 3.0
#         Tolerance: rtol=1e-10, atol=1e-12
```

## Performance Considerations

Validation adds computational overhead. For performance-critical code:

1. Use validators during development and testing
2. Consider removing them in production via environment variable:

```python
import os
os.environ["AGENTBIBLE_SKIP_VALIDATION"] = "1"
```

Or disable per-decorator:

```python
@validate_unitary(enabled=False)  # Skip validation
def production_gate():
    return fast_computation()
```
