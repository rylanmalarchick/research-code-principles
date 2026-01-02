# Style Guide

Coding conventions for research software that prioritizes correctness and maintainability.

## Code Structure

### Function Length

**Maximum 50 lines per function.**

If a function exceeds 50 lines, break it into smaller, focused functions.

```python
# Too long - break it up
def process_experiment():
    # 100+ lines of code
    ...

# Better - decomposed
def load_data():
    ...  # 20 lines

def validate_data(data):
    ...  # 15 lines

def analyze(data):
    ...  # 25 lines

def process_experiment():
    data = load_data()
    validate_data(data)
    return analyze(data)
```

### Single Responsibility

Each function should do one thing well.

```python
# Bad: Multiple responsibilities
def process_and_save_results(data, path):
    results = complex_analysis(data)
    validated = check_constraints(results)
    save_to_disk(validated, path)
    send_notification()
    return validated

# Good: Separated concerns
def analyze(data):
    return complex_analysis(data)

def validate(results):
    return check_constraints(results)

def save(results, path):
    save_to_disk(results, path)
```

## Naming Conventions

### Variables

Use descriptive names that convey meaning:

```python
# Bad
x = compute(m, n)
res = f(a, b)

# Good
energy = compute_ground_state_energy(hamiltonian, num_qubits)
fidelity = calculate_fidelity(state_a, state_b)
```

### Physics Variables

Use conventional physics notation where appropriate:

```python
# Acceptable physics notation
H = create_hamiltonian()      # Hamiltonian
U = create_unitary()          # Unitary operator
rho = create_density_matrix() # Density matrix
psi = create_state_vector()   # State vector
E = compute_energy()          # Energy

# With descriptive suffixes for clarity
H_system = create_system_hamiltonian()
U_gate = create_gate_unitary()
rho_initial = create_initial_state()
```

### Functions

Use verb phrases for functions:

```python
# Good function names
def compute_expectation_value(operator, state): ...
def validate_unitarity(matrix): ...
def create_bell_state(): ...
def save_results(data, path): ...

# Avoid noun-only names
def hamiltonian(): ...  # Bad - use create_hamiltonian()
def result(): ...       # Bad - use compute_result()
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def compute_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute fidelity between two density matrices.
    
    The fidelity is defined as F(ρ, σ) = (Tr√(√ρ σ √ρ))².
    
    Args:
        rho: First density matrix (n x n).
        sigma: Second density matrix (n x n).
        
    Returns:
        Fidelity value in [0, 1].
        
    Raises:
        ValueError: If matrices have different dimensions.
        
    Example:
        >>> rho = np.eye(2) / 2
        >>> fidelity = compute_fidelity(rho, rho)
        >>> assert fidelity == 1.0
        
    References:
        Nielsen & Chuang, Eq. 9.53
    """
```

### Comments

Comment the "why", not the "what":

```python
# Bad: Describes what the code does (obvious)
# Multiply matrix by its conjugate transpose
result = A @ A.conj().T

# Good: Explains why
# Compute A†A to check positive semi-definiteness
result = A @ A.conj().T
```

### Physics References

Cite equations and sources:

```python
def berry_phase(path: List[np.ndarray]) -> float:
    """Compute geometric Berry phase.
    
    Implements Eq. 12 from Berry (1984), Proc. R. Soc. Lond. A 392, 45.
    
    The phase is γ = i∮⟨n|∇n⟩·dR
    """
```

## Type Hints

### Always Use Type Hints

```python
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

def create_rotation_gate(
    axis: str,
    angle: float,
) -> NDArray[np.complex128]:
    """Create single-qubit rotation gate."""
    ...

def diagonalize(
    matrix: NDArray[np.complex128],
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return (eigenvalues, eigenvectors)."""
    ...
```

### Optional Parameters

```python
def simulate(
    initial_state: np.ndarray,
    time: float,
    dt: Optional[float] = None,  # Explicit Optional
) -> np.ndarray:
    if dt is None:
        dt = time / 100
    ...
```

## Error Handling

### Validate Inputs

```python
def apply_gate(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply gate to quantum state.
    
    Raises:
        ValueError: If dimensions don't match.
        TypeError: If inputs aren't arrays.
    """
    if not isinstance(gate, np.ndarray):
        raise TypeError(f"gate must be ndarray, got {type(gate)}")
    if not isinstance(state, np.ndarray):
        raise TypeError(f"state must be ndarray, got {type(state)}")
    if gate.shape[1] != state.shape[0]:
        raise ValueError(
            f"Gate dimension {gate.shape[1]} doesn't match "
            f"state dimension {state.shape[0]}"
        )
    return gate @ state
```

### Custom Exceptions

```python
class PhysicsValidationError(ValueError):
    """Base class for physics validation errors."""
    pass

class UnitarityError(PhysicsValidationError):
    """Matrix failed unitarity check."""
    pass

class NormalizationError(PhysicsValidationError):
    """State failed normalization check."""
    pass
```

## Testing

### Test Physical Properties

```python
def test_pauli_matrices_anticommute():
    """Pauli matrices satisfy {σi, σj} = 2δij."""
    X, Y, Z = pauli_x(), pauli_y(), pauli_z()
    
    # Anti-commutation
    assert np.allclose(X @ Y + Y @ X, np.zeros((2, 2)))
    assert np.allclose(Y @ Z + Z @ Y, np.zeros((2, 2)))
    assert np.allclose(Z @ X + X @ Z, np.zeros((2, 2)))

def test_density_matrix_properties():
    """Valid density matrix has trace 1 and is positive."""
    rho = create_mixed_state()
    
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho, rho.conj().T)  # Hermitian
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-10)  # Positive
```

### Parameterized Tests

```python
import pytest

@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_identity_is_unitary(n):
    I = np.eye(n, dtype=complex)
    assert np.allclose(I @ I.conj().T, np.eye(n))

@pytest.mark.parametrize("gate,name", [
    (pauli_x, "X"),
    (pauli_y, "Y"),
    (pauli_z, "Z"),
    (hadamard, "H"),
])
def test_gates_are_unitary(gate, name):
    U = gate()
    assert np.allclose(U @ U.conj().T, np.eye(2)), f"{name} is not unitary"
```

## Imports

### Import Order

```python
# Standard library
import os
from typing import Optional, Tuple

# Third party
import numpy as np
import scipy.linalg

# Local
from agentbible import validate_unitary
from .core import compute_expectation
```

### Explicit Imports

```python
# Good: Explicit
from numpy import ndarray
from scipy.linalg import expm, sqrtm

# Avoid: Star imports
from numpy import *  # Bad
```
