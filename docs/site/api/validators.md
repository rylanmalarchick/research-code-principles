# API Reference: Validators

::: agentbible.validators

## Quantum Validators

### validate_unitary

```python
from agentbible import validate_unitary

@validate_unitary(rtol=1e-10, atol=1e-12)
def create_gate() -> np.ndarray:
    ...
```

Validates that the returned matrix satisfies U @ U.H = I.

**Parameters:**

- `rtol` (float): Relative tolerance. Default: 1e-10
- `atol` (float): Absolute tolerance. Default: 1e-12
- `enabled` (bool): Whether validation is active. Default: True

**Raises:**

- `UnitarityError`: If matrix is not unitary within tolerance

---

### validate_hermitian

```python
from agentbible import validate_hermitian

@validate_hermitian(rtol=1e-10, atol=1e-12)
def create_observable() -> np.ndarray:
    ...
```

Validates that the returned matrix equals its conjugate transpose: A = A.H.

**Parameters:**

- `rtol` (float): Relative tolerance. Default: 1e-10
- `atol` (float): Absolute tolerance. Default: 1e-12

**Raises:**

- `HermiticityError`: If matrix is not Hermitian within tolerance

---

### validate_density_matrix

```python
from agentbible import validate_density_matrix

@validate_density_matrix
def create_state() -> np.ndarray:
    ...
```

Validates all density matrix properties:

- Hermitian (A = A.H)
- Unit trace (Tr(A) = 1)
- Positive semi-definite (all eigenvalues ≥ 0)

**Raises:**

- `DensityMatrixError`: If any property is violated

---

## Probability Validators

### validate_probability

```python
from agentbible import validate_probability

@validate_probability
def compute_likelihood() -> float:
    ...
```

Validates that the returned value is in the range [0, 1].

**Raises:**

- `ProbabilityError`: If value is outside [0, 1]

---

### validate_probabilities

```python
from agentbible import validate_probabilities

@validate_probabilities
def compute_distribution() -> np.ndarray:
    ...
```

Validates that all elements of the returned array are in [0, 1].

**Raises:**

- `ProbabilityError`: If any element is outside [0, 1]

---

### validate_normalized

```python
from agentbible import validate_normalized

@validate_normalized(norm_type="sum")  # or "l2"
def create_distribution() -> np.ndarray:
    ...
```

Validates normalization.

**Parameters:**

- `norm_type` (str): "sum" for probability distributions, "l2" for state vectors
- `rtol` (float): Relative tolerance. Default: 1e-10
- `atol` (float): Absolute tolerance. Default: 1e-12

**Raises:**

- `NormalizationError`: If normalization check fails

---

## Bounds Validators

### validate_positive

```python
from agentbible import validate_positive

@validate_positive
def compute_energy() -> float:
    ...
```

Validates that the returned value is strictly positive (> 0).

---

### validate_non_negative

```python
from agentbible import validate_non_negative

@validate_non_negative
def compute_entropy() -> float:
    ...
```

Validates that the returned value is non-negative (≥ 0).

---

### validate_range

```python
from agentbible import validate_range

@validate_range(min_val=0, max_val=2*np.pi, inclusive=True)
def compute_angle() -> float:
    ...
```

Validates that the returned value is within the specified range.

**Parameters:**

- `min_val` (float, optional): Minimum allowed value
- `max_val` (float, optional): Maximum allowed value
- `inclusive` (bool): Whether bounds are inclusive. Default: True

---

### validate_finite

```python
from agentbible import validate_finite

@validate_finite
def compute_result() -> np.ndarray:
    ...
```

Validates that the returned value contains no NaN or Inf values.

**Raises:**

- `ValueError`: If NaN or Inf is detected

---

## Exception Classes

```python
from agentbible.validators import (
    PhysicsValidationError,  # Base class
    UnitarityError,
    HermiticityError,
    DensityMatrixError,
    ProbabilityError,
    NormalizationError,
    BoundsError,
)
```

All validation exceptions inherit from `PhysicsValidationError`, which inherits from `ValueError`.
