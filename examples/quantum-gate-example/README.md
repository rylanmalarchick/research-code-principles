# Quantum Gate Example

A minimal but complete example demonstrating how to apply the five research code principles to quantum computing code.

## What This Example Shows

This ~200-line example implements a `Gate` class that represents quantum gates with:

- **Factory methods** for common gates (X, Z, H, RZ, CNOT)
- **Matrix generation** for simulation
- **Rigorous validation** of inputs and physical constraints
- **Comprehensive tests** covering happy paths, edge cases, and physics

## How the Five Principles Are Applied

### 1. Correctness First

Every gate operation is validated:

```python
# Qubit indices must be non-negative
Gate.x(qubit=-1)  # Raises ValueError

# Rotation angles must be finite
Gate.rz(qubit=0, angle=float("nan"))  # Raises ValueError

# CNOT requires distinct qubits
Gate.cnot(control=0, target=0)  # Raises ValueError
```

Physical constraints are enforced:
```python
# Every matrix is verified to be unitary
def _verify_unitarity(self, matrix: np.ndarray) -> None:
    product = matrix @ matrix.conj().T
    identity = np.eye(matrix.shape[0])
    deviation = np.max(np.abs(product - identity))
    if deviation > UNITARITY_TOLERANCE:
        raise RuntimeError(...)
```

### 2. Simple Beats Clever

The design prioritizes readability:

- **Frozen dataclass** instead of complex class hierarchy
- **Factory methods** instead of string parsing or magic constructors
- **Explicit if/elif** for matrix generation instead of dispatch tables
- **No metaprogramming** - everything is visible at the call site

### 3. Make It Inspectable

Every gate can be understood at a glance:

```python
>>> gate = Gate.rz(qubit=0, angle=math.pi/4)
>>> gate
Gate(gate_type='RZ', qubits=(0,), parameter=0.7853981633974483)
>>> gate.to_matrix()
array([[0.92388-0.38268j, 0.     +0.j     ],
       [0.     +0.j     , 0.92388+0.38268j]])
```

Logging is built in for debugging:
```python
logger.debug("Generating matrix for %s", self)
logger.debug("Unitarity verified, max deviation: %.2e", deviation)
```

### 4. Fail Fast and Loud

Errors happen at construction, not at use:

```python
# Bad: Errors at simulation time (hard to debug)
gate = Gate(gate_type="RZ", qubits=(0,), parameter=None)
matrix = gate.to_matrix()  # Crashes here, far from the bug

# Good: Errors immediately (this code prevents the above)
def _validate_parameter(self) -> None:
    if self.parameter is not None:
        if math.isnan(self.parameter):
            raise ValueError(f"Parameter must be finite, got {self.parameter}.")
```

Error messages include context:
```python
raise RuntimeError(
    f"Unitarity violation: max deviation {deviation:.2e} "
    f"exceeds tolerance {UNITARITY_TOLERANCE:.2e}. "
    f"This indicates a bug in matrix generation for {self}."
)
```

### 5. Reproducibility Is Sacred

- **Frozen dataclass** ensures gates can't be mutated after creation
- **No random state** - matrix generation is deterministic
- **Constants are explicit** - `UNITARITY_TOLERANCE = 1e-10`
- **Citations in docstrings** - every gate references Nielsen & Chuang

## Running the Tests

```bash
# Install dependencies
pip install numpy pytest

# Run all tests
pytest test_gate.py -v

# Run with coverage
pytest test_gate.py --cov=gate --cov-report=term-missing
```

## Test Organization

The test file demonstrates proper test structure:

| Test Class | Purpose |
|------------|---------|
| `TestGateConstruction` | Factory methods work correctly |
| `TestInvalidInputs` | Bad inputs fail fast with clear messages |
| `TestMatrixGeneration` | Matrices match textbook definitions |
| `TestEdgeCases` | Boundary values (0, 2π, negative angles) |
| `TestPhysicalProperties` | Quantum mechanics is correct (X²=I, unitarity) |
| `TestIsUnitaryHelper` | Utility function works correctly |

## Extending This Example

To add a new gate:

1. Add a factory method (e.g., `Gate.ry(qubit, angle)`)
2. Add matrix generation in `_compute_matrix()`
3. Add tests for construction, matrix, and physical properties
4. Include citation in the docstring

## References

- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information* (10th Anniversary ed.). Cambridge University Press.
