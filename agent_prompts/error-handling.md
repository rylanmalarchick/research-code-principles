# Error Handling Prompt

Use this for fail-fast patterns in research code.

---

## Principle: Fail Fast with Clarity

Detect errors at boundaries and report immediately with context. Silent failures are the enemy of correctness.

---

## Python Pattern

```python
def optimize_unitary(
    target: np.ndarray,
    max_iterations: int = 1000,
) -> OptimizationResult:
    """Optimize pulses to implement target unitary.

    Args:
        target: Target unitary matrix, shape (d, d).
        max_iterations: Maximum optimization iterations.

    Raises:
        ValueError: If target is not square or not unitary.
        TypeError: If max_iterations is not an integer.
    """
    # Validate inputs at boundary
    if not isinstance(max_iterations, int):
        raise TypeError(
            f"max_iterations must be int, got {type(max_iterations).__name__}"
        )

    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}")

    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError(f"Target must be square, got shape {target.shape}")

    identity = np.eye(target.shape[0])
    if not np.allclose(target @ target.conj().T, identity):
        raise ValueError("Target must be unitary (U @ U† = I)")

    # Now proceed with validated inputs...
```

---

## C++ Pattern

```cpp
void Circuit::addGate(const Gate& gate) {
    if (gate.qubit() >= numQubits_) {
        throw std::out_of_range(
            "Qubit index " + std::to_string(gate.qubit()) +
            " out of range [0, " + std::to_string(numQubits_ - 1) + "]"
        );
    }

    if (gate.isParameterized() && std::isnan(gate.parameter())) {
        throw std::invalid_argument("Gate parameter is NaN");
    }

    gates_.push_back(gate);
}
```

---

## CUDA Pattern

```cpp
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            throw std::runtime_error(                                   \
                std::string("CUDA error in ") + __FILE__ + ":" +       \
                std::to_string(__LINE__) + ": " +                      \
                cudaGetErrorString(err));                               \
        }                                                               \
    } while (0)

// Never ignore return values
CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
```

---

## Error Message Components

Every error message should include:

1. **What** went wrong: "Qubit index 5 out of range"
2. **Expected**: "[0, 4]"
3. **Where**: File, line, function (automatic in exceptions)

```python
raise ValueError(
    f"Qubit index {qubit} out of range [0, {self.num_qubits - 1}]"
)
```
