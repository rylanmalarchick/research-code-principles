# Physics Validation Prompt

Use this when implementing quantum computing or physics code.

---

## Quantum State Validation

```python
def check_normalized(state: np.ndarray, tol: float = 1e-10) -> None:
    """Validate ⟨ψ|ψ⟩ = 1."""
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=tol):
        raise ValueError(f"State not normalized: |⟨ψ|ψ⟩| = {norm}")
```

---

## Unitary Validation

```python
def check_unitarity(U: np.ndarray, tol: float = 1e-10) -> None:
    """Validate U†U = I."""
    identity = np.eye(U.shape[0])
    if not np.allclose(U.conj().T @ U, identity, atol=tol):
        max_err = np.max(np.abs(U.conj().T @ U - identity))
        raise ValueError(f"Matrix not unitary: max|U†U - I| = {max_err:.2e}")
```

---

## Density Matrix Validation

```python
def check_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> None:
    """Validate density matrix properties."""
    # Hermitian: ρ = ρ†
    if not np.allclose(rho, rho.conj().T, atol=tol):
        raise ValueError("Density matrix not Hermitian")

    # Trace 1
    if not np.isclose(np.trace(rho), 1.0, atol=tol):
        raise ValueError(f"Density matrix trace = {np.trace(rho)}")

    # Positive semidefinite
    eigenvalues = np.linalg.eigvalsh(rho)
    if np.min(eigenvalues) < -tol:
        raise ValueError(f"Negative eigenvalue: {np.min(eigenvalues)}")
```

---

## Probability Validation

```python
def check_probabilities(probs: np.ndarray, tol: float = 1e-10) -> None:
    """Validate probability distribution."""
    if np.any(probs < -tol):
        raise ValueError(f"Negative probability: {np.min(probs)}")
    if not np.isclose(np.sum(probs), 1.0, atol=tol):
        raise ValueError(f"Probabilities sum to {np.sum(probs)}")
```

---

## Fidelity Calculations

```python
# State fidelity: F = |⟨ψ|φ⟩|²
# Reference: Nielsen & Chuang, Eq. 9.52
fidelity = np.abs(np.vdot(psi, phi))**2
assert 0 <= fidelity <= 1

# Gate fidelity (average): F = |Tr(U†V)|² / d²
# Reference: Nielsen & Chuang, Eq. 9.113
d = U.shape[0]
fidelity = np.abs(np.trace(U.conj().T @ V))**2 / d**2
```

---

## Citation Format

Always cite sources for physics equations:

```python
def compute_fidelity(U: np.ndarray, V: np.ndarray) -> float:
    """Compute average gate fidelity.

    F = |Tr(U†V)|² / d²

    Reference:
        Nielsen & Chuang, "Quantum Computation and Quantum Information",
        Eq. 9.113 (2010). ISBN: 978-1107002173
    """
```
