# AgentBible Correctness Specification
**Version:** 1.0
**Date:** 2026-04-24
**Applies to:** agentbible (Python), agentbible-rs (Rust), agentbible C++ headers, AgentBible.jl (Julia)

All implementations must use these exact mathematical definitions and default tolerances.

---

## Default Tolerances

Unless overridden by caller:

| Parameter | Default   | Notes |
|-----------|-----------|-------|
| `rtol`    | `1e-10`   | Relative tolerance |
| `atol`    | `1e-12`   | Absolute tolerance |

Tighter than NumPy defaults by design.

---

## Scalar Checks

### finite
    pass iff: !isnan(x) && !isinf(x)

### positive
    pass iff: x > 0.0 && isfinite(x)

### non_negative
    pass iff: x >= 0.0 && isfinite(x)

### probability
    pass iff: 0.0 <= x <= 1.0 && isfinite(x)

---

## Array Checks

All array checks verify finite for all elements first.

### finite_array
    pass iff: finite(x[i]) for all i

### positive_array
    pass iff: x[i] > 0.0 for all i

### non_negative_array
    pass iff: x[i] >= 0.0 for all i

### probability_array
    pass iff: 0.0 <= x[i] <= 1.0 for all i

### normalized_l1
    pass iff: |sum(x[i]) - 1.0| <= atol
Default atol = 1e-10 (looser than scalar atol; float accumulation over n).

---

## Matrix Checks

All matrices n x n unless noted.

### symmetric
Norm: max elementwise (L-inf on A - A^T)

    pass iff: |A[i,j] - A[j,i]| <= atol  for all i, j

### hermitian
Norm: max elementwise on |A - A†|

    pass iff: |A[i,j] - conj(A[j,i])| <= atol  for all i, j

### unitary
Norm: Frobenius on residual U†U - I. Tolerance scales with n.

    residual = U†U - I
    pass iff: ||residual||_F <= atol + rtol * n

### positive_definite
Structural property — no tolerance parameter.

    pass iff: Cholesky factorization of A succeeds

### positive_semidefinite
    pass iff: all eigenvalues of A >= -atol

### density_matrix
All three must hold simultaneously:
1. hermitian: passes hermitian check
2. trace = 1: |tr(rho) - 1| <= atol
3. positive_semidefinite: passes positive_semidefinite check

---

## Provenance

Every check emits a record compliant with schema/provenance_v1.json.
check_name field must exactly match check names above (case-sensitive).

---

## Conformance

An implementation is conformant iff:
1. Implements all checks in this spec
2. Default tolerances match exactly
3. Conformance test suite passes (tests/test_conformance.* in each language dir)
4. Provenance records parse against schema/provenance_v1.json

Required conformance tests per language:
- A matrix that passes unitary at default tolerance
- Same matrix perturbed by 1e-9 that still passes
- Same matrix perturbed by 1e-5 that fails
- NaN injection that fails finite
- Provenance JSON roundtrip
