"""Quantum physics constraint violation errors.

Each error includes:
- A clear description of what went wrong
- The expected physical constraint
- The actual observed value
- Academic references for understanding the constraint
- Guidance on how to fix the issue

References are provided to help researchers understand the underlying physics.
"""

from __future__ import annotations

from agentbible.errors import PhysicsConstraintError


class UnitarityError(PhysicsConstraintError):
    """Raised when a matrix fails the unitarity constraint U†U = I.

    A unitary matrix preserves inner products and is essential for
    reversible quantum operations. All quantum gates must be unitary.
    """

    REFERENCE = (
        "Nielsen & Chuang, 'Quantum Computation and Quantum Information', "
        "Theorem 2.2 (Box 2.2). https://doi.org/10.1017/CBO9780511976667"
    )
    GUIDANCE = (
        "Your quantum gate is not reversible. Common causes:\n"
        "    - Missing normalization factor (e.g., 1/sqrt(2) for Hadamard)\n"
        "    - Incorrect matrix elements or signs\n"
        "    - Numerical precision issues (try increasing tolerance)"
    )


class HermiticityError(PhysicsConstraintError):
    """Raised when a matrix fails the Hermitian constraint H = H†.

    Hermitian (self-adjoint) matrices have real eigenvalues and are
    required for physical observables in quantum mechanics.
    """

    REFERENCE = (
        "Nielsen & Chuang, 'Quantum Computation and Quantum Information', "
        "Definition 2.5. Sakurai & Napolitano, 'Modern Quantum Mechanics', "
        "Section 1.3."
    )
    GUIDANCE = (
        "Observables must be self-adjoint (H = H†). Common causes:\n"
        "    - Off-diagonal elements not complex conjugates\n"
        "    - Diagonal elements not purely real\n"
        "    - Transposed matrix instead of conjugate transpose"
    )


class DensityMatrixError(PhysicsConstraintError):
    """Raised when a matrix fails density matrix constraints.

    A valid density matrix rho must satisfy:
    1. Hermitian: rho = rho†
    2. Unit trace: tr(rho) = 1 (normalization)
    3. Positive semi-definite: all eigenvalues >= 0

    This ensures the matrix represents a valid quantum state.
    """

    REFERENCE = (
        "Nielsen & Chuang, 'Quantum Computation and Quantum Information', "
        "Section 2.4. Preskill, 'Lecture Notes for Physics 229: Quantum "
        "Information and Computation', Chapter 3."
    )
    GUIDANCE = (
        "Density matrix represents a quantum state. Common causes of failure:\n"
        "    - Trace != 1: Matrix not properly normalized\n"
        "    - Negative eigenvalue: State is unphysical (check your operations)\n"
        "    - Not Hermitian: Check complex conjugation of off-diagonal elements"
    )


class TraceError(PhysicsConstraintError):
    """Raised when a matrix has incorrect trace.

    The trace of a density matrix must be 1 (normalization).
    Trace preservation is also required for quantum channels.
    """

    REFERENCE = (
        "Nielsen & Chuang, 'Quantum Computation and Quantum Information', "
        "Section 8.2.4 on trace-preserving operations."
    )
    GUIDANCE = (
        "Trace must equal expected value. Common causes:\n"
        "    - Missing normalization (divide by trace)\n"
        "    - Trace-decreasing operation not accounted for\n"
        "    - Numerical accumulation of errors"
    )


class PositivityError(PhysicsConstraintError):
    """Raised when a matrix fails positive semi-definiteness.

    Positive semi-definite matrices have all eigenvalues >= 0.
    This is required for density matrices, covariance matrices,
    and Gram matrices.
    """

    REFERENCE = (
        "Horn & Johnson, 'Matrix Analysis', Chapter 7 on positive definite "
        "matrices. Higham, 'Functions of Matrices: Theory and Computation'."
    )
    GUIDANCE = (
        "All eigenvalues must be >= 0. Common causes:\n"
        "    - Numerical instability in matrix construction\n"
        "    - Unphysical operation (e.g., measurement back-action)\n"
        "    - Try eigenvalue regularization: max(lambda, 0)"
    )


__all__ = [
    "UnitarityError",
    "HermiticityError",
    "DensityMatrixError",
    "TraceError",
    "PositivityError",
]
