"""Physics constraint violation errors with academic references.

This module provides specialized error classes for physics validation failures.
Each error includes:
- A clear description of what went wrong
- The expected physical constraint
- The actual observed value
- Academic references for understanding the constraint
- Guidance on how to fix the issue

References are provided to help researchers understand the underlying physics
and to make errors educational rather than just informational.
"""

from __future__ import annotations

from typing import Any


class ValidationError(Exception):
    """Raised when a physics validation check fails.

    This is the base class for all validation errors. It provides
    structured error messages with expected/actual values and context.

    Attributes:
        message: Description of what validation failed.
        expected: What the value should satisfy.
        got: What was actually observed.
        function_name: Name of the decorated function.
        tolerance: Tolerance values used in comparison.
        shape: Array shape if applicable.
    """

    def __init__(
        self,
        message: str,
        *,
        expected: str | None = None,
        got: str | None = None,
        function_name: str | None = None,
        tolerance: dict[str, float] | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self.message = message
        self.expected = expected
        self.got = got
        self.function_name = function_name
        self.tolerance = tolerance
        self.shape = shape

        # Build detailed error message
        parts = [message]
        if expected:
            parts.append(f"  Expected: {expected}")
        if got:
            parts.append(f"  Got: {got}")
        if tolerance:
            tol_str = ", ".join(f"{k}={v}" for k, v in tolerance.items())
            parts.append(f"  Tolerance: {tol_str}")
        if shape:
            parts.append(f"  Shape: {shape}")
        if function_name:
            parts.append(f"  Function: {function_name}")

        super().__init__("\n".join(parts))


class PhysicsConstraintError(ValidationError):
    """Base class for physics constraint violations.

    All physics-specific errors inherit from this class, making it easy
    to catch any physics-related validation failure.

    Attributes:
        message: Description of the violation.
        expected: What the value should satisfy.
        got: What was actually observed.
        function_name: Name of the function that failed validation.
        tolerance: Tolerance values used in comparison.
        shape: Array shape if applicable.
        reference: Academic reference for the constraint.
        guidance: Helpful guidance on how to fix the issue.
    """

    REFERENCE: str | None = None
    GUIDANCE: str | None = None

    def __init__(
        self,
        message: str,
        *,
        expected: str | None = None,
        got: str | None = None,
        function_name: str | None = None,
        tolerance: dict[str, float] | None = None,
        shape: tuple[int, ...] | None = None,
        reference: str | None = None,
        guidance: str | None = None,
    ) -> None:
        self.message = message
        self.expected = expected
        self.got = got
        self.function_name = function_name
        self.tolerance = tolerance
        self.shape = shape
        # Use class-level defaults if not provided
        self.reference = reference or self.__class__.REFERENCE
        self.guidance = guidance or self.__class__.GUIDANCE

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Build detailed, educational error message."""
        parts = [f"\n{self.message.upper()}"]
        parts.append("")  # Blank line

        if self.expected:
            parts.append(f"  Expected: {self.expected}")
        if self.got:
            parts.append(f"  Got: {self.got}")
        if self.tolerance:
            tol_str = ", ".join(f"{k}={v}" for k, v in self.tolerance.items())
            parts.append(f"  Tolerance: {tol_str}")
        if self.shape:
            parts.append(f"  Shape: {self.shape}")
        if self.function_name:
            parts.append(f"  Function: {self.function_name}")

        if self.reference:
            parts.append("")
            parts.append(f"  Reference: {self.reference}")

        if self.guidance:
            parts.append("")
            parts.append(f"  Guidance: {self.guidance}")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to a dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "expected": self.expected,
            "got": self.got,
            "function_name": self.function_name,
            "tolerance": self.tolerance,
            "shape": self.shape,
            "reference": self.reference,
            "guidance": self.guidance,
        }


# =============================================================================
# Quantum Physics Errors
# =============================================================================


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
        "    - Missing normalization factor (e.g., 1/√2 for Hadamard)\n"
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

    A valid density matrix ρ must satisfy:
    1. Hermitian: ρ = ρ†
    2. Unit trace: tr(ρ) = 1 (normalization)
    3. Positive semi-definite: all eigenvalues ≥ 0

    This ensures the matrix represents a valid quantum state.
    """

    REFERENCE = (
        "Nielsen & Chuang, 'Quantum Computation and Quantum Information', "
        "Section 2.4. Preskill, 'Lecture Notes for Physics 229: Quantum "
        "Information and Computation', Chapter 3."
    )
    GUIDANCE = (
        "Density matrix represents a quantum state. Common causes of failure:\n"
        "    - Trace ≠ 1: Matrix not properly normalized\n"
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

    Positive semi-definite matrices have all eigenvalues ≥ 0.
    This is required for density matrices, covariance matrices,
    and Gram matrices.
    """

    REFERENCE = (
        "Horn & Johnson, 'Matrix Analysis', Chapter 7 on positive definite "
        "matrices. Higham, 'Functions of Matrices: Theory and Computation'."
    )
    GUIDANCE = (
        "All eigenvalues must be ≥ 0. Common causes:\n"
        "    - Numerical instability in matrix construction\n"
        "    - Unphysical operation (e.g., measurement back-action)\n"
        "    - Try eigenvalue regularization: max(λ, 0)"
    )


# =============================================================================
# Probability Errors
# =============================================================================


class ProbabilityBoundsError(PhysicsConstraintError):
    """Raised when a probability value is outside [0, 1].

    Probabilities must be non-negative and cannot exceed 1.
    This is a fundamental axiom of probability theory.
    """

    REFERENCE = (
        "Kolmogorov, 'Foundations of the Theory of Probability', Axioms I-III. "
        "Jaynes, 'Probability Theory: The Logic of Science', Chapter 2."
    )
    GUIDANCE = (
        "Probability must be in [0, 1]. Common causes:\n"
        "    - Numerical overflow/underflow\n"
        "    - Log-probability not converted back correctly\n"
        "    - Sum of exclusive events exceeds 1 (not mutually exclusive)"
    )


class NormalizationError(PhysicsConstraintError):
    """Raised when a probability distribution doesn't sum to 1.

    A valid probability distribution over a discrete sample space
    must sum to exactly 1.
    """

    REFERENCE = (
        "Kolmogorov, 'Foundations of the Theory of Probability'. "
        "For quantum states: Born rule requires |ψ|² probabilities sum to 1."
    )
    GUIDANCE = (
        "Distribution must sum to 1. Common causes:\n"
        "    - Forgot to normalize after computing unnormalized scores\n"
        "    - Softmax numerical instability (subtract max before exp)\n"
        "    - Truncated distribution (e.g., top-k without renormalization)"
    )


class StateVectorNormError(PhysicsConstraintError):
    """Raised when a quantum state vector is not normalized.

    A quantum state vector |ψ⟩ must satisfy ⟨ψ|ψ⟩ = 1.
    This ensures the total probability of all measurement outcomes is 1.
    """

    REFERENCE = (
        "Nielsen & Chuang, 'Quantum Computation and Quantum Information', "
        "Postulate 1 (Section 2.2.1). Born rule interpretation."
    )
    GUIDANCE = (
        "State vector must have unit norm (||ψ|| = 1). Common causes:\n"
        "    - Forgot to normalize after state preparation\n"
        "    - Non-unitary evolution (decoherence without renormalization)\n"
        "    - Numerical precision loss in long computations"
    )


# =============================================================================
# Numerical Errors
# =============================================================================


class NonFiniteError(PhysicsConstraintError):
    """Raised when array contains NaN or Inf values.

    Physical quantities must be finite. NaN and Inf typically indicate
    numerical instability in the computation.
    """

    REFERENCE = (
        "Higham, 'Accuracy and Stability of Numerical Algorithms', 2nd Ed. "
        "IEEE 754 floating-point standard."
    )
    GUIDANCE = (
        "All values must be finite (no NaN or Inf). Common causes:\n"
        "    - Division by zero (check denominators)\n"
        "    - Log of zero or negative number\n"
        "    - Overflow in exponential (use log-space computation)\n"
        "    - Underflow to zero then reciprocal (use numerically stable formulas)"
    )


class BoundsError(PhysicsConstraintError):
    """Raised when a value is outside expected bounds.

    Many physical quantities have natural bounds (e.g., energy ≥ 0,
    fidelity in [0, 1], correlation in [-1, 1]).
    """

    REFERENCE = (
        "Problem-specific. For fidelity: Jozsa, 'Fidelity for Mixed Quantum "
        "States', J. Mod. Opt. 41, 2315 (1994)."
    )
    GUIDANCE = (
        "Value outside expected range. Common causes:\n"
        "    - Units mismatch (e.g., radians vs degrees)\n"
        "    - Forgot to apply constraint (e.g., clipping, abs)\n"
        "    - Formula error introducing sign flip"
    )


# =============================================================================
# Convenience mapping for backward compatibility
# =============================================================================

# Map from constraint type to error class
ERROR_CLASSES = {
    "unitarity": UnitarityError,
    "hermiticity": HermiticityError,
    "density_matrix": DensityMatrixError,
    "trace": TraceError,
    "positivity": PositivityError,
    "probability": ProbabilityBoundsError,
    "normalization": NormalizationError,
    "state_vector": StateVectorNormError,
    "finite": NonFiniteError,
    "bounds": BoundsError,
}


__all__ = [
    "ValidationError",
    "PhysicsConstraintError",
    "UnitarityError",
    "HermiticityError",
    "DensityMatrixError",
    "TraceError",
    "PositivityError",
    "ProbabilityBoundsError",
    "NormalizationError",
    "StateVectorNormError",
    "NonFiniteError",
    "BoundsError",
    "ERROR_CLASSES",
]
