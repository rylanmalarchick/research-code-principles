"""Physics constraint violation errors with academic references.

This module provides the core error classes for physics validation failures.
Each error includes:
- A clear description of what went wrong
- The expected physical constraint
- The actual observed value
- Academic references for understanding the constraint
- Guidance on how to fix the issue

Domain-specific errors (e.g., quantum) are located in their respective domains:
    from agentbible.domains.quantum import UnitarityError, HermiticityError
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
# Probability Errors (Core)
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
        "For quantum states: Born rule requires |psi|^2 probabilities sum to 1."
    )
    GUIDANCE = (
        "Distribution must sum to 1. Common causes:\n"
        "    - Forgot to normalize after computing unnormalized scores\n"
        "    - Softmax numerical instability (subtract max before exp)\n"
        "    - Truncated distribution (e.g., top-k without renormalization)"
    )


class StateVectorNormError(PhysicsConstraintError):
    """Raised when a state vector is not normalized.

    A state vector |psi> must satisfy <psi|psi> = 1.
    This ensures the total probability of all measurement outcomes is 1.
    """

    REFERENCE = (
        "Nielsen & Chuang, 'Quantum Computation and Quantum Information', "
        "Postulate 1 (Section 2.2.1). Born rule interpretation."
    )
    GUIDANCE = (
        "State vector must have unit norm (||psi|| = 1). Common causes:\n"
        "    - Forgot to normalize after state preparation\n"
        "    - Non-unitary evolution (decoherence without renormalization)\n"
        "    - Numerical precision loss in long computations"
    )


# =============================================================================
# Numerical Errors (Core)
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

    Many physical quantities have natural bounds (e.g., energy >= 0,
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
# Convenience mapping for core errors
# =============================================================================

# Map from constraint type to error class (core only)
ERROR_CLASSES = {
    "probability": ProbabilityBoundsError,
    "normalization": NormalizationError,
    "state_vector": StateVectorNormError,
    "finite": NonFiniteError,
    "bounds": BoundsError,
}


__all__ = [
    # Base classes
    "ValidationError",
    "PhysicsConstraintError",
    # Probability errors
    "ProbabilityBoundsError",
    "NormalizationError",
    "StateVectorNormError",
    # Numerical errors
    "NonFiniteError",
    "BoundsError",
    # Mapping
    "ERROR_CLASSES",
]
