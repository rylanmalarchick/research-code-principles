"""Decorators for physics-aware testing.

Provides the @physics_test decorator for automatic validation of physics constraints.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import numpy as np

from agentbible.validators import ValidationError

F = TypeVar("F", bound=Callable[..., Any])

# Available physics checks and their implementations
PHYSICS_CHECKS: dict[str, Callable[[np.ndarray, float, float], None]] = {}


def _check_unitarity(matrix: np.ndarray, rtol: float, atol: float) -> None:
    """Check if matrix is unitary: U @ U.H = I."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("Unitarity check requires a square matrix")

    product = matrix @ matrix.conj().T
    identity = np.eye(matrix.shape[0], dtype=complex)

    if not np.allclose(product, identity, rtol=rtol, atol=atol):
        max_diff: np.floating[Any] = np.max(np.abs(product - identity))
        raise ValidationError(
            f"Matrix is not unitary: max deviation from identity = {max_diff:.2e}"
        )


def _check_hermiticity(matrix: np.ndarray, rtol: float, atol: float) -> None:
    """Check if matrix is Hermitian: A = A.H."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("Hermiticity check requires a square matrix")

    if not np.allclose(matrix, matrix.conj().T, rtol=rtol, atol=atol):
        max_diff: np.floating[Any] = np.max(np.abs(matrix - matrix.conj().T))
        raise ValidationError(
            f"Matrix is not Hermitian: max asymmetry = {max_diff:.2e}"
        )


def _check_trace_one(matrix: np.ndarray, rtol: float, atol: float) -> None:
    """Check if matrix has trace = 1."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("Trace check requires a square matrix")

    trace = np.trace(matrix)
    if not np.isclose(trace, 1.0, rtol=rtol, atol=atol):
        raise ValidationError(f"Matrix trace is {trace:.6f}, expected 1.0")


def _check_positive_semidefinite(matrix: np.ndarray, rtol: float, atol: float) -> None:
    """Check if matrix is positive semi-definite (all eigenvalues >= 0)."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("Positive semi-definite check requires a square matrix")

    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eigenvalue: np.floating[Any] = np.min(eigenvalues.real)

    if min_eigenvalue < -atol:
        raise ValidationError(f"Matrix has negative eigenvalue: {min_eigenvalue:.2e}")


def _check_normalization(vector: np.ndarray, rtol: float, atol: float) -> None:
    """Check if vector is normalized: ||v|| = 1."""
    norm = np.linalg.norm(vector)
    if not np.isclose(norm, 1.0, rtol=rtol, atol=atol):
        raise ValidationError(f"Vector norm is {norm:.6f}, expected 1.0")


def _check_probability(array: np.ndarray, rtol: float, atol: float) -> None:
    """Check if all values are valid probabilities in [0, 1]."""
    if np.any(array < -atol) or np.any(array > 1 + atol):
        min_val: np.floating[Any] = np.min(array)
        max_val: np.floating[Any] = np.max(array)
        raise ValidationError(
            f"Values outside probability range [0, 1]: min={min_val:.2e}, max={max_val:.2e}"
        )


# Register all checks
PHYSICS_CHECKS = {
    "unitarity": _check_unitarity,
    "hermiticity": _check_hermiticity,
    "trace_one": _check_trace_one,
    "positive_semidefinite": _check_positive_semidefinite,
    "normalization": _check_normalization,
    "probability": _check_probability,
}


def physics_test(
    checks: list[str] | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> Callable[[F], F]:
    """Decorator for physics-aware tests.

    Automatically validates the returned value against physics constraints.

    Args:
        checks: List of checks to perform. Available checks:
            - "unitarity": U @ U.H = I
            - "hermiticity": A = A.H
            - "trace_one": tr(A) = 1
            - "positive_semidefinite": All eigenvalues >= 0
            - "normalization": ||v|| = 1
            - "probability": All values in [0, 1]
        rtol: Relative tolerance for comparisons
        atol: Absolute tolerance for comparisons

    Returns:
        Decorated function that validates its return value

    Example:
        @physics_test(checks=["unitarity", "hermiticity"])
        def test_pauli_x():
            return np.array([[0, 1], [1, 0]], dtype=complex)

        @physics_test(checks=["trace_one", "positive_semidefinite"])
        def test_density_matrix():
            return np.array([[0.5, 0], [0, 0.5]], dtype=complex)

    Raises:
        ValidationError: If any check fails
        ValueError: If an unknown check is specified
    """
    if checks is None:
        checks = []

    # Validate check names
    unknown_checks = set(checks) - set(PHYSICS_CHECKS.keys())
    if unknown_checks:
        raise ValueError(
            f"Unknown physics checks: {unknown_checks}. "
            f"Available: {list(PHYSICS_CHECKS.keys())}"
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Only validate if result is a numpy array
            if isinstance(result, np.ndarray):
                for check_name in checks:
                    check_func = PHYSICS_CHECKS[check_name]
                    try:
                        check_func(result, rtol, atol)
                    except ValidationError as e:
                        raise ValidationError(
                            f"Physics check '{check_name}' failed in {func.__name__}: {e}"
                        ) from e

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def get_available_checks() -> list[str]:
    """Return list of available physics checks."""
    return list(PHYSICS_CHECKS.keys())
