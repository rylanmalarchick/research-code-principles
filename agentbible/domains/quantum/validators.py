"""Quantum physics validation decorators.

Provides decorators for validating quantum mechanical properties:
- Unitarity: U†U = I
- Hermiticity: H = H†
- Density matrix: Hermitian, trace 1, positive semi-definite

All validators include:
- Numerical sanity checks (NaN/Inf detection before physics checks)
- Support for both rtol and atol tolerances
- Educational error messages with academic references
- Conditional validation levels (debug/lite/off)

Validation levels:
- "debug" (default): Full validation with all physics checks
- "lite": Only NaN/Inf sanity checks (fast)
- "off": Skip validation entirely (DANGEROUS - for benchmarking only)
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

from agentbible.domains.quantum.errors import (
    DensityMatrixError,
    HermiticityError,
    PositivityError,
    TraceError,
    UnitarityError,
)
from agentbible.errors import NonFiniteError
from agentbible.validators.base import (
    ValidationLevel,
    get_numpy,
    get_validation_level,
    is_square_matrix,
    maybe_warn_validation_off,
)

F = TypeVar("F", bound=Callable[..., Any])


def _check_finite(arr: Any, function_name: str) -> None:
    """Check array for NaN/Inf values before physics validation.

    This is called before any physics checks to provide clearer error messages
    when the root cause is numerical instability rather than physics violations.
    """
    np = get_numpy()
    if not np.all(np.isfinite(arr)):
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        details = []
        if nan_count > 0:
            details.append(f"{nan_count} NaN")
        if inf_count > 0:
            details.append(f"{inf_count} Inf")

        raise NonFiniteError(
            "Matrix contains non-finite values",
            expected="All finite values (no NaN or Inf)",
            got=", ".join(details),
            function_name=function_name,
            shape=arr.shape,
        )


@overload
def validate_unitary(func: F) -> F: ...


@overload
def validate_unitary(
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_unitary(
    func: F | None = None,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    """Validate that a function returns a unitary matrix.

    A matrix U is unitary if U†U = I (conjugate transpose times self equals identity).

    Can be used with or without arguments:
        @validate_unitary
        def make_gate(): ...

        @validate_unitary(rtol=1e-6)
        def make_precise_gate(): ...

        @validate_unitary(level="lite")  # Only check for NaN/Inf
        def make_gate_fast(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        rtol: Relative tolerance for comparison. Default 1e-5.
        atol: Absolute tolerance for comparison. Default 1e-8.
        level: Validation level - "debug" (full), "lite" (NaN/Inf only), or "off".
            Can also be set globally via AGENTBIBLE_VALIDATION_LEVEL env var.
            WARNING: "off" disables ALL validation - use only for benchmarking.

    Returns:
        Decorated function that validates its return value.

    Raises:
        UnitarityError: If the returned matrix is not unitary (level="debug").
        NonFiniteError: If the matrix contains NaN/Inf (level="debug" or "lite").

    Example:
        >>> import numpy as np
        >>> @validate_unitary
        ... def hadamard():
        ...     return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        >>> H = hadamard()  # Passes validation
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)

            # Determine effective validation level
            effective_level = get_validation_level(level)

            # If validation is off, skip everything (with warning)
            if effective_level == ValidationLevel.OFF:
                maybe_warn_validation_off(fn.__name__)
                return result

            np = get_numpy()

            # Convert to numpy array if needed
            arr = np.asarray(result)

            # Check for NaN/Inf FIRST (both debug and lite modes)
            _check_finite(arr, fn.__name__)

            # If lite mode, stop here (no physics checks)
            if effective_level == ValidationLevel.LITE:
                return result

            # Full validation (debug mode)
            # Check shape
            if not is_square_matrix(arr):
                raise UnitarityError(
                    "Matrix is not unitary",
                    expected="Square matrix",
                    got=f"Shape {arr.shape}",
                    function_name=fn.__name__,
                    tolerance={"rtol": rtol, "atol": atol},
                    shape=arr.shape,
                )

            # Check unitarity: U†U = I
            identity = np.eye(arr.shape[0], dtype=arr.dtype)
            product = arr.conj().T @ arr
            max_deviation = float(np.max(np.abs(product - identity)))

            if not np.allclose(product, identity, rtol=rtol, atol=atol):
                raise UnitarityError(
                    "Matrix is not unitary",
                    expected="U†U = I (conjugate transpose times matrix equals identity)",
                    got=f"max|U†U - I| = {max_deviation:.2e}",
                    function_name=fn.__name__,
                    tolerance={"rtol": rtol, "atol": atol},
                    shape=arr.shape,
                )

            return result

        return wrapper  # type: ignore[return-value]

    # Handle both @validate_unitary and @validate_unitary()
    if func is not None:
        return decorator(func)
    return decorator


@overload
def validate_hermitian(func: F) -> F: ...


@overload
def validate_hermitian(
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_hermitian(
    func: F | None = None,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    """Validate that a function returns a Hermitian matrix.

    A matrix H is Hermitian if H = H† (equals its conjugate transpose).

    Can be used with or without arguments:
        @validate_hermitian
        def make_hamiltonian(): ...

        @validate_hermitian(rtol=1e-6)
        def make_precise_hamiltonian(): ...

        @validate_hermitian(level="off")  # DANGEROUS - for benchmarking only
        def make_hamiltonian_fast(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        rtol: Relative tolerance for comparison. Default 1e-5.
        atol: Absolute tolerance for comparison. Default 1e-8.
        level: Validation level - "debug" (full), "lite" (NaN/Inf only), or "off".
            Can also be set globally via AGENTBIBLE_VALIDATION_LEVEL env var.
            WARNING: "off" disables ALL validation - use only for benchmarking.

    Returns:
        Decorated function that validates its return value.

    Raises:
        HermiticityError: If the returned matrix is not Hermitian (level="debug").
        NonFiniteError: If the matrix contains NaN/Inf (level="debug" or "lite").

    Example:
        >>> import numpy as np
        >>> @validate_hermitian
        ... def pauli_z():
        ...     return np.array([[1, 0], [0, -1]])
        >>> Z = pauli_z()  # Passes validation
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)

            # Determine effective validation level
            effective_level = get_validation_level(level)

            # If validation is off, skip everything (with warning)
            if effective_level == ValidationLevel.OFF:
                maybe_warn_validation_off(fn.__name__)
                return result

            np = get_numpy()

            # Convert to numpy array if needed
            arr = np.asarray(result)

            # Check for NaN/Inf FIRST (both debug and lite modes)
            _check_finite(arr, fn.__name__)

            # If lite mode, stop here (no physics checks)
            if effective_level == ValidationLevel.LITE:
                return result

            # Full validation (debug mode)
            # Check shape
            if not is_square_matrix(arr):
                raise HermiticityError(
                    "Matrix is not Hermitian",
                    expected="Square matrix",
                    got=f"Shape {arr.shape}",
                    function_name=fn.__name__,
                    tolerance={"rtol": rtol, "atol": atol},
                    shape=arr.shape,
                )

            # Check Hermiticity: H = H†
            conjugate_transpose = arr.conj().T
            max_deviation = float(np.max(np.abs(arr - conjugate_transpose)))

            if not np.allclose(arr, conjugate_transpose, rtol=rtol, atol=atol):
                raise HermiticityError(
                    "Matrix is not Hermitian",
                    expected="H = H† (matrix equals its conjugate transpose)",
                    got=f"max|H - H†| = {max_deviation:.2e}",
                    function_name=fn.__name__,
                    tolerance={"rtol": rtol, "atol": atol},
                    shape=arr.shape,
                )

            return result

        return wrapper  # type: ignore[return-value]

    # Handle both @validate_hermitian and @validate_hermitian()
    if func is not None:
        return decorator(func)
    return decorator


@overload
def validate_density_matrix(func: F) -> F: ...


@overload
def validate_density_matrix(
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_density_matrix(
    func: F | None = None,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    """Validate that a function returns a valid density matrix.

    A density matrix rho must satisfy:
    1. Hermitian: rho = rho†
    2. Unit trace: tr(rho) = 1
    3. Positive semi-definite: all eigenvalues >= 0

    Can be used with or without arguments:
        @validate_density_matrix
        def make_state(): ...

        @validate_density_matrix(rtol=1e-6)
        def make_precise_state(): ...

        @validate_density_matrix(level="lite")  # Only NaN/Inf check
        def make_state_fast(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        rtol: Relative tolerance for comparison. Default 1e-5.
        atol: Absolute tolerance for comparison. Default 1e-8.
        level: Validation level - "debug" (full), "lite" (NaN/Inf only), or "off".
            Can also be set globally via AGENTBIBLE_VALIDATION_LEVEL env var.
            WARNING: "off" disables ALL validation - use only for benchmarking.

    Returns:
        Decorated function that validates its return value.

    Raises:
        DensityMatrixError: If the returned matrix is not a valid density matrix.
        TraceError: If the trace is not 1.
        PositivityError: If any eigenvalue is negative.
        NonFiniteError: If the matrix contains NaN/Inf.

    Example:
        >>> import numpy as np
        >>> @validate_density_matrix
        ... def ground_state():
        ...     return np.array([[1, 0], [0, 0]])  # |0><0|
        >>> rho = ground_state()  # Passes validation
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)

            # Determine effective validation level
            effective_level = get_validation_level(level)

            # If validation is off, skip everything (with warning)
            if effective_level == ValidationLevel.OFF:
                maybe_warn_validation_off(fn.__name__)
                return result

            np = get_numpy()

            # Convert to numpy array if needed
            arr = np.asarray(result)

            # Check for NaN/Inf FIRST (both debug and lite modes)
            _check_finite(arr, fn.__name__)

            # If lite mode, stop here (no physics checks)
            if effective_level == ValidationLevel.LITE:
                return result

            # Full validation (debug mode)
            # Check shape
            if not is_square_matrix(arr):
                raise DensityMatrixError(
                    "Matrix is not a valid density matrix",
                    expected="Square matrix",
                    got=f"Shape {arr.shape}",
                    function_name=fn.__name__,
                    tolerance={"rtol": rtol, "atol": atol},
                    shape=arr.shape,
                )

            # Check Hermiticity: rho = rho†
            conjugate_transpose = arr.conj().T
            if not np.allclose(arr, conjugate_transpose, rtol=rtol, atol=atol):
                max_deviation = float(np.max(np.abs(arr - conjugate_transpose)))
                raise DensityMatrixError(
                    "Density matrix is not Hermitian",
                    expected="rho = rho† (matrix equals its conjugate transpose)",
                    got=f"max|rho - rho†| = {max_deviation:.2e}",
                    function_name=fn.__name__,
                    tolerance={"rtol": rtol, "atol": atol},
                    shape=arr.shape,
                    guidance=(
                        "Density matrices must be self-adjoint. Check that:\n"
                        "    - Off-diagonal elements are complex conjugates of each other\n"
                        "    - Diagonal elements are real"
                    ),
                )

            # Check trace: tr(rho) = 1
            trace = complex(np.trace(arr))
            if not np.isclose(trace, 1.0, rtol=rtol, atol=atol):
                raise TraceError(
                    "Density matrix does not have unit trace",
                    expected="tr(rho) = 1 (probabilities must sum to 1)",
                    got=f"tr(rho) = {trace.real:.6f}",
                    function_name=fn.__name__,
                    tolerance={"rtol": rtol, "atol": atol},
                    shape=arr.shape,
                )

            # Check positive semi-definite: all eigenvalues >= 0
            eigenvalues = np.linalg.eigvalsh(arr)
            min_eigenvalue = float(np.min(eigenvalues))
            # Allow small negative eigenvalues within tolerance
            if min_eigenvalue < -atol:
                raise PositivityError(
                    "Density matrix is not positive semi-definite",
                    expected="All eigenvalues >= 0 (physical states have non-negative probabilities)",
                    got=f"min(eigenvalue) = {min_eigenvalue:.2e}",
                    function_name=fn.__name__,
                    tolerance={"rtol": rtol, "atol": atol},
                    shape=arr.shape,
                )

            return result

        return wrapper  # type: ignore[return-value]

    # Handle both @validate_density_matrix and @validate_density_matrix()
    if func is not None:
        return decorator(func)
    return decorator


__all__ = [
    "validate_unitary",
    "validate_hermitian",
    "validate_density_matrix",
]
