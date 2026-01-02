"""Quantum physics validation decorators.

Provides decorators for validating quantum mechanical properties:
- Unitarity: U†U = I
- Hermiticity: H = H†
- Density matrix: Hermitian, trace 1, positive semi-definite

All validators include:
- Numerical sanity checks (NaN/Inf detection before physics checks)
- Support for both rtol and atol tolerances
- Educational error messages with academic references
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

from agentbible.errors import (
    DensityMatrixError,
    HermiticityError,
    NonFiniteError,
    PositivityError,
    TraceError,
    UnitarityError,
)
from agentbible.validators.base import (
    ValidationError,
    get_numpy,
    is_square_matrix,
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
) -> Callable[[F], F]: ...


def validate_unitary(
    func: F | None = None,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> F | Callable[[F], F]:
    """Validate that a function returns a unitary matrix.

    A matrix U is unitary if U†U = I (conjugate transpose times self equals identity).

    Can be used with or without arguments:
        @validate_unitary
        def make_gate(): ...

        @validate_unitary(rtol=1e-6)
        def make_precise_gate(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        rtol: Relative tolerance for comparison. Default 1e-5.
        atol: Absolute tolerance for comparison. Default 1e-8.

    Returns:
        Decorated function that validates its return value.

    Raises:
        ValidationError: If the returned matrix is not unitary.

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
            np = get_numpy()

            # Convert to numpy array if needed
            arr = np.asarray(result)

            # Check for NaN/Inf FIRST (before physics checks)
            _check_finite(arr, fn.__name__)

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
) -> Callable[[F], F]: ...


def validate_hermitian(
    func: F | None = None,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> F | Callable[[F], F]:
    """Validate that a function returns a Hermitian matrix.

    A matrix H is Hermitian if H = H† (equals its conjugate transpose).

    Can be used with or without arguments:
        @validate_hermitian
        def make_hamiltonian(): ...

        @validate_hermitian(rtol=1e-6)
        def make_precise_hamiltonian(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        rtol: Relative tolerance for comparison. Default 1e-5.
        atol: Absolute tolerance for comparison. Default 1e-8.

    Returns:
        Decorated function that validates its return value.

    Raises:
        ValidationError: If the returned matrix is not Hermitian.

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
            np = get_numpy()

            # Convert to numpy array if needed
            arr = np.asarray(result)

            # Check for NaN/Inf FIRST (before physics checks)
            _check_finite(arr, fn.__name__)

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
) -> Callable[[F], F]: ...


def validate_density_matrix(
    func: F | None = None,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> F | Callable[[F], F]:
    """Validate that a function returns a valid density matrix.

    A density matrix ρ must satisfy:
    1. Hermitian: ρ = ρ†
    2. Unit trace: tr(ρ) = 1
    3. Positive semi-definite: all eigenvalues ≥ 0

    Can be used with or without arguments:
        @validate_density_matrix
        def make_state(): ...

        @validate_density_matrix(rtol=1e-6)
        def make_precise_state(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        rtol: Relative tolerance for comparison. Default 1e-5.
        atol: Absolute tolerance for comparison. Default 1e-8.

    Returns:
        Decorated function that validates its return value.

    Raises:
        ValidationError: If the returned matrix is not a valid density matrix.

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
            np = get_numpy()

            # Convert to numpy array if needed
            arr = np.asarray(result)

            # Check for NaN/Inf FIRST (before physics checks)
            _check_finite(arr, fn.__name__)

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

            # Check Hermiticity: ρ = ρ†
            conjugate_transpose = arr.conj().T
            if not np.allclose(arr, conjugate_transpose, rtol=rtol, atol=atol):
                max_deviation = float(np.max(np.abs(arr - conjugate_transpose)))
                raise DensityMatrixError(
                    "Density matrix is not Hermitian",
                    expected="ρ = ρ† (matrix equals its conjugate transpose)",
                    got=f"max|ρ - ρ†| = {max_deviation:.2e}",
                    function_name=fn.__name__,
                    tolerance={"rtol": rtol, "atol": atol},
                    shape=arr.shape,
                    guidance=(
                        "Density matrices must be self-adjoint. Check that:\n"
                        "    - Off-diagonal elements are complex conjugates of each other\n"
                        "    - Diagonal elements are real"
                    ),
                )

            # Check trace: tr(ρ) = 1
            trace = complex(np.trace(arr))
            if not np.isclose(trace, 1.0, rtol=rtol, atol=atol):
                raise TraceError(
                    "Density matrix does not have unit trace",
                    expected="tr(ρ) = 1 (probabilities must sum to 1)",
                    got=f"tr(ρ) = {trace.real:.6f}",
                    function_name=fn.__name__,
                    tolerance={"rtol": rtol, "atol": atol},
                    shape=arr.shape,
                )

            # Check positive semi-definite: all eigenvalues ≥ 0
            eigenvalues = np.linalg.eigvalsh(arr)
            min_eigenvalue = float(np.min(eigenvalues))
            # Allow small negative eigenvalues within tolerance
            if min_eigenvalue < -atol:
                raise PositivityError(
                    "Density matrix is not positive semi-definite",
                    expected="All eigenvalues ≥ 0 (physical states have non-negative probabilities)",
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
