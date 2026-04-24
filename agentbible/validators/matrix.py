"""Matrix validation checks and decorators defined by SPEC.md."""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike

from agentbible.errors import (
    DensityMatrixError,
    HermiticityError,
    PositiveDefiniteError,
    PositiveSemidefiniteError,
    SymmetryError,
    TraceError,
    UnitarityError,
    ValidationError,
)
from agentbible.validators.arrays import check_finite
from agentbible.validators.base import (
    ValidationLevel,
    get_validation_level,
    maybe_warn_validation_off,
)

T = TypeVar("T", bound=ArrayLike)
F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12


def _warn_or_raise(error: Exception, strict: bool) -> None:
    if strict:
        raise error
    warnings.warn(str(error), RuntimeWarning, stacklevel=3)


def _shape_of(arr: np.ndarray) -> tuple[int, ...] | None:
    return arr.shape if arr.ndim > 0 else None


def _function_label(check_name: str, name: str) -> str:
    return f"{check_name}(name='{name}')"


def _require_square(
    arr: np.ndarray,
    name: str,
    error_cls: type[ValidationError],
    strict: bool,
    tolerance: dict[str, float] | None = None,
) -> bool:
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return True
    error = error_cls(
        f"Matrix '{name}' is not square",
        expected="Square matrix",
        got=f"shape = {arr.shape}",
        function_name=_function_label(error_cls.__name__, name),
        tolerance=tolerance,
        shape=_shape_of(arr),
    )
    _warn_or_raise(error, strict)
    return False


def check_symmetric(
    arr: T,
    *,
    name: str = "matrix",
    atol: float = DEFAULT_ATOL,
    strict: bool = True,
) -> T:
    """Validate A = A^T using max elementwise error."""
    np_arr = np.asarray(arr)
    check_finite(np_arr, name=name, strict=True)
    if not _require_square(np_arr, name, SymmetryError, strict, {"atol": atol}):
        return arr
    delta = float(np.max(np.abs(np_arr - np_arr.T)))
    if delta > atol:
        error = SymmetryError(
            f"Matrix '{name}' is not symmetric",
            expected="|A[i,j] - A[j,i]| <= atol",
            got=f"max|A - A^T| = {delta:.2e}",
            function_name=_function_label("check_symmetric", name),
            tolerance={"atol": atol},
            shape=_shape_of(np_arr),
        )
        _warn_or_raise(error, strict)
    return arr


def check_hermitian(
    arr: T,
    *,
    name: str = "matrix",
    atol: float = DEFAULT_ATOL,
    strict: bool = True,
) -> T:
    """Validate A = A† using max elementwise error."""
    np_arr = np.asarray(arr)
    check_finite(np_arr, name=name, strict=True)
    if not _require_square(np_arr, name, HermiticityError, strict, {"atol": atol}):
        return arr
    delta = float(np.max(np.abs(np_arr - np_arr.conj().T)))
    if delta > atol:
        error = HermiticityError(
            f"Matrix '{name}' is not Hermitian",
            expected="H = H† and |A[i,j] - conj(A[j,i])| <= atol",
            got=f"max|A - A†| = {delta:.2e}",
            function_name=_function_label("check_hermitian", name),
            tolerance={"atol": atol},
            shape=_shape_of(np_arr),
        )
        _warn_or_raise(error, strict)
    return arr


def check_unitary(
    arr: T,
    *,
    name: str = "matrix",
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    strict: bool = True,
) -> T:
    """Validate U†U = I using Frobenius residual."""
    np_arr = np.asarray(arr)
    check_finite(np_arr, name=name, strict=True)
    if not _require_square(
        np_arr, name, UnitarityError, strict, {"rtol": rtol, "atol": atol}
    ):
        return arr
    identity = np.eye(np_arr.shape[0], dtype=np_arr.dtype)
    residual = np_arr.conj().T @ np_arr - identity
    residual_norm = float(np.linalg.norm(residual, ord="fro"))
    threshold = atol + (rtol * np_arr.shape[0])
    if residual_norm > threshold:
        error = UnitarityError(
            f"Matrix '{name}' is not unitary",
            expected="U†U = I and ||U†U - I||_F <= atol + rtol * n",
            got=f"||U†U - I||_F = {residual_norm:.2e}",
            function_name=_function_label("check_unitary", name),
            tolerance={"rtol": rtol, "atol": atol},
            shape=_shape_of(np_arr),
        )
        _warn_or_raise(error, strict)
    return arr


def check_positive_definite(
    arr: T,
    *,
    name: str = "matrix",
    strict: bool = True,
) -> T:
    """Validate positive definiteness via Cholesky factorization."""
    np_arr = np.asarray(arr)
    check_finite(np_arr, name=name, strict=True)
    if not _require_square(np_arr, name, PositiveDefiniteError, strict):
        return arr
    try:
        np.linalg.cholesky(np_arr)
    except np.linalg.LinAlgError:
        error = PositiveDefiniteError(
            f"Matrix '{name}' is not positive definite",
            expected="Cholesky factorization succeeds",
            got="cholesky failure",
            function_name=_function_label("check_positive_definite", name),
            shape=_shape_of(np_arr),
        )
        _warn_or_raise(error, strict)
    return arr


def check_positive_semidefinite(
    arr: T,
    *,
    name: str = "matrix",
    atol: float = DEFAULT_ATOL,
    strict: bool = True,
) -> T:
    """Validate semidefiniteness via the smallest eigenvalue."""
    np_arr = np.asarray(arr)
    check_finite(np_arr, name=name, strict=True)
    if not _require_square(
        np_arr, name, PositiveSemidefiniteError, strict, {"atol": atol}
    ):
        return arr
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(np_arr).real))
    if min_eigenvalue < -atol:
        error = PositiveSemidefiniteError(
            f"Matrix '{name}' is not positive semi-definite",
            expected="all eigenvalues >= -atol",
            got=f"min eigenvalue = {min_eigenvalue:.2e}",
            function_name=_function_label("check_positive_semidefinite", name),
            tolerance={"atol": atol},
            shape=_shape_of(np_arr),
        )
        _warn_or_raise(error, strict)
    return arr


def check_density_matrix(
    arr: T,
    *,
    name: str = "matrix",
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    strict: bool = True,
) -> T:
    """Validate Hermitian, trace-1, positive-semidefinite structure."""
    np_arr = np.asarray(arr)
    check_finite(np_arr, name=name, strict=True)
    if not _require_square(
        np_arr, name, DensityMatrixError, strict, {"rtol": rtol, "atol": atol}
    ):
        return arr
    check_hermitian(np_arr, name=name, atol=atol, strict=strict)
    trace = complex(np.trace(np_arr))
    if abs(trace - 1.0) > atol:
        error = TraceError(
            f"Matrix '{name}' does not have unit trace",
            expected="|tr(rho) - 1| <= atol",
            got=f"tr(rho) = {trace.real:.6f}",
            function_name=_function_label("check_density_matrix", name),
            tolerance={"atol": atol},
            shape=_shape_of(np_arr),
        )
        _warn_or_raise(error, strict)
    check_positive_semidefinite(np_arr, name=name, atol=atol, strict=strict)
    return arr


def _wrap_matrix_validator(
    func: F,
    check: Callable[..., Any],
    level: str | ValidationLevel | None,
    **check_kwargs: Any,
) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        effective_level = get_validation_level(level)
        if effective_level == ValidationLevel.OFF:
            maybe_warn_validation_off(func.__name__)
            return result
        check_finite(np.asarray(result), name=func.__name__, strict=True)
        if effective_level == ValidationLevel.LITE:
            return result
        check(result, name=func.__name__, strict=True, **check_kwargs)
        return result

    return wrapper  # type: ignore[return-value]


@overload
def validate_symmetric(func: F) -> F: ...


@overload
def validate_symmetric(
    *,
    atol: float = DEFAULT_ATOL,
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_symmetric(
    func: F | None = None,
    *,
    atol: float = DEFAULT_ATOL,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    def decorator(fn: F) -> F:
        return _wrap_matrix_validator(fn, check_symmetric, level, atol=atol)

    return decorator(func) if func is not None else decorator


@overload
def validate_hermitian(func: F) -> F: ...


@overload
def validate_hermitian(
    *,
    atol: float = DEFAULT_ATOL,
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_hermitian(
    func: F | None = None,
    *,
    atol: float = DEFAULT_ATOL,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    def decorator(fn: F) -> F:
        return _wrap_matrix_validator(fn, check_hermitian, level, atol=atol)

    return decorator(func) if func is not None else decorator


@overload
def validate_unitary(func: F) -> F: ...


@overload
def validate_unitary(
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_unitary(
    func: F | None = None,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    def decorator(fn: F) -> F:
        return _wrap_matrix_validator(
            fn, check_unitary, level, rtol=rtol, atol=atol
        )

    return decorator(func) if func is not None else decorator


@overload
def validate_positive_definite(func: F) -> F: ...


@overload
def validate_positive_definite(
    *,
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_positive_definite(
    func: F | None = None,
    *,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    def decorator(fn: F) -> F:
        return _wrap_matrix_validator(fn, check_positive_definite, level)

    return decorator(func) if func is not None else decorator


@overload
def validate_positive_semidefinite(func: F) -> F: ...


@overload
def validate_positive_semidefinite(
    *,
    atol: float = DEFAULT_ATOL,
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_positive_semidefinite(
    func: F | None = None,
    *,
    atol: float = DEFAULT_ATOL,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    def decorator(fn: F) -> F:
        return _wrap_matrix_validator(
            fn, check_positive_semidefinite, level, atol=atol
        )

    return decorator(func) if func is not None else decorator


@overload
def validate_density_matrix(func: F) -> F: ...


@overload
def validate_density_matrix(
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_density_matrix(
    func: F | None = None,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    def decorator(fn: F) -> F:
        return _wrap_matrix_validator(
            fn, check_density_matrix, level, rtol=rtol, atol=atol
        )

    return decorator(func) if func is not None else decorator


__all__ = [
    "DEFAULT_RTOL",
    "DEFAULT_ATOL",
    "check_symmetric",
    "check_hermitian",
    "check_unitary",
    "check_positive_definite",
    "check_positive_semidefinite",
    "check_density_matrix",
    "validate_symmetric",
    "validate_hermitian",
    "validate_unitary",
    "validate_positive_definite",
    "validate_positive_semidefinite",
    "validate_density_matrix",
]
