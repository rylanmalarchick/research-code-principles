"""General bounds validation decorators.

Provides decorators for validating numerical bounds:
- Positive: value > 0
- Non-negative: value >= 0
- Range: value in [min, max]
- Finite: no NaN or Inf values
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

from agentbible.validators.base import ValidationError, get_numpy

F = TypeVar("F", bound=Callable[..., Any])


@overload
def validate_positive(func: F) -> F: ...


@overload
def validate_positive(
    *,
    atol: float = 0.0,
) -> Callable[[F], F]: ...


def validate_positive(
    func: F | None = None,
    *,
    atol: float = 0.0,
) -> F | Callable[[F], F]:
    """Validate that a function returns positive value(s).

    For arrays, all elements must be positive (> 0).

    Can be used with or without arguments:
        @validate_positive
        def compute_energy(): ...

        @validate_positive(atol=1e-10)
        def compute_near_positive(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        atol: Absolute tolerance. Values > -atol are considered valid.

    Returns:
        Decorated function that validates its return value.

    Raises:
        ValidationError: If any value is not positive.

    Example:
        >>> @validate_positive
        ... def compute_kinetic_energy(mass, velocity):
        ...     return 0.5 * mass * velocity**2
        >>> E = compute_kinetic_energy(1.0, 2.0)  # Passes validation
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            np = get_numpy()

            arr = np.asarray(result)
            min_val = np.min(arr)

            if min_val <= -atol:
                shape = arr.shape if arr.ndim > 0 else None
                raise ValidationError(
                    "Value is not positive",
                    expected="> 0",
                    got=f"min = {min_val}",
                    function_name=fn.__name__,
                    tolerance={"atol": atol} if atol > 0 else None,
                    shape=shape,
                )

            return result

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


@overload
def validate_non_negative(func: F) -> F: ...


@overload
def validate_non_negative(
    *,
    atol: float = 0.0,
) -> Callable[[F], F]: ...


def validate_non_negative(
    func: F | None = None,
    *,
    atol: float = 0.0,
) -> F | Callable[[F], F]:
    """Validate that a function returns non-negative value(s).

    For arrays, all elements must be >= 0.

    Can be used with or without arguments:
        @validate_non_negative
        def get_count(): ...

        @validate_non_negative(atol=1e-10)
        def get_near_zero(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        atol: Absolute tolerance. Values >= -atol are considered valid.

    Returns:
        Decorated function that validates its return value.

    Raises:
        ValidationError: If any value is negative.

    Example:
        >>> @validate_non_negative
        ... def particle_count():
        ...     return 42
        >>> n = particle_count()  # Passes validation
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            np = get_numpy()

            arr = np.asarray(result)
            min_val = np.min(arr)

            if min_val < -atol:
                shape = arr.shape if arr.ndim > 0 else None
                raise ValidationError(
                    "Value is negative",
                    expected=">= 0",
                    got=f"min = {min_val}",
                    function_name=fn.__name__,
                    tolerance={"atol": atol} if atol > 0 else None,
                    shape=shape,
                )

            return result

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


def validate_range(
    min_val: float | None = None,
    max_val: float | None = None,
    *,
    inclusive: bool = True,
    atol: float = 0.0,
) -> Callable[[F], F]:
    """Validate that a function returns value(s) within a specified range.

    For arrays, all elements must be within the range.

    Args:
        min_val: Minimum allowed value. None means no lower bound.
        max_val: Maximum allowed value. None means no upper bound.
        inclusive: Whether bounds are inclusive. Default True.
        atol: Absolute tolerance for boundary checks.

    Returns:
        Decorator that validates function return values.

    Raises:
        ValidationError: If any value is outside the range.

    Example:
        >>> @validate_range(0.0, 1.0)
        ... def compute_fidelity():
        ...     return 0.95
        >>> f = compute_fidelity()  # Passes validation
    """
    if min_val is None and max_val is None:
        msg = "At least one of min_val or max_val must be specified"
        raise ValueError(msg)

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            np = get_numpy()

            arr = np.asarray(result)
            actual_min = np.min(arr)
            actual_max = np.max(arr)

            # Build expected string
            if min_val is not None and max_val is not None:
                if inclusive:
                    expected = f"{min_val} ≤ x ≤ {max_val}"
                else:
                    expected = f"{min_val} < x < {max_val}"
            elif min_val is not None:
                expected = f"x {'≥' if inclusive else '>'} {min_val}"
            else:
                expected = f"x {'≤' if inclusive else '<'} {max_val}"

            # Check minimum
            if min_val is not None:
                if inclusive:
                    if actual_min < min_val - atol:
                        raise ValidationError(
                            "Value below minimum",
                            expected=expected,
                            got=f"min = {actual_min}",
                            function_name=fn.__name__,
                            tolerance={"atol": atol} if atol > 0 else None,
                            shape=arr.shape if arr.ndim > 0 else None,
                        )
                else:
                    if actual_min <= min_val + atol:
                        raise ValidationError(
                            "Value at or below minimum",
                            expected=expected,
                            got=f"min = {actual_min}",
                            function_name=fn.__name__,
                            tolerance={"atol": atol} if atol > 0 else None,
                            shape=arr.shape if arr.ndim > 0 else None,
                        )

            # Check maximum
            if max_val is not None:
                if inclusive:
                    if actual_max > max_val + atol:
                        raise ValidationError(
                            "Value above maximum",
                            expected=expected,
                            got=f"max = {actual_max}",
                            function_name=fn.__name__,
                            tolerance={"atol": atol} if atol > 0 else None,
                            shape=arr.shape if arr.ndim > 0 else None,
                        )
                else:
                    if actual_max >= max_val - atol:
                        raise ValidationError(
                            "Value at or above maximum",
                            expected=expected,
                            got=f"max = {actual_max}",
                            function_name=fn.__name__,
                            tolerance={"atol": atol} if atol > 0 else None,
                            shape=arr.shape if arr.ndim > 0 else None,
                        )

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


@overload
def validate_finite(func: F) -> F: ...


@overload
def validate_finite() -> Callable[[F], F]: ...


def validate_finite(
    func: F | None = None,
) -> F | Callable[[F], F]:
    """Validate that a function returns finite values (no NaN or Inf).

    For arrays, all elements must be finite.

    Can be used with or without parentheses:
        @validate_finite
        def compute(): ...

        @validate_finite()
        def compute(): ...

    Args:
        func: The function to decorate (when used without parentheses).

    Returns:
        Decorated function that validates its return value.

    Raises:
        ValidationError: If any value is NaN or Inf.

    Example:
        >>> import numpy as np
        >>> @validate_finite
        ... def safe_divide(a, b):
        ...     return a / b
        >>> result = safe_divide(1.0, 2.0)  # Passes validation
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            np = get_numpy()

            arr = np.asarray(result)

            if not np.all(np.isfinite(arr)):
                nan_count = np.sum(np.isnan(arr))
                inf_count = np.sum(np.isinf(arr))
                details = []
                if nan_count > 0:
                    details.append(f"{nan_count} NaN")
                if inf_count > 0:
                    details.append(f"{inf_count} Inf")

                raise ValidationError(
                    "Array contains non-finite values",
                    expected="All finite values",
                    got=", ".join(details),
                    function_name=fn.__name__,
                    shape=arr.shape if arr.ndim > 0 else None,
                )

            return result

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator
