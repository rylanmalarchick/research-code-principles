"""Probability validation decorators.

Provides decorators for validating probability-related properties:
- Single probability: value in [0, 1]
- Probability array: all values in [0, 1]
- Normalization: values sum to 1

All validators include:
- Numerical sanity checks (NaN/Inf detection before probability checks)
- Support for rtol and atol tolerances
- Educational error messages with guidance
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

from agentbible.errors import (
    NonFiniteError,
    NormalizationError,
    ProbabilityBoundsError,
)
from agentbible.validators.base import ValidationError, get_numpy

F = TypeVar("F", bound=Callable[..., Any])


def _check_finite_scalar(value: float, function_name: str) -> None:
    """Check scalar value for NaN/Inf."""
    import math

    if math.isnan(value) or math.isinf(value):
        detail = "NaN" if math.isnan(value) else "Inf"
        raise NonFiniteError(
            "Value is not finite",
            expected="Finite value (no NaN or Inf)",
            got=detail,
            function_name=function_name,
        )


def _check_finite_array(arr: Any, function_name: str) -> None:
    """Check array for NaN/Inf values before probability validation."""
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
            "Array contains non-finite values",
            expected="All finite values (no NaN or Inf)",
            got=", ".join(details),
            function_name=function_name,
            shape=arr.shape if hasattr(arr, "shape") else None,
        )


@overload
def validate_probability(func: F) -> F: ...


@overload
def validate_probability(
    *,
    atol: float = 1e-10,
) -> Callable[[F], F]: ...


def validate_probability(
    func: F | None = None,
    *,
    atol: float = 1e-10,
) -> F | Callable[[F], F]:
    """Validate that a function returns a probability value in [0, 1].

    Can be used with or without arguments:
        @validate_probability
        def compute_prob(): ...

        @validate_probability(atol=1e-8)
        def compute_precise_prob(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        atol: Absolute tolerance for boundary checks. Default 1e-10.

    Returns:
        Decorated function that validates its return value.

    Raises:
        ValidationError: If the returned value is not in [0, 1].

    Example:
        >>> @validate_probability
        ... def coin_flip_prob():
        ...     return 0.5
        >>> p = coin_flip_prob()  # Passes validation
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)

            # Convert to float for comparison
            value = float(result)

            # Check for NaN/Inf FIRST
            _check_finite_scalar(value, fn.__name__)

            # Check bounds with tolerance
            if value < -atol or value > 1.0 + atol:
                raise ProbabilityBoundsError(
                    "Value is not a valid probability",
                    expected="0 ≤ p ≤ 1 (probability must be in unit interval)",
                    got=f"p = {value}",
                    function_name=fn.__name__,
                    tolerance={"atol": atol},
                )

            return result

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


@overload
def validate_probabilities(func: F) -> F: ...


@overload
def validate_probabilities(
    *,
    atol: float = 1e-10,
) -> Callable[[F], F]: ...


def validate_probabilities(
    func: F | None = None,
    *,
    atol: float = 1e-10,
) -> F | Callable[[F], F]:
    """Validate that a function returns an array of probabilities.

    All values in the returned array must be in [0, 1].

    Can be used with or without arguments:
        @validate_probabilities
        def get_probs(): ...

        @validate_probabilities(atol=1e-8)
        def get_precise_probs(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        atol: Absolute tolerance for boundary checks. Default 1e-10.

    Returns:
        Decorated function that validates its return value.

    Raises:
        ValidationError: If any value is not in [0, 1].

    Example:
        >>> import numpy as np
        >>> @validate_probabilities
        ... def measurement_probs():
        ...     return np.array([0.25, 0.25, 0.25, 0.25])
        >>> probs = measurement_probs()  # Passes validation
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            np = get_numpy()

            arr = np.asarray(result)

            # Check for NaN/Inf FIRST
            _check_finite_array(arr, fn.__name__)

            min_val = float(np.min(arr))
            max_val = float(np.max(arr))

            if min_val < -atol:
                raise ProbabilityBoundsError(
                    "Array contains values below 0",
                    expected="All values in [0, 1] (probabilities must be non-negative)",
                    got=f"min = {min_val}",
                    function_name=fn.__name__,
                    tolerance={"atol": atol},
                    shape=arr.shape,
                )

            if max_val > 1.0 + atol:
                raise ProbabilityBoundsError(
                    "Array contains values above 1",
                    expected="All values in [0, 1] (probabilities cannot exceed 1)",
                    got=f"max = {max_val}",
                    function_name=fn.__name__,
                    tolerance={"atol": atol},
                    shape=arr.shape,
                )

            return result

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


@overload
def validate_normalized(func: F) -> F: ...


@overload
def validate_normalized(
    *,
    axis: int | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Callable[[F], F]: ...


def validate_normalized(
    func: F | None = None,
    *,
    axis: int | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> F | Callable[[F], F]:
    """Validate that a function returns a normalized array (sums to 1).

    Can be used with or without arguments:
        @validate_normalized
        def get_distribution(): ...

        @validate_normalized(axis=-1)
        def get_row_distributions(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        axis: Axis along which to check normalization. If None, checks total sum.
        rtol: Relative tolerance for comparison. Default 1e-5.
        atol: Absolute tolerance for comparison. Default 1e-8.

    Returns:
        Decorated function that validates its return value.

    Raises:
        ValidationError: If the array does not sum to 1.

    Example:
        >>> import numpy as np
        >>> @validate_normalized
        ... def probability_distribution():
        ...     return np.array([0.1, 0.2, 0.3, 0.4])
        >>> dist = probability_distribution()  # Passes validation
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            np = get_numpy()

            arr = np.asarray(result)

            # Check for NaN/Inf FIRST
            _check_finite_array(arr, fn.__name__)

            total = np.sum(arr, axis=axis)

            if axis is None:
                # Total sum should be 1
                total_float = float(total)
                if not np.isclose(total_float, 1.0, rtol=rtol, atol=atol):
                    raise NormalizationError(
                        "Array is not normalized",
                        expected="sum = 1 (probability distribution must sum to 1)",
                        got=f"sum = {total_float}",
                        function_name=fn.__name__,
                        tolerance={"rtol": rtol, "atol": atol},
                        shape=arr.shape,
                    )
            else:
                # Each slice along axis should sum to 1
                expected_ones = np.ones_like(total)
                if not np.allclose(total, expected_ones, rtol=rtol, atol=atol):
                    max_deviation = float(np.max(np.abs(total - expected_ones)))
                    raise NormalizationError(
                        f"Array is not normalized along axis {axis}",
                        expected="sum along axis = 1",
                        got=f"max|sum - 1| = {max_deviation}",
                        function_name=fn.__name__,
                        tolerance={"rtol": rtol, "atol": atol},
                        shape=arr.shape,
                    )

            return result

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator
