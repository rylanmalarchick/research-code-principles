"""Probability validation decorators.

Provides decorators for validating probability-related properties:
- Single probability: value in [0, 1]
- Probability array: all values in [0, 1]
- Normalization: values sum to 1
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

from agentbible.validators.base import ValidationError, get_numpy

F = TypeVar("F", bound=Callable[..., Any])


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

            # Check bounds with tolerance
            if value < -atol or value > 1.0 + atol:
                raise ValidationError(
                    "Value is not a valid probability",
                    expected="0 ≤ p ≤ 1",
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
            min_val = np.min(arr)
            max_val = np.max(arr)

            if min_val < -atol:
                raise ValidationError(
                    "Array contains values below 0",
                    expected="All values in [0, 1]",
                    got=f"min = {min_val}",
                    function_name=fn.__name__,
                    tolerance={"atol": atol},
                    shape=arr.shape,
                )

            if max_val > 1.0 + atol:
                raise ValidationError(
                    "Array contains values above 1",
                    expected="All values in [0, 1]",
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
            total = np.sum(arr, axis=axis)

            if axis is None:
                # Total sum should be 1
                if not np.isclose(total, 1.0, rtol=rtol, atol=atol):
                    raise ValidationError(
                        "Array is not normalized",
                        expected="sum = 1",
                        got=f"sum = {total}",
                        function_name=fn.__name__,
                        tolerance={"rtol": rtol, "atol": atol},
                        shape=arr.shape,
                    )
            else:
                # Each slice along axis should sum to 1
                expected = np.ones_like(total)
                if not np.allclose(total, expected, rtol=rtol, atol=atol):
                    max_deviation = np.max(np.abs(total - expected))
                    raise ValidationError(
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
