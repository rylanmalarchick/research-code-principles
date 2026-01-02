"""Probability validation decorators.

Provides decorators for validating probability-related properties:
- Single probability: value in [0, 1]
- Probability array: all values in [0, 1]
- Normalization: values sum to 1

All validators include:
- Numerical sanity checks (NaN/Inf detection before probability checks)
- Support for rtol and atol tolerances
- Educational error messages with guidance
- Conditional validation levels (debug/lite/off)

Validation levels:
- "debug" (default): Full validation with all probability checks
- "lite": Only NaN/Inf sanity checks (fast)
- "off": Skip validation entirely (DANGEROUS - for benchmarking only)
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

from agentbible.errors import (
    NonFiniteError,
    NormalizationError,
    ProbabilityBoundsError,
)
from agentbible.validators.base import (
    ValidationLevel,
    get_numpy,
    get_validation_level,
    maybe_warn_validation_off,
)

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
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_probability(
    func: F | None = None,
    *,
    atol: float = 1e-10,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    """Validate that a function returns a probability value in [0, 1].

    Can be used with or without arguments:
        @validate_probability
        def compute_prob(): ...

        @validate_probability(atol=1e-8)
        def compute_precise_prob(): ...

        @validate_probability(level="lite")  # Only check for NaN/Inf
        def compute_prob_fast(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        atol: Absolute tolerance for boundary checks. Default 1e-10.
        level: Validation level - "debug" (full), "lite" (NaN/Inf only), or "off".
            Can also be set globally via AGENTBIBLE_VALIDATION_LEVEL env var.
            WARNING: "off" disables ALL validation - use only for benchmarking.

    Returns:
        Decorated function that validates its return value.

    Raises:
        ProbabilityBoundsError: If the returned value is not in [0, 1] (level="debug").
        NonFiniteError: If the value is NaN/Inf (level="debug" or "lite").

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

            # Determine effective validation level
            effective_level = get_validation_level(level)

            # If validation is off, skip everything (with warning)
            if effective_level == ValidationLevel.OFF:
                maybe_warn_validation_off(fn.__name__)
                return result

            # Convert to float for comparison
            value = float(result)

            # Check for NaN/Inf FIRST (both debug and lite modes)
            _check_finite_scalar(value, fn.__name__)

            # If lite mode, stop here (no bounds checks)
            if effective_level == ValidationLevel.LITE:
                return result

            # Full validation (debug mode)
            # Check bounds with tolerance
            if value < -atol or value > 1.0 + atol:
                raise ProbabilityBoundsError(
                    "Value is not a valid probability",
                    expected="0 <= p <= 1 (probability must be in unit interval)",
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
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_probabilities(
    func: F | None = None,
    *,
    atol: float = 1e-10,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    """Validate that a function returns an array of probabilities.

    All values in the returned array must be in [0, 1].

    Can be used with or without arguments:
        @validate_probabilities
        def get_probs(): ...

        @validate_probabilities(atol=1e-8)
        def get_precise_probs(): ...

        @validate_probabilities(level="off")  # DANGEROUS - for benchmarking only
        def get_probs_fast(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        atol: Absolute tolerance for boundary checks. Default 1e-10.
        level: Validation level - "debug" (full), "lite" (NaN/Inf only), or "off".
            Can also be set globally via AGENTBIBLE_VALIDATION_LEVEL env var.
            WARNING: "off" disables ALL validation - use only for benchmarking.

    Returns:
        Decorated function that validates its return value.

    Raises:
        ProbabilityBoundsError: If any value is not in [0, 1] (level="debug").
        NonFiniteError: If any value is NaN/Inf (level="debug" or "lite").

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

            # Determine effective validation level
            effective_level = get_validation_level(level)

            # If validation is off, skip everything (with warning)
            if effective_level == ValidationLevel.OFF:
                maybe_warn_validation_off(fn.__name__)
                return result

            np = get_numpy()
            arr = np.asarray(result)

            # Check for NaN/Inf FIRST (both debug and lite modes)
            _check_finite_array(arr, fn.__name__)

            # If lite mode, stop here (no bounds checks)
            if effective_level == ValidationLevel.LITE:
                return result

            # Full validation (debug mode)
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
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_normalized(
    func: F | None = None,
    *,
    axis: int | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    """Validate that a function returns a normalized array (sums to 1).

    Can be used with or without arguments:
        @validate_normalized
        def get_distribution(): ...

        @validate_normalized(axis=-1)
        def get_row_distributions(): ...

        @validate_normalized(level="lite")  # Only NaN/Inf check
        def get_distribution_fast(): ...

    Args:
        func: The function to decorate (when used without parentheses).
        axis: Axis along which to check normalization. If None, checks total sum.
        rtol: Relative tolerance for comparison. Default 1e-5.
        atol: Absolute tolerance for comparison. Default 1e-8.
        level: Validation level - "debug" (full), "lite" (NaN/Inf only), or "off".
            Can also be set globally via AGENTBIBLE_VALIDATION_LEVEL env var.
            WARNING: "off" disables ALL validation - use only for benchmarking.

    Returns:
        Decorated function that validates its return value.

    Raises:
        NormalizationError: If the array does not sum to 1 (level="debug").
        NonFiniteError: If any value is NaN/Inf (level="debug" or "lite").

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

            # Determine effective validation level
            effective_level = get_validation_level(level)

            # If validation is off, skip everything (with warning)
            if effective_level == ValidationLevel.OFF:
                maybe_warn_validation_off(fn.__name__)
                return result

            np = get_numpy()
            arr = np.asarray(result)

            # Check for NaN/Inf FIRST (both debug and lite modes)
            _check_finite_array(arr, fn.__name__)

            # If lite mode, stop here (no normalization checks)
            if effective_level == ValidationLevel.LITE:
                return result

            # Full validation (debug mode)
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
