"""Base utilities for validation decorators.

Provides the ValidationError exception, ValidationLevel enum, and shared utilities
for building physics validation decorators.

The validation level can be controlled via:
1. Decorator parameter: @validate_unitary(level="lite")
2. Environment variable: AGENTBIBLE_VALIDATION_LEVEL=off

Levels:
- "debug" (default): Full validation with all physics checks
- "lite": Only NaN/Inf sanity checks (fast, catches numerical instability)
- "off": Skip all validation (DANGEROUS - use only for benchmarking)

WARNING: level="off" completely disables validation. This is useful for
performance benchmarking or tight optimization loops, but invalid physics
will NOT be caught. Use with extreme caution.
"""

from __future__ import annotations

import functools
import os
import warnings
from enum import Enum
from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

# Import ValidationError from errors module (canonical location)
from agentbible.errors import ValidationError

F = TypeVar("F", bound=Callable[..., Any])

# Environment variable for global validation level override
ENV_VALIDATION_LEVEL = "AGENTBIBLE_VALIDATION_LEVEL"


class ValidationLevel(Enum):
    """Validation strictness levels.

    Attributes:
        DEBUG: Full validation with all physics checks (default).
            Use during development and testing.
        LITE: Only NaN/Inf sanity checks. Fast but catches numerical instability.
            Use when you trust the physics but want to catch NaN/Inf.
        OFF: Skip all validation completely.
            DANGEROUS: Invalid physics will NOT be caught.
            Use ONLY for performance benchmarking or profiling.
    """

    DEBUG = "debug"
    LITE = "lite"
    OFF = "off"

    @classmethod
    def from_string(cls, value: str) -> ValidationLevel:
        """Convert string to ValidationLevel.

        Args:
            value: One of "debug", "lite", "off" (case-insensitive).

        Returns:
            The corresponding ValidationLevel.

        Raises:
            ValueError: If value is not a valid level.
        """
        value_lower = value.lower().strip()
        for level in cls:
            if level.value == value_lower:
                return level
        valid = ", ".join(f'"{lvl.value}"' for lvl in cls)
        raise ValueError(
            f"Invalid validation level: {value!r}. Must be one of: {valid}"
        )


def get_validation_level(
    explicit_level: str | ValidationLevel | None = None,
) -> ValidationLevel:
    """Determine the effective validation level.

    Priority order:
    1. Explicit level passed to decorator (highest priority)
    2. AGENTBIBLE_VALIDATION_LEVEL environment variable
    3. Default to DEBUG (full validation)

    Args:
        explicit_level: Level explicitly passed to a decorator, or None.

    Returns:
        The effective ValidationLevel to use.
    """
    # If explicit level provided, use it
    if explicit_level is not None:
        if isinstance(explicit_level, ValidationLevel):
            return explicit_level
        return ValidationLevel.from_string(explicit_level)

    # Check environment variable
    env_level = os.environ.get(ENV_VALIDATION_LEVEL)
    if env_level is not None:
        try:
            return ValidationLevel.from_string(env_level)
        except ValueError:
            warnings.warn(
                f"Invalid {ENV_VALIDATION_LEVEL}={env_level!r}, using 'debug'. "
                f"Valid values: debug, lite, off",
                RuntimeWarning,
                stacklevel=2,
            )

    # Default to full validation
    return ValidationLevel.DEBUG


def warn_validation_off(function_name: str) -> None:
    """Issue a warning when validation is disabled.

    This warning is issued once per function to remind developers
    that validation is off and bugs may go undetected.
    """
    warnings.warn(
        f"Validation is OFF for {function_name}(). "
        f"Physics errors will NOT be caught. "
        f"Set AGENTBIBLE_VALIDATION_LEVEL=debug to re-enable.",
        RuntimeWarning,
        stacklevel=4,
    )


# Track which functions have already warned about being off
_warned_functions: set[str] = set()


def maybe_warn_validation_off(function_name: str) -> None:
    """Warn once per function that validation is off."""
    if function_name not in _warned_functions:
        _warned_functions.add(function_name)
        warn_validation_off(function_name)


def get_numpy() -> Any:
    """Lazy import numpy to avoid startup cost."""
    import numpy

    return numpy


def is_complex_dtype(arr: NDArray[Any]) -> bool:
    """Check if array has complex dtype."""
    return bool(np.issubdtype(arr.dtype, np.complexfloating))


def is_square_matrix(arr: NDArray[Any]) -> bool:
    """Check if array is a square matrix."""
    return bool(arr.ndim == 2 and arr.shape[0] == arr.shape[1])


def make_validator(
    check_fn: Callable[[Any], tuple[bool, str]],
    error_template: str,
    expected: str,
) -> Callable[[F], F]:
    """Create a validation decorator from a check function.

    Args:
        check_fn: Function that takes the return value and returns
            (is_valid, detail_message).
        error_template: Template for error message.
        expected: Description of expected condition.

    Returns:
        A decorator that validates function return values.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            is_valid, detail = check_fn(result)
            if not is_valid:
                raise ValidationError(
                    error_template,
                    expected=expected,
                    got=detail,
                    function_name=func.__name__,
                )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


__all__ = [
    "ValidationError",
    "ValidationLevel",
    "get_validation_level",
    "maybe_warn_validation_off",
    "get_numpy",
    "is_complex_dtype",
    "is_square_matrix",
    "make_validator",
    "ENV_VALIDATION_LEVEL",
]
