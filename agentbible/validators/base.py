"""Base utilities for validation decorators.

Provides the ValidationError exception and shared utilities for
building physics validation decorators.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

# Import ValidationError from errors module (canonical location)
from agentbible.errors import ValidationError

F = TypeVar("F", bound=Callable[..., Any])


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
