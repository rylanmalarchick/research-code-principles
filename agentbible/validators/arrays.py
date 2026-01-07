"""Direct array validation functions for feature engineering pipelines.

Unlike decorator-based validators that wrap functions, these are direct
validation functions that operate on arrays. They are designed for use
in data pipelines where intermediate arrays need validation.

All functions:
- Take an array as the first argument
- Take a `name` parameter for clear error messages
- Return the input array unchanged (for chaining)
- Raise appropriate errors when validation fails

Strict Mode:
    By default, these functions raise exceptions on validation failure.
    Set `strict=False` to issue warnings instead (useful for exploratory work).

Example:
    >>> import numpy as np
    >>> from agentbible import check_finite, check_positive, check_range
    >>>
    >>> # Direct validation
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> check_finite(arr, name="temperature")
    >>> check_positive(arr, name="temperature")
    >>>
    >>> # Chaining
    >>> validated = check_positive(check_finite(arr, name="x"), name="x")
    >>>
    >>> # Warn instead of raise (strict=False)
    >>> check_positive(arr, name="x", strict=False)  # Warns if invalid
"""

from __future__ import annotations

import warnings
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from agentbible.errors import (
    BoundsError,
    NonFiniteError,
    NormalizationError,
    ProbabilityBoundsError,
)

# Type variable for array-like inputs
T = TypeVar("T", bound=ArrayLike)


def check_finite(
    arr: T,
    *,
    name: str = "array",
    strict: bool = True,
) -> T:
    """Validate that an array contains only finite values (no NaN or Inf).

    Args:
        arr: Array-like input to validate.
        name: Descriptive name for error messages (e.g., "temperature", "velocity").
        strict: If True (default), raise NonFiniteError on failure.
            If False, issue a warning instead and return the array.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        NonFiniteError: If array contains NaN or Inf values (when strict=True).

    Example:
        >>> import numpy as np
        >>> from agentbible import check_finite
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> check_finite(arr, name="temperature")  # Passes
        >>> check_finite(np.array([1.0, np.nan]), name="bad")  # Raises
    """
    np_arr = np.asarray(arr)

    if not np.all(np.isfinite(np_arr)):
        nan_count = int(np.sum(np.isnan(np_arr)))
        inf_count = int(np.sum(np.isinf(np_arr)))
        details = []
        if nan_count > 0:
            details.append(f"{nan_count} NaN")
        if inf_count > 0:
            details.append(f"{inf_count} Inf")

        error = NonFiniteError(
            f"Array '{name}' contains non-finite values",
            expected="All finite values (no NaN or Inf)",
            got=", ".join(details),
            function_name=f"check_finite(name='{name}')",
            shape=np_arr.shape if np_arr.ndim > 0 else None,
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return arr


def check_positive(
    arr: T,
    *,
    name: str = "array",
    atol: float = 0.0,
    strict: bool = True,
) -> T:
    """Validate that all values in an array are positive (> 0).

    Args:
        arr: Array-like input to validate.
        name: Descriptive name for error messages.
        atol: Absolute tolerance. Values > -atol are considered valid.
        strict: If True (default), raise BoundsError on failure.
            If False, issue a warning instead.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        NonFiniteError: If array contains NaN or Inf values.
        BoundsError: If any value is not positive (when strict=True).

    Example:
        >>> import numpy as np
        >>> from agentbible import check_positive
        >>> check_positive(np.array([1.0, 2.0, 3.0]), name="mass")  # Passes
        >>> check_positive(np.array([0.0, 1.0]), name="mass")  # Raises
    """
    np_arr = np.asarray(arr)

    # Check finite first (always raises - NaN/Inf is never acceptable)
    check_finite(np_arr, name=name, strict=True)

    min_val = float(np.min(np_arr))

    if min_val <= -atol:
        error = BoundsError(
            f"Array '{name}' is not positive",
            expected="> 0 (strictly positive)",
            got=f"min = {min_val}",
            function_name=f"check_positive(name='{name}')",
            tolerance={"atol": atol} if atol > 0 else None,
            shape=np_arr.shape if np_arr.ndim > 0 else None,
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return arr


def check_non_negative(
    arr: T,
    *,
    name: str = "array",
    atol: float = 0.0,
    strict: bool = True,
) -> T:
    """Validate that all values in an array are non-negative (>= 0).

    Args:
        arr: Array-like input to validate.
        name: Descriptive name for error messages.
        atol: Absolute tolerance. Values >= -atol are considered valid.
        strict: If True (default), raise BoundsError on failure.
            If False, issue a warning instead.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        NonFiniteError: If array contains NaN or Inf values.
        BoundsError: If any value is negative (when strict=True).

    Example:
        >>> import numpy as np
        >>> from agentbible import check_non_negative
        >>> check_non_negative(np.array([0.0, 1.0, 2.0]), name="count")  # Passes
        >>> check_non_negative(np.array([-1.0, 0.0]), name="count")  # Raises
    """
    np_arr = np.asarray(arr)

    # Check finite first
    check_finite(np_arr, name=name, strict=True)

    min_val = float(np.min(np_arr))

    if min_val < -atol:
        error = BoundsError(
            f"Array '{name}' contains negative values",
            expected=">= 0 (non-negative)",
            got=f"min = {min_val}",
            function_name=f"check_non_negative(name='{name}')",
            tolerance={"atol": atol} if atol > 0 else None,
            shape=np_arr.shape if np_arr.ndim > 0 else None,
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return arr


def check_range(
    arr: T,
    min_val: float | None = None,
    max_val: float | None = None,
    *,
    name: str = "array",
    inclusive: bool = True,
    atol: float = 0.0,
    strict: bool = True,
) -> T:
    """Validate that all values in an array are within a specified range.

    Args:
        arr: Array-like input to validate.
        min_val: Minimum allowed value. None means no lower bound.
        max_val: Maximum allowed value. None means no upper bound.
        name: Descriptive name for error messages.
        inclusive: Whether bounds are inclusive. Default True.
        atol: Absolute tolerance for boundary checks.
        strict: If True (default), raise BoundsError on failure.
            If False, issue a warning instead.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        ValueError: If neither min_val nor max_val is specified.
        NonFiniteError: If array contains NaN or Inf values.
        BoundsError: If any value is outside the range (when strict=True).

    Example:
        >>> import numpy as np
        >>> from agentbible import check_range
        >>> check_range(np.array([0.5, 0.8]), 0.0, 1.0, name="fidelity")  # Passes
        >>> check_range(np.array([1.5, 0.8]), 0.0, 1.0, name="fidelity")  # Raises
    """
    if min_val is None and max_val is None:
        msg = "At least one of min_val or max_val must be specified"
        raise ValueError(msg)

    np_arr = np.asarray(arr)

    # Check finite first
    check_finite(np_arr, name=name, strict=True)

    actual_min = float(np.min(np_arr))
    actual_max = float(np.max(np_arr))

    # Build expected string
    if min_val is not None and max_val is not None:
        if inclusive:
            expected = f"{min_val} <= x <= {max_val}"
        else:
            expected = f"{min_val} < x < {max_val}"
    elif min_val is not None:
        expected = f"x {'>=' if inclusive else '>'} {min_val}"
    else:
        expected = f"x {'<=' if inclusive else '<'} {max_val}"

    # Check minimum
    if min_val is not None:
        if inclusive:
            if actual_min < min_val - atol:
                error = BoundsError(
                    f"Array '{name}' has value below minimum",
                    expected=expected,
                    got=f"min = {actual_min}",
                    function_name=f"check_range(name='{name}')",
                    tolerance={"atol": atol} if atol > 0 else None,
                    shape=np_arr.shape if np_arr.ndim > 0 else None,
                )
                if strict:
                    raise error
                else:
                    warnings.warn(str(error), RuntimeWarning, stacklevel=2)
                    return arr
        else:
            if actual_min <= min_val + atol:
                error = BoundsError(
                    f"Array '{name}' has value at or below minimum",
                    expected=expected,
                    got=f"min = {actual_min}",
                    function_name=f"check_range(name='{name}')",
                    tolerance={"atol": atol} if atol > 0 else None,
                    shape=np_arr.shape if np_arr.ndim > 0 else None,
                )
                if strict:
                    raise error
                else:
                    warnings.warn(str(error), RuntimeWarning, stacklevel=2)
                    return arr

    # Check maximum
    if max_val is not None:
        if inclusive:
            if actual_max > max_val + atol:
                error = BoundsError(
                    f"Array '{name}' has value above maximum",
                    expected=expected,
                    got=f"max = {actual_max}",
                    function_name=f"check_range(name='{name}')",
                    tolerance={"atol": atol} if atol > 0 else None,
                    shape=np_arr.shape if np_arr.ndim > 0 else None,
                )
                if strict:
                    raise error
                else:
                    warnings.warn(str(error), RuntimeWarning, stacklevel=2)
        else:
            if actual_max >= max_val - atol:
                error = BoundsError(
                    f"Array '{name}' has value at or above maximum",
                    expected=expected,
                    got=f"max = {actual_max}",
                    function_name=f"check_range(name='{name}')",
                    tolerance={"atol": atol} if atol > 0 else None,
                    shape=np_arr.shape if np_arr.ndim > 0 else None,
                )
                if strict:
                    raise error
                else:
                    warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return arr


def check_probability(
    value: float,
    *,
    name: str = "probability",
    atol: float = 1e-10,
    strict: bool = True,
) -> float:
    """Validate that a scalar value is a valid probability in [0, 1].

    Args:
        value: Scalar value to validate.
        name: Descriptive name for error messages.
        atol: Absolute tolerance for boundary checks.
        strict: If True (default), raise ProbabilityBoundsError on failure.
            If False, issue a warning instead.

    Returns:
        The input value unchanged.

    Raises:
        NonFiniteError: If value is NaN or Inf.
        ProbabilityBoundsError: If value is not in [0, 1] (when strict=True).

    Example:
        >>> from agentbible import check_probability
        >>> check_probability(0.5, name="p_success")  # Passes
        >>> check_probability(1.5, name="p_success")  # Raises
    """
    import math

    if math.isnan(value) or math.isinf(value):
        detail = "NaN" if math.isnan(value) else "Inf"
        raise NonFiniteError(
            f"Value '{name}' is not finite",
            expected="Finite value (no NaN or Inf)",
            got=detail,
            function_name=f"check_probability(name='{name}')",
        )

    if value < -atol or value > 1.0 + atol:
        error = ProbabilityBoundsError(
            f"Value '{name}' is not a valid probability",
            expected="0 <= p <= 1 (probability must be in unit interval)",
            got=f"p = {value}",
            function_name=f"check_probability(name='{name}')",
            tolerance={"atol": atol},
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return value


def check_probabilities(
    arr: T,
    *,
    name: str = "probabilities",
    atol: float = 1e-10,
    strict: bool = True,
) -> T:
    """Validate that all values in an array are valid probabilities in [0, 1].

    Args:
        arr: Array-like input to validate.
        name: Descriptive name for error messages.
        atol: Absolute tolerance for boundary checks.
        strict: If True (default), raise ProbabilityBoundsError on failure.
            If False, issue a warning instead.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        NonFiniteError: If array contains NaN or Inf values.
        ProbabilityBoundsError: If any value is not in [0, 1] (when strict=True).

    Example:
        >>> import numpy as np
        >>> from agentbible import check_probabilities
        >>> check_probabilities(np.array([0.1, 0.5, 0.9]), name="probs")  # Passes
        >>> check_probabilities(np.array([0.1, 1.5, 0.9]), name="probs")  # Raises
    """
    np_arr = np.asarray(arr)

    # Check finite first
    check_finite(np_arr, name=name, strict=True)

    min_val = float(np.min(np_arr))
    max_val = float(np.max(np_arr))

    if min_val < -atol:
        error = ProbabilityBoundsError(
            f"Array '{name}' contains values below 0",
            expected="All values in [0, 1] (probabilities must be non-negative)",
            got=f"min = {min_val}",
            function_name=f"check_probabilities(name='{name}')",
            tolerance={"atol": atol},
            shape=np_arr.shape,
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
            return arr

    if max_val > 1.0 + atol:
        error = ProbabilityBoundsError(
            f"Array '{name}' contains values above 1",
            expected="All values in [0, 1] (probabilities cannot exceed 1)",
            got=f"max = {max_val}",
            function_name=f"check_probabilities(name='{name}')",
            tolerance={"atol": atol},
            shape=np_arr.shape,
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return arr


def check_normalized(
    arr: T,
    *,
    name: str = "distribution",
    axis: int | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    strict: bool = True,
) -> T:
    """Validate that an array is normalized (sums to 1).

    Args:
        arr: Array-like input to validate.
        name: Descriptive name for error messages.
        axis: Axis along which to check normalization. If None, checks total sum.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        strict: If True (default), raise NormalizationError on failure.
            If False, issue a warning instead.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        NonFiniteError: If array contains NaN or Inf values.
        NormalizationError: If array doesn't sum to 1 (when strict=True).

    Example:
        >>> import numpy as np
        >>> from agentbible import check_normalized
        >>> check_normalized(np.array([0.25, 0.25, 0.25, 0.25]), name="dist")  # Passes
        >>> check_normalized(np.array([0.1, 0.2, 0.3]), name="dist")  # Raises
    """
    np_arr = np.asarray(arr)

    # Check finite first
    check_finite(np_arr, name=name, strict=True)

    total = np.sum(np_arr, axis=axis)

    if axis is None:
        # Total sum should be 1
        total_float = float(total)
        if not np.isclose(total_float, 1.0, rtol=rtol, atol=atol):
            error = NormalizationError(
                f"Array '{name}' is not normalized",
                expected="sum = 1 (probability distribution must sum to 1)",
                got=f"sum = {total_float}",
                function_name=f"check_normalized(name='{name}')",
                tolerance={"rtol": rtol, "atol": atol},
                shape=np_arr.shape,
            )

            if strict:
                raise error
            else:
                warnings.warn(str(error), RuntimeWarning, stacklevel=2)
    else:
        # Each slice along axis should sum to 1
        expected_ones = np.ones_like(total)
        if not np.allclose(total, expected_ones, rtol=rtol, atol=atol):
            max_deviation = float(np.max(np.abs(total - expected_ones)))
            error = NormalizationError(
                f"Array '{name}' is not normalized along axis {axis}",
                expected="sum along axis = 1",
                got=f"max|sum - 1| = {max_deviation}",
                function_name=f"check_normalized(name='{name}')",
                tolerance={"rtol": rtol, "atol": atol},
                shape=np_arr.shape,
            )

            if strict:
                raise error
            else:
                warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return arr


__all__ = [
    "check_finite",
    "check_positive",
    "check_non_negative",
    "check_range",
    "check_probability",
    "check_probabilities",
    "check_normalized",
]
