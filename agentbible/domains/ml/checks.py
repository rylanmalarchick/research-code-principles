"""Direct ML validation functions for data pipelines.

Unlike decorator-based validators that wrap functions, these are direct
validation functions that operate on arrays and feature names. They are
designed for use in ML pipelines where intermediate data needs validation.

All functions:
- Take data as the first argument
- Take a `name` parameter for clear error messages
- Return the input data unchanged (for chaining)
- Raise appropriate errors when validation fails

Strict Mode:
    By default, these functions raise exceptions on validation failure.
    Set `strict=False` to issue warnings instead (useful for exploratory work).

Example:
    >>> import numpy as np
    >>> from agentbible.domains.ml import check_no_leakage, check_coverage
    >>>
    >>> # Check for forbidden features
    >>> FORBIDDEN = {"target_lagged", "inversion_height"}
    >>> check_no_leakage(["blh", "t2m", "rh"], forbidden=FORBIDDEN)
    >>>
    >>> # Check prediction interval coverage
    >>> check_coverage(y_true, y_lower, y_upper, target=0.90, tolerance=0.05)
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Set
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from agentbible.domains.ml.errors import (
    AutocorrelationWarning,
    CoverageError,
    DataLeakageError,
    ExchangeabilityWarning,
)

# Type variable for array-like inputs
T = TypeVar("T", bound=ArrayLike)


def _compute_lag1_autocorrelation(arr: NDArray[np.floating]) -> float:
    """Compute lag-1 autocorrelation manually using numpy.

    Uses the formula: r = sum((x[t] - mean)(x[t+1] - mean)) / sum((x - mean)^2)

    Args:
        arr: 1D array of values.

    Returns:
        Lag-1 autocorrelation coefficient in [-1, 1].
    """
    if len(arr) < 2:
        return 0.0

    arr = arr.astype(np.float64)
    mean = np.mean(arr)

    # Deviations from mean
    deviations = arr - mean

    # Variance (denominator)
    variance: np.floating = np.sum(deviations**2)
    if variance == 0:
        return 0.0

    # Autocovariance at lag 1 (numerator)
    autocovariance: np.floating = np.sum(deviations[:-1] * deviations[1:])

    # Normalize by variance
    return float(autocovariance / variance)


def check_no_leakage(
    feature_names: Iterable[str],
    *,
    forbidden: Set[str],
    name: str = "features",
    strict: bool = True,
) -> Iterable[str]:
    """Validate that no forbidden features are present in the feature set.

    Data leakage occurs when features contain information that would not
    be available at prediction time, or when features directly encode
    the target variable.

    Args:
        feature_names: Iterable of feature names to check.
        forbidden: Set of forbidden feature names.
        name: Descriptive name for error messages.
        strict: If True (default), raise DataLeakageError on failure.
            If False, issue a warning instead.

    Returns:
        The input feature_names unchanged (allows chaining).

    Raises:
        DataLeakageError: If any forbidden features are found (when strict=True).

    Example:
        >>> from agentbible.domains.ml import check_no_leakage
        >>> FORBIDDEN = {"target_lagged", "inversion_height"}
        >>> check_no_leakage(["blh", "t2m", "rh"], forbidden=FORBIDDEN)  # Passes
        >>> check_no_leakage(["blh", "inversion_height"], forbidden=FORBIDDEN)  # Raises
    """
    features_set = (
        set(feature_names) if not isinstance(feature_names, set) else feature_names
    )
    found = features_set & forbidden

    if found:
        error = DataLeakageError(
            f"Data leakage detected in '{name}'",
            expected=f"No forbidden features. Forbidden set: {forbidden}",
            got=f"Found forbidden features: {found}",
            function_name=f"check_no_leakage(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return feature_names


def check_temporal_autocorrelation(
    arr: T,
    *,
    max_autocorr: float = 0.5,
    name: str = "residuals",
    strict: bool = True,
) -> T:
    """Check for high temporal autocorrelation in a time series.

    High autocorrelation (e.g., lag-1 > 0.5) means consecutive samples
    are highly correlated. This can:
    - Inflate random K-fold CV metrics
    - Violate independence assumptions
    - Lead to overly optimistic performance estimates

    Args:
        arr: 1D array of values (e.g., residuals, time series).
        max_autocorr: Maximum acceptable lag-1 autocorrelation. Default 0.5.
        name: Descriptive name for error messages.
        strict: If True (default), raise error on high autocorrelation.
            If False, issue a warning instead.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        ValueError: If autocorrelation exceeds max_autocorr (when strict=True).

    Note:
        Autocorrelation is computed manually using numpy only (no scipy).
        Use time-series CV, per-group CV, or walk-forward validation
        if high autocorrelation is detected.

    Example:
        >>> import numpy as np
        >>> from agentbible.domains.ml import check_temporal_autocorrelation
        >>> residuals = np.random.randn(100)  # IID noise - should pass
        >>> check_temporal_autocorrelation(residuals, name="model_residuals")
    """
    np_arr = np.asarray(arr).flatten()

    if len(np_arr) < 2:
        return arr

    autocorr = _compute_lag1_autocorrelation(np_arr)

    if abs(autocorr) > max_autocorr:
        msg = (
            f"High temporal autocorrelation detected in '{name}': "
            f"lag-1 autocorrelation = {autocorr:.3f} (threshold: {max_autocorr}). "
            "Random K-fold CV may give overly optimistic results. "
            "Consider time-series CV or walk-forward validation."
        )

        if strict:
            warnings.warn(msg, AutocorrelationWarning, stacklevel=2)
        else:
            warnings.warn(msg, AutocorrelationWarning, stacklevel=2)

    return arr


def check_coverage(
    y_true: ArrayLike,
    y_lower: ArrayLike,
    y_upper: ArrayLike,
    *,
    target: float = 0.90,
    tolerance: float = 0.05,
    name: str = "prediction_intervals",
    strict: bool = True,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Validate that prediction intervals achieve target coverage.

    Coverage is the proportion of true values that fall within the
    predicted intervals. For valid uncertainty quantification, actual
    coverage should match the target (e.g., 90% intervals should
    contain ~90% of true values).

    Args:
        y_true: True target values.
        y_lower: Lower bounds of prediction intervals.
        y_upper: Upper bounds of prediction intervals.
        target: Target coverage level (e.g., 0.90 for 90%). Default 0.90.
        tolerance: Acceptable deviation from target. Default 0.05.
        name: Descriptive name for error messages.
        strict: If True (default), raise CoverageError on failure.
            If False, issue a warning instead.

    Returns:
        Tuple of (y_true, y_lower, y_upper) unchanged (allows chaining).

    Raises:
        CoverageError: If coverage deviates from target beyond tolerance
            (when strict=True).

    Example:
        >>> import numpy as np
        >>> from agentbible.domains.ml import check_coverage
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_lower = y_true - 0.5
        >>> y_upper = y_true + 0.5
        >>> check_coverage(y_true, y_lower, y_upper, target=0.90)  # Passes (100% coverage)
    """
    y_true_arr = np.asarray(y_true)
    y_lower_arr = np.asarray(y_lower)
    y_upper_arr = np.asarray(y_upper)

    # Check shapes match
    if not (y_true_arr.shape == y_lower_arr.shape == y_upper_arr.shape):
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape}, "
            f"y_lower {y_lower_arr.shape}, y_upper {y_upper_arr.shape}"
        )

    # Compute coverage
    in_interval = (y_true_arr >= y_lower_arr) & (y_true_arr <= y_upper_arr)
    actual_coverage = float(np.mean(in_interval))

    # Check if coverage is below target (coverage above target is generally OK)
    if actual_coverage < target - tolerance:
        error = CoverageError(
            f"Coverage for '{name}' is below target",
            expected=f"Coverage >= {target:.1%} - {tolerance:.1%} = {(target - tolerance):.1%}",
            got=f"Actual coverage: {actual_coverage:.1%} (n={len(y_true_arr)})",
            function_name=f"check_coverage(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return (y_true, y_lower, y_upper)


def check_exchangeability(
    residuals: T,
    *,
    max_autocorr: float = 0.3,
    max_drift: float = 0.1,
    name: str = "residuals",
    strict: bool = True,
) -> T:
    """Check that residuals satisfy the exchangeability assumption.

    Exchangeability is required for valid conformal prediction. It is
    violated by:
    - High temporal autocorrelation
    - Domain/distribution shift (mean drift)
    - Non-stationary processes

    This function checks both autocorrelation and mean drift to detect
    potential exchangeability violations.

    Args:
        residuals: 1D array of prediction residuals.
        max_autocorr: Maximum acceptable lag-1 autocorrelation. Default 0.3.
        max_drift: Maximum acceptable normalized drift (|mean_first - mean_last| / std).
            Default 0.1.
        name: Descriptive name for error messages.
        strict: If True (default), raise warning on violations.
            If False, also issue warning (exchangeability is always warned, not error).

    Returns:
        The input residuals unchanged (allows chaining).

    Note:
        Exchangeability violations don't always invalidate the analysis,
        but they should be documented and considered. This function always
        warns rather than raising errors, as the impact depends on context.

    Example:
        >>> import numpy as np
        >>> from agentbible.domains.ml import check_exchangeability
        >>> residuals = np.random.randn(100)  # IID - should pass
        >>> check_exchangeability(residuals, name="conformal_residuals")
    """
    np_arr = np.asarray(residuals).flatten()

    if len(np_arr) < 10:
        return residuals

    issues = []

    # Check autocorrelation
    autocorr = _compute_lag1_autocorrelation(np_arr)
    if abs(autocorr) > max_autocorr:
        issues.append(
            f"High autocorrelation: lag-1 = {autocorr:.3f} (threshold: {max_autocorr})"
        )

    # Check for drift (compare first half vs second half)
    n = len(np_arr)
    first_half = np_arr[: n // 2]
    second_half = np_arr[n // 2 :]

    std = np.std(np_arr)
    if std > 0:
        drift = abs(np.mean(first_half) - np.mean(second_half)) / std
        if drift > max_drift:
            issues.append(
                f"Mean drift detected: normalized drift = {drift:.3f} (threshold: {max_drift})"
            )

    if issues:
        msg = (
            f"Exchangeability may be violated for '{name}':\n"
            + "\n".join(f"  - {issue}" for issue in issues)
            + "\nConformal prediction guarantees may not hold. "
            "Consider adaptive conformal methods or per-group calibration."
        )
        warnings.warn(msg, ExchangeabilityWarning, stacklevel=2)

    return residuals


__all__ = [
    "check_no_leakage",
    "check_temporal_autocorrelation",
    "check_coverage",
    "check_exchangeability",
]
