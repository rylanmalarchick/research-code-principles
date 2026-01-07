"""ML validation decorators.

Provides decorators for validating ML-specific properties:
- No data leakage: Ensure forbidden features are not used
- CV strategy validation: Warn when random CV may be inappropriate

All validators include:
- Numerical sanity checks
- Support for warn_only mode
- Educational error messages with academic references
- Conditional validation levels (debug/lite/off)

Validation levels:
- "debug" (default): Full validation with all checks
- "lite": Only basic sanity checks (fast)
- "off": Skip validation entirely (DANGEROUS - for benchmarking only)
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Set
from typing import Any, Callable, TypeVar, overload

import numpy as np

from agentbible.domains.ml.checks import (
    _compute_lag1_autocorrelation,
    check_no_leakage,
)
from agentbible.domains.ml.errors import (
    CVStrategyWarning,
)
from agentbible.validators.base import (
    ValidationLevel,
    get_validation_level,
    maybe_warn_validation_off,
)

F = TypeVar("F", bound=Callable[..., Any])


@overload
def validate_no_leakage(
    *,
    forbidden: Set[str],
    feature_names_arg: str = "feature_names",
) -> Callable[[F], F]: ...


@overload
def validate_no_leakage(
    *,
    forbidden: Set[str],
    feature_names_arg: int = 0,
) -> Callable[[F], F]: ...


def validate_no_leakage(
    *,
    forbidden: Set[str],
    feature_names_arg: str | int = "feature_names",
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]:
    """Validate that a function's feature_names argument contains no forbidden features.

    This decorator checks the feature names passed to a function to ensure
    no data leakage occurs through forbidden features.

    Args:
        forbidden: Set of forbidden feature names.
        feature_names_arg: Name or position of the argument containing feature names.
            Default is "feature_names".
        level: Validation level - "debug" (full), "lite" (skip), or "off".
            Can also be set globally via AGENTBIBLE_VALIDATION_LEVEL env var.

    Returns:
        Decorated function that validates feature names before execution.

    Raises:
        DataLeakageError: If any forbidden features are found.

    Example:
        >>> from agentbible.domains.ml import validate_no_leakage
        >>>
        >>> FORBIDDEN = {"target_lagged", "inversion_height"}
        >>>
        >>> @validate_no_leakage(forbidden=FORBIDDEN)
        ... def load_features(feature_names: list[str]) -> dict:
        ...     # Load features from database
        ...     return {"features": feature_names}
        >>>
        >>> load_features(["blh", "t2m"])  # Passes
        >>> load_features(["blh", "inversion_height"])  # Raises DataLeakageError
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine effective validation level
            effective_level = get_validation_level(level)

            # If validation is off, skip everything (with warning)
            if effective_level == ValidationLevel.OFF:
                maybe_warn_validation_off(fn.__name__)
                return fn(*args, **kwargs)

            # If lite mode, skip leakage check (it's not a sanity check)
            if effective_level == ValidationLevel.LITE:
                return fn(*args, **kwargs)

            # Extract feature names from arguments
            feature_names = None

            if isinstance(feature_names_arg, str):
                if feature_names_arg in kwargs:
                    feature_names = kwargs[feature_names_arg]
            elif isinstance(feature_names_arg, int) and len(args) > feature_names_arg:
                feature_names = args[feature_names_arg]

            # If we found feature names, check them
            if feature_names is not None:
                check_no_leakage(
                    feature_names,
                    forbidden=forbidden,
                    name=f"{fn.__name__}.{feature_names_arg}",
                    strict=True,
                )

            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@overload
def validate_cv_strategy(func: F) -> F: ...


@overload
def validate_cv_strategy(
    *,
    max_autocorr: float = 0.5,
    target_arg: str | int = "y",
    warn_only: bool = True,
    level: str | ValidationLevel | None = None,
) -> Callable[[F], F]: ...


def validate_cv_strategy(
    func: F | None = None,
    *,
    max_autocorr: float = 0.5,
    target_arg: str | int = "y",
    warn_only: bool = True,
    level: str | ValidationLevel | None = None,
) -> F | Callable[[F], F]:
    """Validate that random CV is appropriate for the data.

    This decorator checks the target variable for temporal autocorrelation.
    High autocorrelation indicates that random K-fold CV may give overly
    optimistic results, and time-series CV should be used instead.

    Can be used with or without arguments:
        @validate_cv_strategy
        def train_model(X, y): ...

        @validate_cv_strategy(max_autocorr=0.3)
        def train_precise_model(X, y): ...

    Args:
        func: The function to decorate (when used without parentheses).
        max_autocorr: Maximum acceptable lag-1 autocorrelation. Default 0.5.
        target_arg: Name or position of the target variable argument.
            Default is "y".
        warn_only: If True (default), issue warning on violation.
            If False, raise ValueError.
        level: Validation level - "debug" (full), "lite" (skip), or "off".

    Returns:
        Decorated function that validates CV appropriateness.

    Example:
        >>> import numpy as np
        >>> from agentbible.domains.ml import validate_cv_strategy
        >>>
        >>> @validate_cv_strategy
        ... def train_model(X, y):
        ...     # Train model using CV
        ...     return "model"
        >>>
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randn(100)  # IID - no warning
        >>> train_model(X, y)
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine effective validation level
            effective_level = get_validation_level(level)

            # If validation is off, skip everything
            if effective_level == ValidationLevel.OFF:
                maybe_warn_validation_off(fn.__name__)
                return fn(*args, **kwargs)

            # If lite mode, skip autocorrelation check
            if effective_level == ValidationLevel.LITE:
                return fn(*args, **kwargs)

            # Extract target from arguments
            target = None

            if isinstance(target_arg, str):
                if target_arg in kwargs:
                    target = kwargs[target_arg]
                # Also check positional args - common pattern is train(X, y)
                # where y is the second positional argument
                elif target_arg == "y" and len(args) > 1:
                    target = args[1]
            elif isinstance(target_arg, int) and len(args) > target_arg:
                target = args[target_arg]

            # If we found target, check autocorrelation
            if target is not None:
                target_arr = np.asarray(target).flatten()

                if len(target_arr) >= 10:
                    autocorr = _compute_lag1_autocorrelation(target_arr)

                    if abs(autocorr) > max_autocorr:
                        msg = (
                            f"High temporal autocorrelation in target (lag-1 = {autocorr:.3f}). "
                            f"Random K-fold CV may give overly optimistic results. "
                            f"Consider using time-series CV, GroupKFold, or walk-forward validation."
                        )

                        if warn_only:
                            warnings.warn(msg, CVStrategyWarning, stacklevel=2)
                        else:
                            raise ValueError(msg)

            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    # Handle both @validate_cv_strategy and @validate_cv_strategy()
    if func is not None:
        return decorator(func)
    return decorator


__all__ = [
    "validate_no_leakage",
    "validate_cv_strategy",
]
