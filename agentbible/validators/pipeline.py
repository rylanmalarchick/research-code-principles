"""Validation pipeline for composing multiple validators.

This module provides a `ValidationPipeline` class that allows composing
multiple validation checks into a single reusable pipeline.

Example:
    >>> import numpy as np
    >>> from agentbible.validators.pipeline import ValidationPipeline
    >>> from agentbible.validators import check_finite, check_positive, check_range
    >>>
    >>> # Create a pipeline for temperature validation
    >>> validate_temperature = ValidationPipeline([
    ...     check_finite,
    ...     check_positive,
    ...     lambda arr, **kw: check_range(arr, max_val=1000, **kw),
    ... ], name="temperature")
    >>>
    >>> # Use the pipeline
    >>> temps = np.array([20.0, 25.0, 30.0])
    >>> validate_temperature(temps)  # Passes all checks
    >>>
    >>> # Pipeline can also be used as a context manager
    >>> with ValidationPipeline.strict_mode(False):
    ...     validate_temperature(bad_temps)  # Warns instead of raising
"""

from __future__ import annotations

import contextvars
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from numpy.typing import ArrayLike

# Type variables
T = TypeVar("T", bound=ArrayLike)

# Context variable for strict mode override
_strict_mode_override: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "strict_mode_override", default=None
)


@dataclass
class ValidationResult:
    """Result of a validation pipeline execution.

    Attributes:
        passed: Whether all validations passed.
        checks_run: Number of checks that were executed.
        checks_passed: Number of checks that passed.
        errors: List of (check_name, error) tuples for failed checks.
        warnings: List of (check_name, warning_message) tuples.
    """

    passed: bool
    checks_run: int
    checks_passed: int
    errors: list[tuple[str, Exception]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow using result directly in conditionals."""
        return self.passed

    def raise_if_failed(self) -> None:
        """Raise the first error if validation failed."""
        if not self.passed and self.errors:
            raise self.errors[0][1]


CheckFunction = Callable[..., Any]


class ValidationPipeline:
    """Compose multiple validation checks into a reusable pipeline.

    A ValidationPipeline chains multiple check functions together and
    executes them in order. It provides a clean interface for validating
    arrays through multiple constraints.

    Args:
        checks: Sequence of check functions to apply. Each function should
            accept an array as the first argument and `name` and `strict`
            keyword arguments.
        name: Default name to use in error messages.
        strict: Default strict mode. If True (default), raise on first failure.
            If False, collect all failures as warnings.
        stop_on_first_error: If True (default), stop at first failed check.
            If False, run all checks and collect all errors.

    Example:
        >>> from agentbible.validators import check_finite, check_positive
        >>> from agentbible.validators.pipeline import ValidationPipeline
        >>>
        >>> # Simple pipeline
        >>> pipeline = ValidationPipeline([check_finite, check_positive])
        >>> pipeline(np.array([1.0, 2.0, 3.0]), name="values")
        >>>
        >>> # Pipeline with custom name
        >>> energy_validator = ValidationPipeline(
        ...     [check_finite, check_positive],
        ...     name="energy",
        ... )
        >>> energy_validator(energy_array)

        >>> # Collect all errors instead of stopping at first
        >>> result = pipeline.validate_all(data)
        >>> if not result:
        ...     for check_name, error in result.errors:
        ...         print(f"{check_name}: {error}")
    """

    def __init__(
        self,
        checks: Sequence[CheckFunction],
        *,
        name: str = "array",
        strict: bool = True,
        stop_on_first_error: bool = True,
    ) -> None:
        """Initialize the validation pipeline."""
        if not checks:
            msg = "Pipeline must have at least one check"
            raise ValueError(msg)

        self._checks = list(checks)
        self._name = name
        self._strict = strict
        self._stop_on_first_error = stop_on_first_error

    @property
    def checks(self) -> list[CheckFunction]:
        """Return the list of check functions."""
        return self._checks.copy()

    @property
    def name(self) -> str:
        """Return the default name."""
        return self._name

    def __call__(
        self,
        arr: T,
        *,
        name: str | None = None,
        strict: bool | None = None,
    ) -> T:
        """Execute the validation pipeline on an array.

        Args:
            arr: Array to validate.
            name: Name to use in error messages. Defaults to pipeline's name.
            strict: Override strict mode. Defaults to pipeline's strict setting.

        Returns:
            The input array unchanged (for chaining).

        Raises:
            ValidationError: If strict=True and any validation check fails.
        """
        effective_name = name if name is not None else self._name

        # Check for context override
        override = _strict_mode_override.get()
        if override is not None:
            effective_strict = override
        elif strict is not None:
            effective_strict = strict
        else:
            effective_strict = self._strict

        for check in self._checks:
            arr = check(arr, name=effective_name, strict=effective_strict)

        return arr

    def validate_all(
        self,
        arr: T,
        *,
        name: str | None = None,
    ) -> ValidationResult:
        """Run all checks and collect results without raising.

        Unlike `__call__`, this method runs ALL checks regardless of
        failures and returns a ValidationResult with all errors collected.

        Args:
            arr: Array to validate.
            name: Name to use in error messages.

        Returns:
            ValidationResult with all check outcomes.
        """
        effective_name = name if name is not None else self._name

        checks_run = 0
        checks_passed = 0
        errors: list[tuple[str, Exception]] = []
        warning_messages: list[tuple[str, str]] = []

        for check in self._checks:
            check_name = getattr(check, "__name__", str(check))
            checks_run += 1

            try:
                # Run with strict=False to get warnings instead of errors
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    check(arr, name=effective_name, strict=False)

                    # Check if any warnings were issued
                    if w:
                        for warning in w:
                            warning_messages.append((check_name, str(warning.message)))
                    else:
                        checks_passed += 1

            except Exception as e:
                errors.append((check_name, e))

        # If we got warnings but no exceptions, treat warnings as soft failures
        passed = len(errors) == 0 and len(warning_messages) == 0

        return ValidationResult(
            passed=passed,
            checks_run=checks_run,
            checks_passed=checks_passed,
            errors=errors,
            warnings=warning_messages,
        )

    def extend(self, *checks: CheckFunction) -> ValidationPipeline:
        """Create a new pipeline with additional checks.

        Args:
            *checks: Additional check functions to append.

        Returns:
            New ValidationPipeline with the additional checks.
        """
        return ValidationPipeline(
            [*self._checks, *checks],
            name=self._name,
            strict=self._strict,
            stop_on_first_error=self._stop_on_first_error,
        )

    def with_name(self, name: str) -> ValidationPipeline:
        """Create a new pipeline with a different default name.

        Args:
            name: New default name for error messages.

        Returns:
            New ValidationPipeline with the new name.
        """
        return ValidationPipeline(
            self._checks,
            name=name,
            strict=self._strict,
            stop_on_first_error=self._stop_on_first_error,
        )

    @staticmethod
    def strict_mode(strict: bool) -> StrictModeContext:
        """Context manager to temporarily override strict mode.

        Args:
            strict: Whether to use strict mode in the context.

        Returns:
            Context manager that sets the strict mode override.

        Example:
            >>> with ValidationPipeline.strict_mode(False):
            ...     # All pipelines will warn instead of raise
            ...     pipeline(data)
        """
        return StrictModeContext(strict)

    def __repr__(self) -> str:
        """Return a string representation."""
        check_names = [getattr(c, "__name__", str(c)) for c in self._checks]
        return (
            f"ValidationPipeline(checks={check_names}, "
            f"name={self._name!r}, strict={self._strict})"
        )


class StrictModeContext:
    """Context manager for temporarily overriding strict mode."""

    def __init__(self, strict: bool) -> None:
        self._strict = strict
        self._token: contextvars.Token[bool | None] | None = None

    def __enter__(self) -> StrictModeContext:
        self._token = _strict_mode_override.set(self._strict)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _strict_mode_override.reset(self._token)


# Pre-built common pipelines
def create_numeric_pipeline(name: str = "array") -> ValidationPipeline:
    """Create a pipeline for basic numeric validation.

    Checks: finite

    Args:
        name: Default name for error messages.

    Returns:
        ValidationPipeline with basic numeric checks.
    """
    from agentbible.validators.arrays import check_finite

    return ValidationPipeline([check_finite], name=name)


def create_positive_pipeline(name: str = "array") -> ValidationPipeline:
    """Create a pipeline for positive value validation.

    Checks: finite, positive

    Args:
        name: Default name for error messages.

    Returns:
        ValidationPipeline with positive value checks.
    """
    from agentbible.validators.arrays import check_finite, check_positive

    return ValidationPipeline([check_finite, check_positive], name=name)


def create_probability_pipeline(name: str = "probabilities") -> ValidationPipeline:
    """Create a pipeline for probability array validation.

    Checks: finite, probabilities (all values in [0, 1])

    Args:
        name: Default name for error messages.

    Returns:
        ValidationPipeline with probability checks.
    """
    from agentbible.validators.arrays import check_finite, check_probabilities

    return ValidationPipeline([check_finite, check_probabilities], name=name)


def create_distribution_pipeline(name: str = "distribution") -> ValidationPipeline:
    """Create a pipeline for probability distribution validation.

    Checks: finite, probabilities, normalized (sums to 1)

    Args:
        name: Default name for error messages.

    Returns:
        ValidationPipeline with distribution checks.
    """
    from agentbible.validators.arrays import (
        check_finite,
        check_normalized,
        check_probabilities,
    )

    return ValidationPipeline(
        [check_finite, check_probabilities, check_normalized], name=name
    )


__all__ = [
    "ValidationPipeline",
    "ValidationResult",
    "StrictModeContext",
    "create_numeric_pipeline",
    "create_positive_pipeline",
    "create_probability_pipeline",
    "create_distribution_pipeline",
]
