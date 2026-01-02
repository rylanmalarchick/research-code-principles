"""Tests for bounds validators."""

from __future__ import annotations

import numpy as np
import pytest

from agentbible.validators import (
    ValidationError,
    validate_finite,
    validate_non_negative,
    validate_positive,
    validate_range,
)


class TestValidatePositive:
    """Tests for @validate_positive decorator."""

    def test_valid_positive_scalar(self) -> None:
        """Positive scalar is valid."""

        @validate_positive
        def get_value() -> float:
            return 1.5

        assert get_value() == 1.5

    def test_valid_positive_array(self) -> None:
        """Array with all positive values is valid."""

        @validate_positive
        def get_values() -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

        result = get_values()
        assert len(result) == 3

    def test_invalid_zero(self) -> None:
        """Zero raises ValidationError (not positive)."""

        @validate_positive
        def get_value() -> float:
            return 0.0

        with pytest.raises(ValidationError) as exc_info:
            get_value()

        assert "not positive" in str(exc_info.value).lower()
        assert "> 0" in str(exc_info.value)

    def test_invalid_negative(self) -> None:
        """Negative value raises ValidationError."""

        @validate_positive
        def get_value() -> float:
            return -1.0

        with pytest.raises(ValidationError):
            get_value()

    def test_array_with_zero(self) -> None:
        """Array containing zero raises ValidationError."""

        @validate_positive
        def get_values() -> np.ndarray:
            return np.array([1.0, 0.0, 2.0])

        with pytest.raises(ValidationError):
            get_values()

    def test_tolerance(self) -> None:
        """Tolerance allows near-zero positive."""

        @validate_positive(atol=1e-6)
        def get_value() -> float:
            return 1e-8  # Very small but positive

        # Should pass - value is positive
        get_value()


class TestValidateNonNegative:
    """Tests for @validate_non_negative decorator."""

    def test_valid_positive(self) -> None:
        """Positive value is valid."""

        @validate_non_negative
        def get_value() -> float:
            return 1.5

        assert get_value() == 1.5

    def test_valid_zero(self) -> None:
        """Zero is valid."""

        @validate_non_negative
        def get_value() -> float:
            return 0.0

        assert get_value() == 0.0

    def test_valid_array(self) -> None:
        """Array with non-negative values is valid."""

        @validate_non_negative
        def get_values() -> np.ndarray:
            return np.array([0.0, 1.0, 2.0])

        get_values()

    def test_invalid_negative(self) -> None:
        """Negative value raises ValidationError."""

        @validate_non_negative
        def get_value() -> float:
            return -1.0

        with pytest.raises(ValidationError) as exc_info:
            get_value()

        assert "negative" in str(exc_info.value)
        assert ">= 0" in str(exc_info.value)

    def test_tolerance(self) -> None:
        """Tolerance allows small negative."""

        @validate_non_negative(atol=1e-6)
        def get_value() -> float:
            return -1e-8  # Small negative within tolerance

        get_value()

        @validate_non_negative(atol=1e-10)
        def strict() -> float:
            return -1e-8

        with pytest.raises(ValidationError):
            strict()


class TestValidateRange:
    """Tests for @validate_range decorator."""

    def test_valid_in_range(self) -> None:
        """Value in range is valid."""

        @validate_range(0.0, 1.0)
        def get_value() -> float:
            return 0.5

        assert get_value() == 0.5

    def test_valid_at_min(self) -> None:
        """Value at minimum is valid (inclusive)."""

        @validate_range(0.0, 1.0)
        def get_value() -> float:
            return 0.0

        get_value()

    def test_valid_at_max(self) -> None:
        """Value at maximum is valid (inclusive)."""

        @validate_range(0.0, 1.0)
        def get_value() -> float:
            return 1.0

        get_value()

    def test_invalid_below_min(self) -> None:
        """Value below minimum raises ValidationError."""

        @validate_range(0.0, 1.0)
        def get_value() -> float:
            return -0.1

        with pytest.raises(ValidationError) as exc_info:
            get_value()

        assert "below minimum" in str(exc_info.value).lower()

    def test_invalid_above_max(self) -> None:
        """Value above maximum raises ValidationError."""

        @validate_range(0.0, 1.0)
        def get_value() -> float:
            return 1.1

        with pytest.raises(ValidationError) as exc_info:
            get_value()

        assert "above maximum" in str(exc_info.value).lower()

    def test_exclusive_bounds(self) -> None:
        """Exclusive bounds work correctly."""

        @validate_range(0.0, 1.0, inclusive=False)
        def get_value() -> float:
            return 0.5

        get_value()

        @validate_range(0.0, 1.0, inclusive=False)
        def at_min() -> float:
            return 0.0

        with pytest.raises(ValidationError):
            at_min()

        @validate_range(0.0, 1.0, inclusive=False)
        def at_max() -> float:
            return 1.0

        with pytest.raises(ValidationError):
            at_max()

    def test_min_only(self) -> None:
        """Can specify only minimum."""

        @validate_range(min_val=0.0)
        def get_value() -> float:
            return 100.0

        get_value()

        @validate_range(min_val=0.0)
        def negative() -> float:
            return -1.0

        with pytest.raises(ValidationError):
            negative()

    def test_max_only(self) -> None:
        """Can specify only maximum."""

        @validate_range(max_val=100.0)
        def get_value() -> float:
            return -50.0

        get_value()

        @validate_range(max_val=100.0)
        def too_high() -> float:
            return 101.0

        with pytest.raises(ValidationError):
            too_high()

    def test_array_validation(self) -> None:
        """Works with arrays."""

        @validate_range(0.0, 1.0)
        def get_values() -> np.ndarray:
            return np.array([0.1, 0.5, 0.9])

        get_values()

        @validate_range(0.0, 1.0)
        def bad_array() -> np.ndarray:
            return np.array([0.1, 1.5, 0.9])

        with pytest.raises(ValidationError):
            bad_array()

    def test_no_bounds_raises_error(self) -> None:
        """Must specify at least one bound."""
        with pytest.raises(ValueError):
            validate_range()  # type: ignore[call-overload]


class TestValidateFinite:
    """Tests for @validate_finite decorator."""

    def test_valid_finite_scalar(self) -> None:
        """Finite scalar is valid."""

        @validate_finite
        def get_value() -> float:
            return 1.5

        assert get_value() == 1.5

    def test_valid_finite_array(self) -> None:
        """Array with all finite values is valid."""

        @validate_finite
        def get_values() -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

        get_values()

    def test_valid_zero(self) -> None:
        """Zero is finite."""

        @validate_finite
        def get_value() -> float:
            return 0.0

        get_value()

    def test_valid_negative(self) -> None:
        """Negative numbers are finite."""

        @validate_finite
        def get_value() -> float:
            return -1e10

        get_value()

    def test_invalid_nan(self, array_with_nan: np.ndarray) -> None:
        """NaN raises ValidationError."""

        @validate_finite
        def get_values() -> np.ndarray:
            return array_with_nan

        with pytest.raises(ValidationError) as exc_info:
            get_values()

        assert "non-finite" in str(exc_info.value).lower()
        assert "NaN" in str(exc_info.value)

    def test_invalid_inf(self, array_with_inf: np.ndarray) -> None:
        """Inf raises ValidationError."""

        @validate_finite
        def get_values() -> np.ndarray:
            return array_with_inf

        with pytest.raises(ValidationError) as exc_info:
            get_values()

        assert "non-finite" in str(exc_info.value).lower()
        assert "Inf" in str(exc_info.value)

    def test_invalid_negative_inf(self) -> None:
        """Negative Inf raises ValidationError."""

        @validate_finite
        def get_value() -> float:
            return float("-inf")

        with pytest.raises(ValidationError):
            get_value()

    def test_scalar_nan(self) -> None:
        """Scalar NaN raises ValidationError."""

        @validate_finite
        def get_value() -> float:
            return float("nan")

        with pytest.raises(ValidationError):
            get_value()

    def test_decorator_with_parentheses(self) -> None:
        """Decorator works with explicit parentheses."""

        @validate_finite()
        def get_value() -> float:
            return 1.5

        get_value()
