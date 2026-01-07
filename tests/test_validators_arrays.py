"""Tests for array validation functions (check_* API)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from agentbible import (
    BoundsError,
    NonFiniteError,
    NormalizationError,
    ProbabilityBoundsError,
)
from agentbible.validators.arrays import (
    check_finite,
    check_non_negative,
    check_normalized,
    check_positive,
    check_probabilities,
    check_probability,
    check_range,
)


class TestCheckFinite:
    """Tests for check_finite function."""

    def test_valid_finite_array(self) -> None:
        """Finite array passes validation."""
        arr = np.array([1.0, 2.0, 3.0])
        result = check_finite(arr, name="test")
        assert np.array_equal(result, arr)

    def test_valid_finite_scalar(self) -> None:
        """Finite scalar passes validation."""
        result = check_finite(np.array(1.5), name="scalar")
        assert result == 1.5

    def test_returns_input_for_chaining(self) -> None:
        """Returns input array unchanged for chaining."""
        arr = np.array([1.0, 2.0, 3.0])
        result = check_finite(arr, name="test")
        assert result is arr

    def test_invalid_nan(self) -> None:
        """Array with NaN raises NonFiniteError."""
        arr = np.array([1.0, np.nan, 3.0])
        with pytest.raises(NonFiniteError) as exc_info:
            check_finite(arr, name="test_array")

        assert "non-finite" in str(exc_info.value).lower()
        assert "NaN" in str(exc_info.value)
        assert "test_array" in str(exc_info.value)

    def test_invalid_inf(self) -> None:
        """Array with Inf raises NonFiniteError."""
        arr = np.array([1.0, np.inf, 3.0])
        with pytest.raises(NonFiniteError) as exc_info:
            check_finite(arr, name="test_array")

        assert "Inf" in str(exc_info.value)

    def test_invalid_negative_inf(self) -> None:
        """Array with -Inf raises NonFiniteError."""
        arr = np.array([1.0, -np.inf, 3.0])
        with pytest.raises(NonFiniteError):
            check_finite(arr, name="test")

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning instead of raising."""
        arr = np.array([1.0, np.nan, 3.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_finite(arr, name="test", strict=False)
            assert len(w) == 1
            assert "non-finite" in str(w[0].message).lower()
        # Still returns the array
        assert np.array_equal(result, arr, equal_nan=True)


class TestCheckPositive:
    """Tests for check_positive function."""

    def test_valid_positive_array(self) -> None:
        """All positive values pass validation."""
        arr = np.array([1.0, 2.0, 3.0])
        result = check_positive(arr, name="mass")
        assert np.array_equal(result, arr)

    def test_invalid_zero(self) -> None:
        """Zero raises BoundsError."""
        arr = np.array([1.0, 0.0, 3.0])
        with pytest.raises(BoundsError) as exc_info:
            check_positive(arr, name="mass")

        assert "not positive" in str(exc_info.value).lower()
        assert "> 0" in str(exc_info.value)

    def test_invalid_negative(self) -> None:
        """Negative value raises BoundsError."""
        arr = np.array([1.0, -0.5, 3.0])
        with pytest.raises(BoundsError):
            check_positive(arr, name="mass")

    def test_tolerance(self) -> None:
        """Tolerance allows near-zero positive values."""
        arr = np.array([1e-12, 1.0])
        # Without tolerance, this is fine (it's positive)
        check_positive(arr, name="small")

        # With tolerance allowing slight negative
        arr_neg = np.array([-1e-12, 1.0])
        check_positive(arr_neg, name="small", atol=1e-10)

    def test_raises_on_nan(self) -> None:
        """NaN values raise NonFiniteError (checked first)."""
        arr = np.array([1.0, np.nan])
        with pytest.raises(NonFiniteError):
            check_positive(arr, name="test")

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning for non-positive."""
        arr = np.array([1.0, -0.5, 3.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_positive(arr, name="test", strict=False)
            assert len(w) == 1
            assert "not positive" in str(w[0].message).lower()
        assert np.array_equal(result, arr)


class TestCheckNonNegative:
    """Tests for check_non_negative function."""

    def test_valid_non_negative(self) -> None:
        """Non-negative values pass validation."""
        arr = np.array([0.0, 1.0, 2.0])
        result = check_non_negative(arr, name="count")
        assert np.array_equal(result, arr)

    def test_valid_zero(self) -> None:
        """Zero is valid for non-negative."""
        arr = np.array([0.0])
        check_non_negative(arr, name="count")

    def test_invalid_negative(self) -> None:
        """Negative value raises BoundsError."""
        arr = np.array([0.0, -0.5, 1.0])
        with pytest.raises(BoundsError) as exc_info:
            check_non_negative(arr, name="count")

        assert "negative" in str(exc_info.value).lower()
        assert ">= 0" in str(exc_info.value)

    def test_tolerance(self) -> None:
        """Tolerance allows small negative values."""
        arr = np.array([-1e-12, 0.0, 1.0])
        check_non_negative(arr, name="count", atol=1e-10)

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning for negative values."""
        arr = np.array([-1.0, 0.0, 1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_non_negative(arr, name="test", strict=False)
            assert len(w) == 1


class TestCheckRange:
    """Tests for check_range function."""

    def test_valid_in_range(self) -> None:
        """Values in range pass validation."""
        arr = np.array([0.2, 0.5, 0.8])
        result = check_range(arr, 0.0, 1.0, name="fidelity")
        assert np.array_equal(result, arr)

    def test_valid_at_boundaries(self) -> None:
        """Values at boundaries pass (inclusive by default)."""
        arr = np.array([0.0, 0.5, 1.0])
        check_range(arr, 0.0, 1.0, name="fidelity")

    def test_invalid_below_min(self) -> None:
        """Value below minimum raises BoundsError."""
        arr = np.array([-0.1, 0.5, 0.8])
        with pytest.raises(BoundsError) as exc_info:
            check_range(arr, 0.0, 1.0, name="fidelity")

        assert "below minimum" in str(exc_info.value).lower()

    def test_invalid_above_max(self) -> None:
        """Value above maximum raises BoundsError."""
        arr = np.array([0.2, 0.5, 1.1])
        with pytest.raises(BoundsError) as exc_info:
            check_range(arr, 0.0, 1.0, name="fidelity")

        assert "above maximum" in str(exc_info.value).lower()

    def test_exclusive_bounds(self) -> None:
        """Exclusive bounds reject boundary values."""
        arr = np.array([0.5])
        check_range(arr, 0.0, 1.0, name="test", inclusive=False)

        arr_at_min = np.array([0.0])
        with pytest.raises(BoundsError):
            check_range(arr_at_min, 0.0, 1.0, name="test", inclusive=False)

        arr_at_max = np.array([1.0])
        with pytest.raises(BoundsError):
            check_range(arr_at_max, 0.0, 1.0, name="test", inclusive=False)

    def test_min_only(self) -> None:
        """Can specify only minimum bound."""
        arr = np.array([100.0, 200.0])
        check_range(arr, min_val=0.0, name="test")

        arr_neg = np.array([-1.0, 100.0])
        with pytest.raises(BoundsError):
            check_range(arr_neg, min_val=0.0, name="test")

    def test_max_only(self) -> None:
        """Can specify only maximum bound."""
        arr = np.array([-100.0, 50.0])
        check_range(arr, max_val=100.0, name="test")

        arr_high = np.array([50.0, 150.0])
        with pytest.raises(BoundsError):
            check_range(arr_high, max_val=100.0, name="test")

    def test_no_bounds_raises_error(self) -> None:
        """Must specify at least one bound."""
        arr = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            check_range(arr, name="test")

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning for out-of-range values."""
        arr = np.array([0.5, 1.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_range(arr, 0.0, 1.0, name="test", strict=False)
            assert len(w) == 1


class TestCheckProbability:
    """Tests for check_probability function (scalar)."""

    def test_valid_probability(self) -> None:
        """Valid probability in [0, 1] passes."""
        result = check_probability(0.5, name="p_success")
        assert result == 0.5

    def test_valid_at_boundaries(self) -> None:
        """Values at 0 and 1 are valid."""
        check_probability(0.0, name="p")
        check_probability(1.0, name="p")

    def test_invalid_below_zero(self) -> None:
        """Negative probability raises ProbabilityBoundsError."""
        with pytest.raises(ProbabilityBoundsError) as exc_info:
            check_probability(-0.1, name="p_fail")

        assert "not a valid probability" in str(exc_info.value).lower()
        assert "p_fail" in str(exc_info.value)

    def test_invalid_above_one(self) -> None:
        """Probability > 1 raises ProbabilityBoundsError."""
        with pytest.raises(ProbabilityBoundsError):
            check_probability(1.1, name="p_fail")

    def test_invalid_nan(self) -> None:
        """NaN raises NonFiniteError."""
        with pytest.raises(NonFiniteError):
            check_probability(float("nan"), name="p")

    def test_invalid_inf(self) -> None:
        """Inf raises NonFiniteError."""
        with pytest.raises(NonFiniteError):
            check_probability(float("inf"), name="p")

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning for invalid probability."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_probability(1.5, name="p", strict=False)
            assert len(w) == 1
        assert result == 1.5


class TestCheckProbabilities:
    """Tests for check_probabilities function (array)."""

    def test_valid_probabilities(self) -> None:
        """Valid probability array passes."""
        arr = np.array([0.1, 0.5, 0.9])
        result = check_probabilities(arr, name="probs")
        assert np.array_equal(result, arr)

    def test_invalid_negative(self) -> None:
        """Negative values raise ProbabilityBoundsError."""
        arr = np.array([-0.1, 0.5, 0.9])
        with pytest.raises(ProbabilityBoundsError) as exc_info:
            check_probabilities(arr, name="probs")

        assert "below 0" in str(exc_info.value).lower()

    def test_invalid_above_one(self) -> None:
        """Values > 1 raise ProbabilityBoundsError."""
        arr = np.array([0.1, 1.5, 0.9])
        with pytest.raises(ProbabilityBoundsError) as exc_info:
            check_probabilities(arr, name="probs")

        assert "above 1" in str(exc_info.value).lower()

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning for invalid probabilities."""
        arr = np.array([0.1, 1.5, 0.9])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_probabilities(arr, name="probs", strict=False)
            assert len(w) == 1


class TestCheckNormalized:
    """Tests for check_normalized function."""

    def test_valid_normalized(self) -> None:
        """Normalized array (sums to 1) passes."""
        arr = np.array([0.25, 0.25, 0.25, 0.25])
        result = check_normalized(arr, name="dist")
        assert np.array_equal(result, arr)

    def test_invalid_not_normalized(self) -> None:
        """Array not summing to 1 raises NormalizationError."""
        arr = np.array([0.1, 0.2, 0.3])  # sum = 0.6
        with pytest.raises(NormalizationError) as exc_info:
            check_normalized(arr, name="dist")

        assert "not normalized" in str(exc_info.value).lower()

    def test_axis_normalization(self) -> None:
        """Can check normalization along specific axis."""
        # Each row sums to 1
        arr = np.array([[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])
        check_normalized(arr, name="probs", axis=1)

        # Rows don't all sum to 1
        bad_arr = np.array([[0.5, 0.5], [0.3, 0.3]])  # second row sums to 0.6
        with pytest.raises(NormalizationError):
            check_normalized(bad_arr, name="probs", axis=1)

    def test_tolerance(self) -> None:
        """Tolerance allows slight deviation from 1."""
        arr = np.array([0.25, 0.25, 0.25, 0.25 + 1e-7])
        check_normalized(arr, name="dist", atol=1e-6)

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning for non-normalized."""
        arr = np.array([0.1, 0.2, 0.3])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_normalized(arr, name="dist", strict=False)
            assert len(w) == 1


class TestChaining:
    """Tests for chaining multiple check functions."""

    def test_chaining_checks(self) -> None:
        """Multiple checks can be chained."""
        arr = np.array([0.1, 0.2, 0.3, 0.4])

        # Chain: finite -> positive -> probabilities
        result = check_probabilities(
            check_positive(
                check_finite(arr, name="probs"),
                name="probs",
            ),
            name="probs",
        )

        assert np.array_equal(result, arr)

    def test_chaining_fails_on_first_invalid(self) -> None:
        """Chain stops at first failing check."""
        arr = np.array([np.nan, 0.5])

        with pytest.raises(NonFiniteError):
            check_positive(
                check_finite(arr, name="test"),
                name="test",
            )


class TestNameParameter:
    """Tests for name parameter in error messages."""

    def test_name_in_error_message(self) -> None:
        """Name appears in error messages."""
        arr = np.array([np.nan])
        with pytest.raises(NonFiniteError) as exc_info:
            check_finite(arr, name="my_custom_variable")

        assert "my_custom_variable" in str(exc_info.value)

    def test_default_name(self) -> None:
        """Default name is 'array' when not specified."""
        arr = np.array([np.nan])
        with pytest.raises(NonFiniteError) as exc_info:
            check_finite(arr)

        assert "array" in str(exc_info.value)
