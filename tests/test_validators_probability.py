"""Tests for probability validators."""

from __future__ import annotations

import numpy as np
import pytest

from agentbible.validators import (
    ValidationError,
    validate_normalized,
    validate_probabilities,
    validate_probability,
)


class TestValidateProbability:
    """Tests for @validate_probability decorator."""

    def test_valid_zero(self) -> None:
        """Zero is a valid probability."""

        @validate_probability
        def get_prob() -> float:
            return 0.0

        assert get_prob() == 0.0

    def test_valid_one(self) -> None:
        """One is a valid probability."""

        @validate_probability
        def get_prob() -> float:
            return 1.0

        assert get_prob() == 1.0

    def test_valid_half(self) -> None:
        """0.5 is a valid probability."""

        @validate_probability
        def get_prob() -> float:
            return 0.5

        assert get_prob() == 0.5

    def test_invalid_negative(self) -> None:
        """Negative value raises ValidationError."""

        @validate_probability
        def get_prob() -> float:
            return -0.1

        with pytest.raises(ValidationError) as exc_info:
            get_prob()

        assert "not a valid probability" in str(exc_info.value).lower()
        assert "0 <= p <= 1" in str(exc_info.value)

    def test_invalid_greater_than_one(self) -> None:
        """Value > 1 raises ValidationError."""

        @validate_probability
        def get_prob() -> float:
            return 1.5

        with pytest.raises(ValidationError) as exc_info:
            get_prob()

        assert "not a valid probability" in str(exc_info.value).lower()

    def test_decorator_with_parentheses(self) -> None:
        """Decorator works with explicit parentheses."""

        @validate_probability()
        def get_prob() -> float:
            return 0.5

        get_prob()

    def test_custom_tolerance(self) -> None:
        """Custom tolerance allows small violations."""

        @validate_probability(atol=0.01)
        def get_prob() -> float:
            return 1.005  # Slightly over 1

        # Should pass with loose tolerance
        get_prob()

        @validate_probability(atol=1e-10)
        def get_strict() -> float:
            return 1.005

        # Should fail with strict tolerance
        with pytest.raises(ValidationError):
            get_strict()


class TestValidateProbabilities:
    """Tests for @validate_probabilities decorator."""

    def test_valid_array(self, probability_array: np.ndarray) -> None:
        """Array with all values in [0,1] is valid."""

        @validate_probabilities
        def get_probs() -> np.ndarray:
            return probability_array

        result = get_probs()
        assert np.allclose(result, probability_array)

    def test_valid_zeros(self) -> None:
        """Array of zeros is valid."""

        @validate_probabilities
        def get_probs() -> np.ndarray:
            return np.zeros(5)

        get_probs()

    def test_valid_ones(self) -> None:
        """Array of ones is valid."""

        @validate_probabilities
        def get_probs() -> np.ndarray:
            return np.ones(5)

        get_probs()

    def test_invalid_negative(self) -> None:
        """Array with negative value raises ValidationError."""

        @validate_probabilities
        def get_probs() -> np.ndarray:
            return np.array([0.5, -0.1, 0.5])

        with pytest.raises(ValidationError) as exc_info:
            get_probs()

        assert "below 0" in str(exc_info.value).lower()

    def test_invalid_greater_than_one(self) -> None:
        """Array with value > 1 raises ValidationError."""

        @validate_probabilities
        def get_probs() -> np.ndarray:
            return np.array([0.5, 1.5, 0.5])

        with pytest.raises(ValidationError) as exc_info:
            get_probs()

        assert "above 1" in str(exc_info.value).lower()

    def test_2d_array(self) -> None:
        """Works with 2D arrays."""

        @validate_probabilities
        def get_matrix() -> np.ndarray:
            return np.array([[0.1, 0.2], [0.3, 0.4]])

        get_matrix()


class TestValidateNormalized:
    """Tests for @validate_normalized decorator."""

    def test_valid_normalized(self, normalized_array: np.ndarray) -> None:
        """Array that sums to 1 is valid."""

        @validate_normalized
        def get_dist() -> np.ndarray:
            return normalized_array

        result = get_dist()
        assert np.allclose(result, normalized_array)

    def test_valid_single_value(self) -> None:
        """Single value of 1.0 is normalized."""

        @validate_normalized
        def get_dist() -> np.ndarray:
            return np.array([1.0])

        get_dist()

    def test_invalid_not_normalized(self, non_normalized_array: np.ndarray) -> None:
        """Array that doesn't sum to 1 raises ValidationError."""

        @validate_normalized
        def get_dist() -> np.ndarray:
            return non_normalized_array

        with pytest.raises(ValidationError) as exc_info:
            get_dist()

        assert "not normalized" in str(exc_info.value).lower()
        assert "sum = 1" in str(exc_info.value)

    def test_axis_normalization(self) -> None:
        """Can validate normalization along specific axis."""
        # Each row sums to 1
        row_normalized = np.array([[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])

        @validate_normalized(axis=1)
        def get_row_probs() -> np.ndarray:
            return row_normalized

        get_row_probs()

        # Each column sums to 1
        col_normalized = np.array([[0.5, 0.3], [0.5, 0.7]])

        @validate_normalized(axis=0)
        def get_col_probs() -> np.ndarray:
            return col_normalized

        get_col_probs()

    def test_axis_normalization_invalid(self) -> None:
        """Invalid axis normalization raises error."""
        # Rows don't sum to 1
        bad_rows = np.array([[0.5, 0.3], [0.5, 0.7]])

        @validate_normalized(axis=1)
        def get_bad() -> np.ndarray:
            return bad_rows

        with pytest.raises(ValidationError) as exc_info:
            get_bad()

        assert "axis 1" in str(exc_info.value).lower()

    def test_custom_tolerance(self) -> None:
        """Custom tolerance is respected."""
        almost_normalized = np.array([0.5, 0.5001])

        @validate_normalized(rtol=1e-2)
        def loose() -> np.ndarray:
            return almost_normalized

        @validate_normalized(rtol=1e-6)
        def strict() -> np.ndarray:
            return almost_normalized

        loose()

        with pytest.raises(ValidationError):
            strict()
