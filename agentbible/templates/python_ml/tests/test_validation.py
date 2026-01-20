"""Tests for ML validation utilities.

These tests demonstrate the specification-before-code principle:
each test defines expected behavior for the validation functions.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.validation import (
    check_array_shape,
    check_class_balance,
    check_feature_scaling,
    check_no_data_leakage,
    check_no_nan_inf,
    check_prediction_range,
    check_probability_predictions,
)


class TestCheckNoDataLeakage:
    """Tests for check_no_data_leakage function."""

    def test_no_overlap_passes(self, train_test_indices: tuple) -> None:
        """Non-overlapping indices pass validation."""
        train_idx, test_idx = train_test_indices
        check_no_data_leakage(train_idx, test_idx)  # Should not raise

    def test_overlap_raises(self, leaky_indices: tuple) -> None:
        """Overlapping indices raise ValueError."""
        train_idx, test_idx = leaky_indices
        with pytest.raises(ValueError, match="Data leakage"):
            check_no_data_leakage(train_idx, test_idx)

    def test_works_with_lists(self) -> None:
        """Works with Python lists."""
        check_no_data_leakage([0, 1, 2], [3, 4, 5])  # Should not raise

    def test_works_with_sets(self) -> None:
        """Works with Python sets."""
        check_no_data_leakage({0, 1, 2}, {3, 4, 5})  # Should not raise

    def test_error_includes_name(self) -> None:
        """Error message includes name if provided."""
        with pytest.raises(ValueError, match="'my_split'"):
            check_no_data_leakage([0, 1], [1, 2], name="my_split")


class TestCheckClassBalance:
    """Tests for check_class_balance function."""

    def test_balanced_classes_pass(self) -> None:
        """Balanced classes pass validation."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        check_class_balance(labels, max_ratio=2.0)  # Should not raise

    def test_imbalanced_classes_raise(self, imbalanced_data: tuple) -> None:
        """Imbalanced classes raise ValueError."""
        _, y = imbalanced_data
        with pytest.raises(ValueError, match="Class imbalance"):
            check_class_balance(y, max_ratio=5.0)

    def test_single_class_raises(self) -> None:
        """Single class raises ValueError."""
        labels = np.array([0, 0, 0, 0])
        with pytest.raises(ValueError, match="only one class"):
            check_class_balance(labels)

    def test_custom_max_ratio(self) -> None:
        """Custom max_ratio is respected."""
        labels = np.array([0, 0, 0, 0, 0, 1])  # 5:1 ratio
        check_class_balance(labels, max_ratio=6.0)  # Should pass
        with pytest.raises(ValueError):
            check_class_balance(labels, max_ratio=4.0)  # Should fail


class TestCheckFeatureScaling:
    """Tests for check_feature_scaling function."""

    def test_standardized_features_pass(
        self, standardized_features: NDArray[np.floating]
    ) -> None:
        """Standardized features pass standard scaling check."""
        check_feature_scaling(standardized_features, expected="standard")

    def test_minmax_features_pass(
        self, minmax_features: NDArray[np.floating]
    ) -> None:
        """Min-max scaled features pass minmax check."""
        check_feature_scaling(minmax_features, expected="minmax")

    def test_unscaled_fails_standard(
        self, unscaled_features: NDArray[np.floating]
    ) -> None:
        """Unscaled features fail standard scaling check."""
        with pytest.raises(ValueError, match="not standardized"):
            check_feature_scaling(unscaled_features, expected="standard")


class TestCheckNoNanInf:
    """Tests for check_no_nan_inf function."""

    def test_clean_data_passes(self, standardized_features: NDArray[np.floating]) -> None:
        """Clean data passes validation."""
        check_no_nan_inf(standardized_features)  # Should not raise

    def test_nan_raises(self, features_with_nan: NDArray[np.floating]) -> None:
        """Data with NaN raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            check_no_nan_inf(features_with_nan)

    def test_inf_raises(self, features_with_inf: NDArray[np.floating]) -> None:
        """Data with Inf raises ValueError."""
        with pytest.raises(ValueError, match="Inf"):
            check_no_nan_inf(features_with_inf)


class TestCheckArrayShape:
    """Tests for check_array_shape function."""

    def test_correct_ndim_passes(self) -> None:
        """Correct ndim passes validation."""
        X = np.random.randn(10, 5)
        check_array_shape(X, expected_ndim=2)

    def test_wrong_ndim_raises(self) -> None:
        """Wrong ndim raises ValueError."""
        X = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="dimensions"):
            check_array_shape(X, expected_ndim=3)

    def test_min_samples_passes(self) -> None:
        """Sufficient samples pass validation."""
        X = np.random.randn(100, 5)
        check_array_shape(X, min_samples=50)

    def test_insufficient_samples_raises(self) -> None:
        """Insufficient samples raise ValueError."""
        X = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="minimum"):
            check_array_shape(X, min_samples=50)


class TestCheckPredictionRange:
    """Tests for check_prediction_range function."""

    def test_in_range_passes(self) -> None:
        """Predictions in range pass validation."""
        preds = np.array([0.1, 0.5, 0.9])
        check_prediction_range(preds, min_val=0.0, max_val=1.0)

    def test_below_min_raises(self) -> None:
        """Predictions below min raise ValueError."""
        preds = np.array([-0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="below minimum"):
            check_prediction_range(preds, min_val=0.0)

    def test_above_max_raises(self) -> None:
        """Predictions above max raise ValueError."""
        preds = np.array([0.1, 0.5, 1.1])
        with pytest.raises(ValueError, match="above maximum"):
            check_prediction_range(preds, max_val=1.0)


class TestCheckProbabilityPredictions:
    """Tests for check_probability_predictions function."""

    def test_valid_probabilities_pass(
        self, valid_probabilities: NDArray[np.floating]
    ) -> None:
        """Valid probability predictions pass."""
        check_probability_predictions(valid_probabilities)

    def test_invalid_sum_raises(
        self, invalid_probabilities: NDArray[np.floating]
    ) -> None:
        """Probabilities not summing to 1 raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1"):
            check_probability_predictions(invalid_probabilities)

    def test_out_of_range_raises(self) -> None:
        """Probabilities outside [0, 1] raise ValueError."""
        bad_probs = np.array([[0.5, 0.6], [0.4, 0.6]])  # First row > 1
        with pytest.raises(ValueError, match="out of range|sum to 1"):
            check_probability_predictions(bad_probs)
