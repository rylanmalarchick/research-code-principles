"""Tests for ML domain validation functions and decorators."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from agentbible.domains.ml import (
    AutocorrelationWarning,
    CoverageError,
    CVStrategyWarning,
    DataLeakageError,
    ExchangeabilityWarning,
    check_coverage,
    check_exchangeability,
    check_no_leakage,
    check_temporal_autocorrelation,
    validate_cv_strategy,
    validate_no_leakage,
)


class TestCheckNoLeakage:
    """Tests for check_no_leakage function."""

    def test_valid_no_forbidden_features(self) -> None:
        """No forbidden features passes validation."""
        features = ["blh", "t2m", "rh", "pressure"]
        forbidden = {"target_lagged", "inversion_height"}
        result = check_no_leakage(features, forbidden=forbidden)
        assert result == features

    def test_returns_input_for_chaining(self) -> None:
        """Returns input unchanged for chaining."""
        features = ["blh", "t2m"]
        forbidden = {"other"}
        result = check_no_leakage(features, forbidden=forbidden)
        assert result is features

    def test_invalid_forbidden_feature_found(self) -> None:
        """Forbidden feature raises DataLeakageError."""
        features = ["blh", "t2m", "inversion_height"]
        forbidden = {"target_lagged", "inversion_height"}
        with pytest.raises(DataLeakageError) as exc_info:
            check_no_leakage(features, forbidden=forbidden)

        assert "leakage" in str(exc_info.value).lower()
        assert "inversion_height" in str(exc_info.value)

    def test_multiple_forbidden_features(self) -> None:
        """Multiple forbidden features all reported."""
        features = ["blh", "target_lagged", "inversion_height"]
        forbidden = {"target_lagged", "inversion_height"}
        with pytest.raises(DataLeakageError) as exc_info:
            check_no_leakage(features, forbidden=forbidden)

        error_str = str(exc_info.value)
        assert "target_lagged" in error_str or "inversion_height" in error_str

    def test_empty_forbidden_set(self) -> None:
        """Empty forbidden set always passes."""
        features = ["blh", "t2m"]
        result = check_no_leakage(features, forbidden=set())
        assert result == features

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning instead of raising."""
        features = ["blh", "inversion_height"]
        forbidden = {"inversion_height"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_no_leakage(features, forbidden=forbidden, strict=False)
            assert len(w) == 1
            assert "leakage" in str(w[0].message).lower()
        assert result == features

    def test_name_in_error(self) -> None:
        """Name parameter appears in error message."""
        features = ["inversion_height"]
        forbidden = {"inversion_height"}
        with pytest.raises(DataLeakageError) as exc_info:
            check_no_leakage(features, forbidden=forbidden, name="training_features")
        assert "training_features" in str(exc_info.value)


class TestCheckTemporalAutocorrelation:
    """Tests for check_temporal_autocorrelation function."""

    def test_iid_noise_passes(self) -> None:
        """IID noise has low autocorrelation and passes."""
        np.random.seed(42)
        arr = np.random.randn(100)
        # Should not warn for IID noise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_temporal_autocorrelation(arr, name="residuals")
            # Filter for our specific warning
            autocorr_warnings = [x for x in w if issubclass(x.category, AutocorrelationWarning)]
            assert len(autocorr_warnings) == 0
        assert np.array_equal(result, arr)

    def test_high_autocorrelation_warns(self) -> None:
        """Highly autocorrelated series triggers warning."""
        # Create AR(1) process with high autocorrelation
        np.random.seed(42)
        n = 100
        arr = np.zeros(n)
        arr[0] = np.random.randn()
        for i in range(1, n):
            arr[i] = 0.9 * arr[i - 1] + 0.1 * np.random.randn()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_temporal_autocorrelation(arr, name="correlated", max_autocorr=0.5)
            autocorr_warnings = [x for x in w if issubclass(x.category, AutocorrelationWarning)]
            assert len(autocorr_warnings) == 1
            assert "autocorrelation" in str(autocorr_warnings[0].message).lower()

    def test_returns_input_for_chaining(self) -> None:
        """Returns input unchanged for chaining."""
        arr = np.array([1.0, 2.0, 3.0])
        result = check_temporal_autocorrelation(arr, name="test")
        assert result is arr

    def test_short_array_skipped(self) -> None:
        """Arrays with fewer than 2 elements skip check."""
        arr = np.array([1.0])
        result = check_temporal_autocorrelation(arr, name="single")
        assert result is arr

    def test_adjustable_threshold(self) -> None:
        """max_autocorr parameter adjusts threshold."""
        # Create moderately correlated data
        np.random.seed(42)
        n = 100
        arr = np.zeros(n)
        arr[0] = np.random.randn()
        for i in range(1, n):
            arr[i] = 0.4 * arr[i - 1] + 0.6 * np.random.randn()

        # With threshold 0.5, should pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_temporal_autocorrelation(arr, max_autocorr=0.5)
            autocorr_warnings = [x for x in w if issubclass(x.category, AutocorrelationWarning)]
            assert len(autocorr_warnings) == 0

        # With threshold 0.2, should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_temporal_autocorrelation(arr, max_autocorr=0.2)
            autocorr_warnings = [x for x in w if issubclass(x.category, AutocorrelationWarning)]
            assert len(autocorr_warnings) == 1


class TestCheckCoverage:
    """Tests for check_coverage function."""

    def test_perfect_coverage_passes(self) -> None:
        """100% coverage passes for 90% target."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        result = check_coverage(y_true, y_lower, y_upper, target=0.90)
        assert result[0] is y_true

    def test_exact_target_coverage_passes(self) -> None:
        """Exact target coverage passes."""
        # 9 out of 10 in interval = 90%
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
        y_lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 0.0])
        y_upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.0])
        check_coverage(y_true, y_lower, y_upper, target=0.90, tolerance=0.01)

    def test_coverage_too_low_raises(self) -> None:
        """Coverage below target raises CoverageError."""
        # Only 50% coverage (5 out of 10 in interval)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        y_lower = np.zeros(10)
        y_upper = np.full(10, 5.5)  # Only first 5 covered

        with pytest.raises(CoverageError) as exc_info:
            check_coverage(y_true, y_lower, y_upper, target=0.90, tolerance=0.05)

        assert "below" in str(exc_info.value).lower()
        assert "coverage" in str(exc_info.value).lower()

    def test_coverage_too_high_is_ok(self) -> None:
        """Coverage above target is OK (only below is an error)."""
        # 100% coverage when target is 50% - this is fine
        y_true = np.array([1.0, 2.0, 3.0])
        y_lower = y_true - 1.0
        y_upper = y_true + 1.0

        # Should pass without error - higher coverage is OK
        result = check_coverage(y_true, y_lower, y_upper, target=0.50, tolerance=0.05)
        assert result[0] is y_true

    def test_shape_mismatch_raises(self) -> None:
        """Shape mismatch raises ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_lower = np.array([0.5, 1.5])  # Wrong shape
        y_upper = np.array([1.5, 2.5, 3.5])

        with pytest.raises(ValueError) as exc_info:
            check_coverage(y_true, y_lower, y_upper)

        assert "shape" in str(exc_info.value).lower()

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning instead of raising."""
        y_true = np.array([1.0, 100.0])  # 50% coverage
        y_lower = np.array([0.5, 0.0])
        y_upper = np.array([1.5, 10.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_coverage(y_true, y_lower, y_upper, target=0.90, strict=False)
            assert len(w) == 1

    def test_returns_tuple_for_chaining(self) -> None:
        """Returns (y_true, y_lower, y_upper) for chaining."""
        y_true = np.array([1.0, 2.0])
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        result = check_coverage(y_true, y_lower, y_upper)
        assert result == (y_true, y_lower, y_upper)


class TestCheckExchangeability:
    """Tests for check_exchangeability function."""

    def test_iid_residuals_pass(self) -> None:
        """IID residuals pass exchangeability check with relaxed thresholds."""
        np.random.seed(42)
        # Use larger sample to reduce random variation
        residuals = np.random.randn(500)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Use relaxed thresholds that IID data should easily pass
            result = check_exchangeability(residuals, name="conformal", max_autocorr=0.3, max_drift=0.3)
            exch_warnings = [x for x in w if issubclass(x.category, ExchangeabilityWarning)]
            assert len(exch_warnings) == 0
        assert np.array_equal(result, residuals)

    def test_high_autocorrelation_warns(self) -> None:
        """High autocorrelation triggers exchangeability warning."""
        np.random.seed(42)
        n = 100
        residuals = np.zeros(n)
        residuals[0] = np.random.randn()
        for i in range(1, n):
            residuals[i] = 0.8 * residuals[i - 1] + 0.2 * np.random.randn()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_exchangeability(residuals, name="correlated", max_autocorr=0.3)
            exch_warnings = [x for x in w if issubclass(x.category, ExchangeabilityWarning)]
            assert len(exch_warnings) == 1
            assert "exchangeability" in str(exch_warnings[0].message).lower()

    def test_mean_drift_warns(self) -> None:
        """Mean drift triggers exchangeability warning."""
        # First half centered at 0, second half centered at 2
        residuals = np.concatenate([
            np.random.randn(50),
            np.random.randn(50) + 2.0
        ])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_exchangeability(residuals, name="drifted", max_drift=0.1)
            exch_warnings = [x for x in w if issubclass(x.category, ExchangeabilityWarning)]
            assert len(exch_warnings) == 1
            assert "drift" in str(exch_warnings[0].message).lower()

    def test_short_array_skipped(self) -> None:
        """Arrays with fewer than 10 elements skip check."""
        residuals = np.array([1.0, 2.0, 3.0])
        result = check_exchangeability(residuals, name="short")
        assert result is residuals

    def test_returns_input_for_chaining(self) -> None:
        """Returns input unchanged for chaining."""
        np.random.seed(42)
        residuals = np.random.randn(100)
        result = check_exchangeability(residuals, name="test")
        assert result is residuals


class TestValidateNoLeakageDecorator:
    """Tests for validate_no_leakage decorator."""

    def test_valid_features_pass(self) -> None:
        """Decorated function runs with valid features."""
        FORBIDDEN = {"target_lagged", "inversion_height"}

        @validate_no_leakage(forbidden=FORBIDDEN)
        def load_data(feature_names: list[str]) -> dict:
            return {"features": feature_names}

        result = load_data(feature_names=["blh", "t2m"])
        assert result == {"features": ["blh", "t2m"]}

    def test_forbidden_features_raise(self) -> None:
        """Decorated function raises on forbidden features."""
        FORBIDDEN = {"target_lagged", "inversion_height"}

        @validate_no_leakage(forbidden=FORBIDDEN)
        def load_data(feature_names: list[str]) -> dict:
            return {"features": feature_names}

        with pytest.raises(DataLeakageError):
            load_data(feature_names=["blh", "inversion_height"])

    def test_positional_arg(self) -> None:
        """Can specify positional argument index."""
        FORBIDDEN = {"bad_feature"}

        @validate_no_leakage(forbidden=FORBIDDEN, feature_names_arg=0)
        def process(features: list[str], other: int) -> str:
            return f"processed {len(features)} features"

        result = process(["good", "features"], 42)
        assert result == "processed 2 features"

        with pytest.raises(DataLeakageError):
            process(["bad_feature"], 42)

    def test_preserves_function_metadata(self) -> None:
        """Decorator preserves function name and docstring."""
        FORBIDDEN = {"bad"}

        @validate_no_leakage(forbidden=FORBIDDEN)
        def my_function(feature_names: list[str]) -> None:
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestValidateCvStrategyDecorator:
    """Tests for validate_cv_strategy decorator."""

    def test_iid_target_no_warning(self) -> None:
        """IID target doesn't trigger warning."""
        @validate_cv_strategy
        def train_model(X, y):
            return "model"

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = train_model(X, y)
            cv_warnings = [x for x in w if issubclass(x.category, CVStrategyWarning)]
            assert len(cv_warnings) == 0
        assert result == "model"

    def test_autocorrelated_target_warns(self) -> None:
        """Highly autocorrelated target triggers warning."""
        @validate_cv_strategy(max_autocorr=0.3)
        def train_model(X, y):
            return "model"

        np.random.seed(42)
        X = np.random.randn(100, 5)
        # Create autocorrelated target
        y = np.zeros(100)
        y[0] = np.random.randn()
        for i in range(1, 100):
            y[i] = 0.8 * y[i - 1] + 0.2 * np.random.randn()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            train_model(X, y)
            cv_warnings = [x for x in w if issubclass(x.category, CVStrategyWarning)]
            assert len(cv_warnings) == 1

    def test_warn_only_false_raises(self) -> None:
        """With warn_only=False, raises ValueError."""
        @validate_cv_strategy(max_autocorr=0.1, warn_only=False)
        def train_model(X, y):
            return "model"

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.zeros(100)
        y[0] = np.random.randn()
        for i in range(1, 100):
            y[i] = 0.9 * y[i - 1] + 0.1 * np.random.randn()

        with pytest.raises(ValueError):
            train_model(X, y)

    def test_custom_target_arg(self) -> None:
        """Can specify custom target argument name."""
        @validate_cv_strategy(target_arg="target")
        def train_model(X, target):
            return "model"

        np.random.seed(42)
        X = np.random.randn(100, 5)
        target = np.random.randn(100)

        result = train_model(X, target)
        assert result == "model"

    def test_preserves_function_metadata(self) -> None:
        """Decorator preserves function name and docstring."""
        @validate_cv_strategy
        def my_training_function(X, y):
            """Train a model."""
            pass

        assert my_training_function.__name__ == "my_training_function"
        assert my_training_function.__doc__ == "Train a model."


class TestAutocorrelationComputation:
    """Tests for internal autocorrelation computation."""

    def test_perfect_positive_autocorrelation(self) -> None:
        """Constant array has autocorrelation of 0 (no variance)."""
        from agentbible.domains.ml.checks import _compute_lag1_autocorrelation

        arr = np.array([1.0, 1.0, 1.0, 1.0])
        autocorr = _compute_lag1_autocorrelation(arr)
        assert autocorr == 0.0  # Zero variance case

    def test_alternating_sequence(self) -> None:
        """Alternating sequence has negative autocorrelation."""
        from agentbible.domains.ml.checks import _compute_lag1_autocorrelation

        arr = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        autocorr = _compute_lag1_autocorrelation(arr)
        assert autocorr < -0.5

    def test_random_walk(self) -> None:
        """Random walk has positive autocorrelation."""
        from agentbible.domains.ml.checks import _compute_lag1_autocorrelation

        np.random.seed(42)
        increments = np.random.randn(100)
        walk = np.cumsum(increments)
        autocorr = _compute_lag1_autocorrelation(walk)
        assert autocorr > 0.5

    def test_short_array(self) -> None:
        """Single element array returns 0."""
        from agentbible.domains.ml.checks import _compute_lag1_autocorrelation

        arr = np.array([1.0])
        autocorr = _compute_lag1_autocorrelation(arr)
        assert autocorr == 0.0
