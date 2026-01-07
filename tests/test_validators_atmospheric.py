"""Tests for atmospheric domain validation functions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from agentbible.domains.atmospheric import (
    BoundaryLayerHeightError,
    CloudBaseHeightError,
    CloudLayerConsistencyError,
    LiftingCondensationLevelError,
    RelativeHumidityWarning,
    TemperatureInversionError,
    check_boundary_layer_height,
    check_cloud_base_height,
    check_cloud_layer_consistency,
    check_lifting_condensation_level,
    check_relative_humidity,
    check_temperature_inversion,
)


class TestCheckCloudBaseHeight:
    """Tests for check_cloud_base_height function."""

    def test_valid_heights(self) -> None:
        """Valid cloud base heights pass validation."""
        cbh = np.array([500.0, 1000.0, 2000.0])
        result = check_cloud_base_height(cbh, name="cbh")
        assert np.array_equal(result, cbh)

    def test_returns_input_for_chaining(self) -> None:
        """Returns input unchanged for chaining."""
        cbh = np.array([500.0, 1000.0])
        result = check_cloud_base_height(cbh)
        assert result is cbh

    def test_negative_height_raises(self) -> None:
        """Negative cloud base height raises error."""
        cbh = np.array([-100.0, 500.0, 1000.0])
        with pytest.raises(CloudBaseHeightError) as exc_info:
            check_cloud_base_height(cbh, name="cbh")

        assert "below surface" in str(exc_info.value).lower()

    def test_height_above_tropopause_raises(self) -> None:
        """Height above tropopause raises error."""
        cbh = np.array([500.0, 1000.0, 15000.0])
        with pytest.raises(CloudBaseHeightError) as exc_info:
            check_cloud_base_height(cbh, name="cbh")

        assert "above tropopause" in str(exc_info.value).lower()

    def test_custom_limits(self) -> None:
        """Custom min/max limits work."""
        cbh = np.array([100.0, 500.0])

        # With custom limits
        check_cloud_base_height(cbh, min_height=50, max_height=1000)

        # Violates custom limit
        with pytest.raises(CloudBaseHeightError):
            check_cloud_base_height(cbh, min_height=200)

    def test_nan_raises(self) -> None:
        """NaN values raise error."""
        cbh = np.array([500.0, np.nan])
        with pytest.raises(CloudBaseHeightError):
            check_cloud_base_height(cbh)

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning instead of raising."""
        cbh = np.array([-100.0, 500.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_cloud_base_height(cbh, strict=False)
            assert len(w) == 1
        assert np.array_equal(result, cbh)


class TestCheckBoundaryLayerHeight:
    """Tests for check_boundary_layer_height function."""

    def test_valid_heights(self) -> None:
        """Valid BLH values pass validation."""
        blh = np.array([200.0, 800.0, 1500.0])
        result = check_boundary_layer_height(blh, name="blh")
        assert np.array_equal(result, blh)

    def test_too_low_raises(self) -> None:
        """Implausibly low BLH raises error."""
        blh = np.array([5.0, 200.0, 800.0])
        with pytest.raises(BoundaryLayerHeightError) as exc_info:
            check_boundary_layer_height(blh)

        assert "low" in str(exc_info.value).lower()

    def test_too_high_raises(self) -> None:
        """Implausibly high BLH raises error."""
        blh = np.array([200.0, 800.0, 8000.0])
        with pytest.raises(BoundaryLayerHeightError) as exc_info:
            check_boundary_layer_height(blh)

        assert "high" in str(exc_info.value).lower()

    def test_custom_limits(self) -> None:
        """Custom min/max limits work."""
        blh = np.array([100.0, 500.0])
        check_boundary_layer_height(blh, min_height=50, max_height=1000)

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning instead of raising."""
        blh = np.array([5.0, 500.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_boundary_layer_height(blh, strict=False)
            assert len(w) == 1


class TestCheckLiftingCondensationLevel:
    """Tests for check_lifting_condensation_level function."""

    def test_valid_heights(self) -> None:
        """Valid LCL values pass validation."""
        lcl = np.array([500.0, 1000.0, 2000.0])
        result = check_lifting_condensation_level(lcl, name="lcl")
        assert np.array_equal(result, lcl)

    def test_negative_raises(self) -> None:
        """Negative LCL raises error."""
        lcl = np.array([-100.0, 500.0])
        with pytest.raises(LiftingCondensationLevelError) as exc_info:
            check_lifting_condensation_level(lcl)

        assert "negative" in str(exc_info.value).lower()

    def test_too_high_raises(self) -> None:
        """Implausibly high LCL raises error."""
        lcl = np.array([500.0, 8000.0])
        with pytest.raises(LiftingCondensationLevelError):
            check_lifting_condensation_level(lcl)

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning instead of raising."""
        lcl = np.array([-100.0, 500.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_lifting_condensation_level(lcl, strict=False)
            assert len(w) == 1


class TestCheckCloudLayerConsistency:
    """Tests for check_cloud_layer_consistency function."""

    def test_valid_layers(self) -> None:
        """Valid cloud layers pass validation."""
        base = np.array([500.0, 2000.0])
        top = np.array([1500.0, 3000.0])
        result = check_cloud_layer_consistency(base, top)
        assert result == (base, top)

    def test_inverted_layer_raises(self) -> None:
        """Inverted layer (base >= top) raises error."""
        base = np.array([500.0, 2000.0])
        top = np.array([1500.0, 1800.0])  # Second layer inverted

        with pytest.raises(CloudLayerConsistencyError) as exc_info:
            check_cloud_layer_consistency(base, top)

        assert "inverted" in str(exc_info.value).lower()

    def test_equal_base_top_raises(self) -> None:
        """Base equal to top raises error."""
        base = np.array([500.0])
        top = np.array([500.0])

        with pytest.raises(CloudLayerConsistencyError):
            check_cloud_layer_consistency(base, top)

    def test_shape_mismatch_raises(self) -> None:
        """Shape mismatch raises ValueError."""
        base = np.array([500.0, 1000.0])
        top = np.array([1500.0])

        with pytest.raises(ValueError) as exc_info:
            check_cloud_layer_consistency(base, top)

        assert "shape" in str(exc_info.value).lower()

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning instead of raising."""
        base = np.array([2000.0])
        top = np.array([1000.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_cloud_layer_consistency(base, top, strict=False)
            assert len(w) == 1


class TestCheckRelativeHumidity:
    """Tests for check_relative_humidity function."""

    def test_valid_rh(self) -> None:
        """Valid RH values pass validation."""
        rh = np.array([50.0, 75.0, 90.0])
        result = check_relative_humidity(rh, name="rh")
        assert np.array_equal(result, rh)

    def test_boundary_values(self) -> None:
        """0% and 100% are valid."""
        rh = np.array([0.0, 100.0])
        check_relative_humidity(rh)

    def test_negative_raises(self) -> None:
        """Negative RH raises error."""
        rh = np.array([-5.0, 50.0])
        with pytest.raises(ValueError) as exc_info:
            check_relative_humidity(rh)

        assert "negative" in str(exc_info.value).lower()

    def test_above_100_raises(self) -> None:
        """RH above 100% raises error."""
        rh = np.array([50.0, 110.0])
        with pytest.raises(ValueError) as exc_info:
            check_relative_humidity(rh)

        assert "exceeds" in str(exc_info.value).lower()

    def test_supersaturation_allowed(self) -> None:
        """Supersaturation allowed when enabled."""
        rh = np.array([50.0, 103.0])
        check_relative_humidity(rh, allow_supersaturation=True)

        # But not too much
        rh_high = np.array([50.0, 110.0])
        with pytest.raises(ValueError):
            check_relative_humidity(rh_high, allow_supersaturation=True)

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning instead of raising."""
        rh = np.array([-5.0, 50.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_relative_humidity(rh, strict=False)
            rh_warnings = [x for x in w if issubclass(x.category, RelativeHumidityWarning)]
            assert len(rh_warnings) == 1


class TestCheckTemperatureInversion:
    """Tests for check_temperature_inversion function."""

    def test_valid_inversion(self) -> None:
        """Valid inversion passes validation."""
        base_h = np.array([100.0])
        top_h = np.array([500.0])
        base_t = np.array([280.0])
        top_t = np.array([285.0])

        result = check_temperature_inversion(base_h, top_h, base_t, top_t)
        assert result == (base_h, top_h, base_t, top_t)

    def test_height_inverted_raises(self) -> None:
        """Base height >= top height raises error."""
        base_h = np.array([500.0])
        top_h = np.array([100.0])
        base_t = np.array([280.0])
        top_t = np.array([285.0])

        with pytest.raises(TemperatureInversionError) as exc_info:
            check_temperature_inversion(base_h, top_h, base_t, top_t)

        assert "base >= top" in str(exc_info.value).lower()

    def test_temperature_not_inverted_raises(self) -> None:
        """Temperature decreasing with height raises error."""
        base_h = np.array([100.0])
        top_h = np.array([500.0])
        base_t = np.array([285.0])  # Warmer at base
        top_t = np.array([280.0])  # Cooler at top = not an inversion

        with pytest.raises(TemperatureInversionError) as exc_info:
            check_temperature_inversion(base_h, top_h, base_t, top_t)

        assert "decreasing" in str(exc_info.value).lower()

    def test_too_strong_raises(self) -> None:
        """Unusually strong inversion raises error."""
        base_h = np.array([100.0])
        top_h = np.array([500.0])
        base_t = np.array([260.0])
        top_t = np.array([300.0])  # 40 K difference

        with pytest.raises(TemperatureInversionError) as exc_info:
            check_temperature_inversion(base_h, top_h, base_t, top_t)

        assert "strong" in str(exc_info.value).lower()

    def test_custom_max_strength(self) -> None:
        """Custom max strength works."""
        base_h = np.array([100.0])
        top_h = np.array([500.0])
        base_t = np.array([270.0])
        top_t = np.array([290.0])  # 20 K difference

        # Should pass with higher limit
        check_temperature_inversion(base_h, top_h, base_t, top_t, max_strength=25.0)

        # Should fail with lower limit
        with pytest.raises(TemperatureInversionError):
            check_temperature_inversion(base_h, top_h, base_t, top_t, max_strength=15.0)

    def test_strict_false_warns(self) -> None:
        """With strict=False, issues warning instead of raising."""
        base_h = np.array([500.0])
        top_h = np.array([100.0])
        base_t = np.array([280.0])
        top_t = np.array([285.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_temperature_inversion(base_h, top_h, base_t, top_t, strict=False)
            assert len(w) == 1


class TestAtmosphericErrorAttributes:
    """Tests for atmospheric error classes."""

    def test_cloud_base_height_error_has_reference(self) -> None:
        """CloudBaseHeightError has REFERENCE and GUIDANCE."""
        assert hasattr(CloudBaseHeightError, "REFERENCE")
        assert hasattr(CloudBaseHeightError, "GUIDANCE")
        assert "WMO" in CloudBaseHeightError.REFERENCE or "Stull" in CloudBaseHeightError.REFERENCE

    def test_boundary_layer_height_error_has_reference(self) -> None:
        """BoundaryLayerHeightError has REFERENCE and GUIDANCE."""
        assert hasattr(BoundaryLayerHeightError, "REFERENCE")
        assert "Stull" in BoundaryLayerHeightError.REFERENCE

    def test_lifting_condensation_level_error_has_reference(self) -> None:
        """LiftingCondensationLevelError has REFERENCE and GUIDANCE."""
        assert hasattr(LiftingCondensationLevelError, "REFERENCE")
        assert "Bolton" in LiftingCondensationLevelError.REFERENCE or "Romps" in LiftingCondensationLevelError.REFERENCE
