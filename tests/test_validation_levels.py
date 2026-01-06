"""Tests for conditional validation levels (debug/lite/off)."""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pytest

from agentbible.domains.quantum import (
    UnitarityError,
    validate_density_matrix,
    validate_hermitian,
    validate_unitary,
)
from agentbible.errors import NonFiniteError
from agentbible.validators import (
    ENV_VALIDATION_LEVEL,
    ValidationLevel,
    get_validation_level,
    validate_finite,
    validate_non_negative,
    validate_normalized,
    validate_positive,
    validate_probabilities,
    validate_probability,
    validate_range,
)


class TestValidationLevel:
    """Tests for ValidationLevel enum."""

    def test_from_string_debug(self) -> None:
        """Test parsing 'debug' level."""
        assert ValidationLevel.from_string("debug") == ValidationLevel.DEBUG
        assert ValidationLevel.from_string("DEBUG") == ValidationLevel.DEBUG
        assert ValidationLevel.from_string("  debug  ") == ValidationLevel.DEBUG

    def test_from_string_lite(self) -> None:
        """Test parsing 'lite' level."""
        assert ValidationLevel.from_string("lite") == ValidationLevel.LITE
        assert ValidationLevel.from_string("LITE") == ValidationLevel.LITE

    def test_from_string_off(self) -> None:
        """Test parsing 'off' level."""
        assert ValidationLevel.from_string("off") == ValidationLevel.OFF
        assert ValidationLevel.from_string("OFF") == ValidationLevel.OFF

    def test_from_string_invalid(self) -> None:
        """Test invalid level strings."""
        with pytest.raises(ValueError, match="Invalid validation level"):
            ValidationLevel.from_string("invalid")

        with pytest.raises(ValueError, match="Invalid validation level"):
            ValidationLevel.from_string("")


class TestGetValidationLevel:
    """Tests for get_validation_level function."""

    def test_explicit_level_string(self) -> None:
        """Test explicit level as string."""
        assert get_validation_level("debug") == ValidationLevel.DEBUG
        assert get_validation_level("lite") == ValidationLevel.LITE
        assert get_validation_level("off") == ValidationLevel.OFF

    def test_explicit_level_enum(self) -> None:
        """Test explicit level as enum."""
        assert get_validation_level(ValidationLevel.DEBUG) == ValidationLevel.DEBUG
        assert get_validation_level(ValidationLevel.LITE) == ValidationLevel.LITE
        assert get_validation_level(ValidationLevel.OFF) == ValidationLevel.OFF

    def test_default_is_debug(self) -> None:
        """Test that default level is DEBUG."""
        # Clear env var if set
        old_val = os.environ.pop(ENV_VALIDATION_LEVEL, None)
        try:
            assert get_validation_level(None) == ValidationLevel.DEBUG
        finally:
            if old_val is not None:
                os.environ[ENV_VALIDATION_LEVEL] = old_val

    def test_env_var_override(self) -> None:
        """Test environment variable override."""
        old_val = os.environ.get(ENV_VALIDATION_LEVEL)
        try:
            os.environ[ENV_VALIDATION_LEVEL] = "lite"
            assert get_validation_level(None) == ValidationLevel.LITE

            os.environ[ENV_VALIDATION_LEVEL] = "off"
            assert get_validation_level(None) == ValidationLevel.OFF
        finally:
            if old_val is not None:
                os.environ[ENV_VALIDATION_LEVEL] = old_val
            else:
                os.environ.pop(ENV_VALIDATION_LEVEL, None)

    def test_explicit_overrides_env(self) -> None:
        """Test that explicit level overrides env var."""
        old_val = os.environ.get(ENV_VALIDATION_LEVEL)
        try:
            os.environ[ENV_VALIDATION_LEVEL] = "off"
            # Explicit level should override env var
            assert get_validation_level("debug") == ValidationLevel.DEBUG
        finally:
            if old_val is not None:
                os.environ[ENV_VALIDATION_LEVEL] = old_val
            else:
                os.environ.pop(ENV_VALIDATION_LEVEL, None)

    def test_invalid_env_var_warns(self) -> None:
        """Test that invalid env var triggers warning."""
        old_val = os.environ.get(ENV_VALIDATION_LEVEL)
        try:
            os.environ[ENV_VALIDATION_LEVEL] = "invalid"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                level = get_validation_level(None)
                assert level == ValidationLevel.DEBUG  # Falls back to default
                assert len(w) == 1
                assert "Invalid" in str(w[0].message)
        finally:
            if old_val is not None:
                os.environ[ENV_VALIDATION_LEVEL] = old_val
            else:
                os.environ.pop(ENV_VALIDATION_LEVEL, None)


class TestValidateUnitaryLevels:
    """Tests for validate_unitary with different levels."""

    def test_debug_catches_non_unitary(self) -> None:
        """Debug mode catches non-unitary matrices."""

        @validate_unitary(level="debug")
        def make_non_unitary() -> Any:
            return np.array([[1, 1], [1, 0]]) / np.sqrt(2)

        with pytest.raises(UnitarityError):
            make_non_unitary()

    def test_debug_catches_nan(self) -> None:
        """Debug mode catches NaN values."""

        @validate_unitary(level="debug")
        def make_nan() -> Any:
            return np.array([[np.nan, 0], [0, 1]])

        with pytest.raises(NonFiniteError):
            make_nan()

    def test_lite_catches_nan(self) -> None:
        """Lite mode catches NaN values."""

        @validate_unitary(level="lite")
        def make_nan() -> Any:
            return np.array([[np.nan, 0], [0, 1]])

        with pytest.raises(NonFiniteError):
            make_nan()

    def test_lite_allows_non_unitary(self) -> None:
        """Lite mode allows non-unitary matrices (only checks NaN/Inf)."""

        @validate_unitary(level="lite")
        def make_non_unitary() -> Any:
            return np.array([[1, 1], [1, 0]]) / np.sqrt(2)

        # Should NOT raise - lite mode only checks NaN/Inf
        result = make_non_unitary()
        assert result is not None

    def test_off_allows_everything(self) -> None:
        """Off mode allows everything (with warning)."""

        @validate_unitary(level="off")
        def make_nan() -> Any:
            return np.array([[np.nan, 0], [0, 1]])

        # Should NOT raise - validation is off
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = make_nan()
            assert result is not None
            # Should have warned about validation being off
            assert len(w) >= 1
            assert "OFF" in str(w[0].message)

    def test_off_warns_once_per_function(self) -> None:
        """Off mode only warns once per function."""

        @validate_unitary(level="off")
        def make_something() -> Any:
            return np.eye(2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Call multiple times
            make_something()
            make_something()
            make_something()
            # Should only warn once
            off_warnings = [x for x in w if "OFF" in str(x.message)]
            assert len(off_warnings) == 1


class TestValidateHermitianLevels:
    """Tests for validate_hermitian with different levels."""

    def test_lite_allows_non_hermitian(self) -> None:
        """Lite mode allows non-Hermitian matrices."""

        @validate_hermitian(level="lite")
        def make_non_hermitian() -> Any:
            return np.array([[1, 1], [0, 1]])  # Not Hermitian

        result = make_non_hermitian()
        assert result is not None

    def test_lite_catches_inf(self) -> None:
        """Lite mode catches Inf values."""

        @validate_hermitian(level="lite")
        def make_inf() -> Any:
            return np.array([[np.inf, 0], [0, 1]])

        with pytest.raises(NonFiniteError):
            make_inf()


class TestValidateDensityMatrixLevels:
    """Tests for validate_density_matrix with different levels."""

    def test_lite_allows_invalid_trace(self) -> None:
        """Lite mode allows invalid trace."""

        @validate_density_matrix(level="lite")
        def make_wrong_trace() -> Any:
            return np.array([[0.5, 0], [0, 0.3]])  # Trace = 0.8

        result = make_wrong_trace()
        assert result is not None


class TestProbabilityValidatorsLevels:
    """Tests for probability validators with different levels."""

    def test_validate_probability_lite(self) -> None:
        """Lite mode allows invalid probabilities."""

        @validate_probability(level="lite")
        def make_negative_prob() -> float:
            return -0.5

        result = make_negative_prob()
        assert result == -0.5

    def test_validate_probabilities_lite(self) -> None:
        """Lite mode allows invalid probability arrays."""

        @validate_probabilities(level="lite")
        def make_negative_probs() -> Any:
            return np.array([-0.1, 0.5, 0.6])

        result = make_negative_probs()
        assert np.allclose(result, [-0.1, 0.5, 0.6])

    def test_validate_normalized_lite(self) -> None:
        """Lite mode allows non-normalized arrays."""

        @validate_normalized(level="lite")
        def make_non_normalized() -> Any:
            return np.array([0.1, 0.2, 0.3])  # Sums to 0.6

        result = make_non_normalized()
        assert np.allclose(result, [0.1, 0.2, 0.3])


class TestBoundsValidatorsLevels:
    """Tests for bounds validators with different levels."""

    def test_validate_positive_lite(self) -> None:
        """Lite mode allows non-positive values."""

        @validate_positive(level="lite")
        def make_negative() -> float:
            return -5.0

        result = make_negative()
        assert result == -5.0

    def test_validate_non_negative_lite(self) -> None:
        """Lite mode allows negative values."""

        @validate_non_negative(level="lite")
        def make_negative() -> float:
            return -5.0

        result = make_negative()
        assert result == -5.0

    def test_validate_range_lite(self) -> None:
        """Lite mode allows out-of-range values."""

        @validate_range(0.0, 1.0, level="lite")
        def make_out_of_range() -> float:
            return 5.0

        result = make_out_of_range()
        assert result == 5.0

    def test_validate_finite_off(self) -> None:
        """Off mode allows NaN/Inf values (with warning)."""

        @validate_finite(level="off")
        def make_nan() -> Any:
            return np.array([np.nan, np.inf])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = make_nan()
            assert result is not None


class TestEnvVarIntegration:
    """Tests for environment variable integration with decorators."""

    def test_env_var_affects_decorator(self) -> None:
        """Environment variable affects decorator behavior."""
        old_val = os.environ.get(ENV_VALIDATION_LEVEL)
        try:
            # Set to lite mode via env var
            os.environ[ENV_VALIDATION_LEVEL] = "lite"

            # Decorator without explicit level should use env var
            @validate_unitary
            def make_non_unitary() -> Any:
                return np.array([[1, 1], [1, 0]]) / np.sqrt(2)

            # Should NOT raise in lite mode
            result = make_non_unitary()
            assert result is not None
        finally:
            if old_val is not None:
                os.environ[ENV_VALIDATION_LEVEL] = old_val
            else:
                os.environ.pop(ENV_VALIDATION_LEVEL, None)

    def test_explicit_level_overrides_env(self) -> None:
        """Explicit level in decorator overrides env var."""
        old_val = os.environ.get(ENV_VALIDATION_LEVEL)
        try:
            # Set to off mode via env var
            os.environ[ENV_VALIDATION_LEVEL] = "off"

            # Decorator with explicit debug level should override
            @validate_unitary(level="debug")
            def make_non_unitary() -> Any:
                return np.array([[1, 1], [1, 0]]) / np.sqrt(2)

            # Should raise because explicit level is debug
            with pytest.raises(UnitarityError):
                make_non_unitary()
        finally:
            if old_val is not None:
                os.environ[ENV_VALIDATION_LEVEL] = old_val
            else:
                os.environ.pop(ENV_VALIDATION_LEVEL, None)
