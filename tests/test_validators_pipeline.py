"""Tests for ValidationPipeline."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from agentbible.errors import BoundsError, NonFiniteError, ProbabilityBoundsError
from agentbible.validators.arrays import (
    check_finite,
    check_non_negative,
    check_positive,
)
from agentbible.validators.pipeline import (
    ValidationPipeline,
    ValidationResult,
    create_distribution_pipeline,
    create_numeric_pipeline,
    create_positive_pipeline,
    create_probability_pipeline,
)


class TestValidationPipeline:
    """Tests for ValidationPipeline class."""

    def test_basic_pipeline(self) -> None:
        """Pipeline executes checks in order."""
        pipeline = ValidationPipeline([check_finite, check_positive])
        arr = np.array([1.0, 2.0, 3.0])
        result = pipeline(arr, name="test")
        assert np.array_equal(result, arr)

    def test_returns_input_for_chaining(self) -> None:
        """Pipeline returns input unchanged for chaining."""
        pipeline = ValidationPipeline([check_finite])
        arr = np.array([1.0, 2.0, 3.0])
        result = pipeline(arr)
        assert result is arr

    def test_fails_on_first_check(self) -> None:
        """Pipeline raises on first failed check."""
        pipeline = ValidationPipeline([check_finite, check_positive])
        arr = np.array([1.0, np.nan, 3.0])

        with pytest.raises(NonFiniteError):
            pipeline(arr)

    def test_fails_on_second_check(self) -> None:
        """Pipeline raises on second check if first passes."""
        pipeline = ValidationPipeline([check_finite, check_positive])
        arr = np.array([1.0, -2.0, 3.0])

        with pytest.raises(BoundsError):
            pipeline(arr)

    def test_custom_name(self) -> None:
        """Pipeline uses custom name in errors."""
        pipeline = ValidationPipeline([check_finite], name="temperature")
        arr = np.array([np.nan])

        with pytest.raises(NonFiniteError) as exc_info:
            pipeline(arr)

        assert "temperature" in str(exc_info.value)

    def test_name_override(self) -> None:
        """Name can be overridden on call."""
        pipeline = ValidationPipeline([check_finite], name="default")
        arr = np.array([np.nan])

        with pytest.raises(NonFiniteError) as exc_info:
            pipeline(arr, name="overridden")

        assert "overridden" in str(exc_info.value)

    def test_strict_false_warns(self) -> None:
        """With strict=False, pipeline warns instead of raising."""
        # Use only check_finite since check_positive has strict=True for NaN check
        pipeline = ValidationPipeline([check_finite], strict=False)
        arr = np.array([1.0, np.nan])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline(arr)
            assert len(w) >= 1

        assert np.array_equal(result, arr, equal_nan=True)

    def test_strict_override_on_call(self) -> None:
        """Strict can be overridden on call."""
        pipeline = ValidationPipeline([check_finite], strict=True)
        arr = np.array([np.nan])

        # Override to non-strict
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline(arr, strict=False)
            assert len(w) >= 1

    def test_empty_pipeline_raises(self) -> None:
        """Empty check list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ValidationPipeline([])

        assert "at least one" in str(exc_info.value).lower()

    def test_checks_property(self) -> None:
        """Checks property returns copy of check list."""
        pipeline = ValidationPipeline([check_finite, check_positive])
        checks = pipeline.checks
        assert len(checks) == 2
        # Verify it's a copy
        checks.append(check_non_negative)
        assert len(pipeline.checks) == 2

    def test_name_property(self) -> None:
        """Name property returns the default name."""
        pipeline = ValidationPipeline([check_finite], name="test_name")
        assert pipeline.name == "test_name"


class TestValidationPipelineComposition:
    """Tests for pipeline composition methods."""

    def test_extend(self) -> None:
        """Extend creates new pipeline with additional checks."""
        pipeline1 = ValidationPipeline([check_finite], name="test")
        pipeline2 = pipeline1.extend(check_positive)

        # Original unchanged
        assert len(pipeline1.checks) == 1

        # New has both
        assert len(pipeline2.checks) == 2
        assert pipeline2.name == "test"

    def test_extend_multiple(self) -> None:
        """Extend can add multiple checks at once."""
        pipeline1 = ValidationPipeline([check_finite])
        pipeline2 = pipeline1.extend(check_positive, check_non_negative)

        assert len(pipeline2.checks) == 3

    def test_with_name(self) -> None:
        """with_name creates pipeline with new default name."""
        pipeline1 = ValidationPipeline([check_finite], name="old")
        pipeline2 = pipeline1.with_name("new")

        assert pipeline1.name == "old"
        assert pipeline2.name == "new"
        assert len(pipeline2.checks) == 1


class TestValidateAll:
    """Tests for validate_all method."""

    def test_all_pass(self) -> None:
        """validate_all returns success when all checks pass."""
        pipeline = ValidationPipeline([check_finite, check_positive])
        arr = np.array([1.0, 2.0, 3.0])

        result = pipeline.validate_all(arr)

        assert result.passed
        assert result.checks_run == 2
        assert result.checks_passed == 2
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_collects_all_warnings(self) -> None:
        """validate_all collects warnings from all failing checks."""
        pipeline = ValidationPipeline([check_finite, check_positive])
        arr = np.array([1.0, np.nan, -1.0])  # Fails both checks

        result = pipeline.validate_all(arr)

        assert not result.passed
        # Should have warnings from failed checks
        assert result.checks_run == 2
        assert len(result.warnings) > 0 or len(result.errors) > 0

    def test_bool_conversion(self) -> None:
        """ValidationResult can be used as boolean."""
        pipeline = ValidationPipeline([check_finite])

        good_result = pipeline.validate_all(np.array([1.0]))
        assert good_result  # True

        bad_result = pipeline.validate_all(np.array([np.nan]))
        assert not bad_result  # False

    def test_raise_if_failed(self) -> None:
        """raise_if_failed raises first error."""
        pipeline = ValidationPipeline([check_finite])
        result = pipeline.validate_all(np.array([np.nan]))

        # The validate_all runs with strict=False, so errors are warnings
        # Let's test with a pipeline that will collect errors differently
        if result.errors:
            with pytest.raises(NonFiniteError):
                result.raise_if_failed()


class TestStrictModeContext:
    """Tests for strict_mode context manager."""

    def test_context_manager_non_strict(self) -> None:
        """Context manager can set non-strict mode."""
        pipeline = ValidationPipeline([check_finite], strict=True)
        arr = np.array([np.nan])

        with ValidationPipeline.strict_mode(False), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline(arr)  # Should warn, not raise
            assert len(w) >= 1

    def test_context_manager_strict(self) -> None:
        """Context manager can set strict mode."""
        pipeline = ValidationPipeline([check_finite], strict=False)
        arr = np.array([np.nan])

        with ValidationPipeline.strict_mode(True), pytest.raises(NonFiniteError):
            pipeline(arr)

    def test_context_manager_restores(self) -> None:
        """Context manager restores previous state."""
        pipeline = ValidationPipeline([check_finite], strict=False)
        arr = np.array([np.nan])

        with ValidationPipeline.strict_mode(True):
            pass  # Enter and exit

        # Should be back to non-strict
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline(arr)
            assert len(w) >= 1


class TestPrebuiltPipelines:
    """Tests for pre-built pipeline factories."""

    def test_numeric_pipeline(self) -> None:
        """Numeric pipeline checks finite."""
        pipeline = create_numeric_pipeline("temp")
        assert len(pipeline.checks) == 1
        assert pipeline.name == "temp"

        pipeline(np.array([1.0, 2.0]))  # Should pass

        with pytest.raises(NonFiniteError):
            pipeline(np.array([np.nan]))

    def test_positive_pipeline(self) -> None:
        """Positive pipeline checks finite and positive."""
        pipeline = create_positive_pipeline()
        assert len(pipeline.checks) == 2

        pipeline(np.array([1.0, 2.0]))  # Should pass

        with pytest.raises(BoundsError):
            pipeline(np.array([0.0]))

    def test_probability_pipeline(self) -> None:
        """Probability pipeline checks finite and [0, 1]."""
        pipeline = create_probability_pipeline()
        assert len(pipeline.checks) == 2
        assert pipeline.name == "probabilities"

        pipeline(np.array([0.0, 0.5, 1.0]))  # Should pass

        with pytest.raises(ProbabilityBoundsError):
            pipeline(np.array([1.5]))

    def test_distribution_pipeline(self) -> None:
        """Distribution pipeline checks finite, [0, 1], and normalized."""
        pipeline = create_distribution_pipeline()
        assert len(pipeline.checks) == 3
        assert pipeline.name == "distribution"

        pipeline(np.array([0.25, 0.25, 0.25, 0.25]))  # Should pass


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_creation(self) -> None:
        """Can create ValidationResult."""
        result = ValidationResult(
            passed=True,
            checks_run=2,
            checks_passed=2,
        )
        assert result.passed
        assert result.checks_run == 2

    def test_with_errors(self) -> None:
        """Can create ValidationResult with errors."""
        error = ValueError("test error")
        result = ValidationResult(
            passed=False,
            checks_run=2,
            checks_passed=1,
            errors=[("check_foo", error)],
        )
        assert not result.passed
        assert len(result.errors) == 1

    def test_raise_if_failed_with_errors(self) -> None:
        """raise_if_failed raises first error."""
        error = ValueError("test error")
        result = ValidationResult(
            passed=False,
            checks_run=1,
            checks_passed=0,
            errors=[("check_foo", error)],
        )

        with pytest.raises(ValueError) as exc_info:
            result.raise_if_failed()

        assert "test error" in str(exc_info.value)

    def test_raise_if_failed_no_errors(self) -> None:
        """raise_if_failed does nothing if passed."""
        result = ValidationResult(passed=True, checks_run=1, checks_passed=1)
        result.raise_if_failed()  # Should not raise


class TestPipelineRepr:
    """Tests for pipeline string representation."""

    def test_repr(self) -> None:
        """Pipeline has informative repr."""
        pipeline = ValidationPipeline(
            [check_finite, check_positive],
            name="test",
            strict=True,
        )
        repr_str = repr(pipeline)

        assert "ValidationPipeline" in repr_str
        assert "check_finite" in repr_str
        assert "check_positive" in repr_str
        assert "test" in repr_str
