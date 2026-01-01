"""Tests for testing module (fixtures and decorators)."""

from __future__ import annotations

import numpy as np
import pytest

from agentbible.testing.decorators import get_available_checks, physics_test
from agentbible.testing.fixtures import (
    deterministic_seed,
    quantum_tolerance,
    tolerance,
)
from agentbible.validators import ValidationError


class TestDeterministicSeed:
    """Tests for deterministic_seed fixture."""

    def test_sets_numpy_seed(self) -> None:
        """Sets numpy random seed."""
        gen = deterministic_seed(seed=12345)
        next(gen)  # Consume generator to set seed

        val1 = np.random.rand()

        # Reset and get same value
        gen2 = deterministic_seed(seed=12345)
        next(gen2)
        val2 = np.random.rand()

        assert val1 == val2

    def test_yields_seed_value(self) -> None:
        """Yields the seed value that was set."""
        gen = deterministic_seed(seed=42)
        seed = next(gen)
        assert seed == 42

    def test_default_seed_is_42(self) -> None:
        """Default seed is 42."""
        gen = deterministic_seed()
        seed = next(gen)
        assert seed == 42


class TestTolerance:
    """Tests for tolerance fixtures."""

    def test_tolerance_returns_dict(self) -> None:
        """tolerance() returns dict with rtol and atol."""
        tol = tolerance()
        assert "rtol" in tol
        assert "atol" in tol

    def test_tolerance_values(self) -> None:
        """tolerance() has strict values."""
        tol = tolerance()
        assert tol["rtol"] == 1e-10
        assert tol["atol"] == 1e-12

    def test_quantum_tolerance_returns_dict(self) -> None:
        """quantum_tolerance() returns dict with rtol and atol."""
        tol = quantum_tolerance()
        assert "rtol" in tol
        assert "atol" in tol

    def test_quantum_tolerance_is_relaxed(self) -> None:
        """quantum_tolerance() is more relaxed than tolerance()."""
        strict = tolerance()
        relaxed = quantum_tolerance()
        assert relaxed["rtol"] > strict["rtol"]
        assert relaxed["atol"] > strict["atol"]

    def test_tolerance_works_with_allclose(self) -> None:
        """Tolerance dict works with np.allclose."""
        tol = tolerance()
        a = np.array([1.0, 2.0, 3.0])
        b = a + 1e-13  # Within tolerance
        assert np.allclose(a, b, **tol)


class TestPhysicsTest:
    """Tests for @physics_test decorator."""

    def test_unitarity_check_passes(self) -> None:
        """Unitarity check passes for unitary matrix."""

        @physics_test(checks=["unitarity"])
        def get_hadamard() -> np.ndarray:
            return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

        result = get_hadamard()
        assert result.shape == (2, 2)

    def test_unitarity_check_fails(self) -> None:
        """Unitarity check fails for non-unitary matrix."""

        @physics_test(checks=["unitarity"])
        def get_non_unitary() -> np.ndarray:
            return np.array([[1, 1], [0, 1]], dtype=complex)

        with pytest.raises(ValidationError, match="unitarity"):
            get_non_unitary()

    def test_hermiticity_check_passes(self) -> None:
        """Hermiticity check passes for Hermitian matrix."""

        @physics_test(checks=["hermiticity"])
        def get_hermitian() -> np.ndarray:
            return np.array([[1, 2 + 1j], [2 - 1j, 3]], dtype=complex)

        result = get_hermitian()
        assert result.shape == (2, 2)

    def test_hermiticity_check_fails(self) -> None:
        """Hermiticity check fails for non-Hermitian matrix."""

        @physics_test(checks=["hermiticity"])
        def get_non_hermitian() -> np.ndarray:
            return np.array([[1, 2], [3, 4]], dtype=complex)

        with pytest.raises(ValidationError, match="hermiticity"):
            get_non_hermitian()

    def test_trace_one_check_passes(self) -> None:
        """Trace check passes for trace-1 matrix."""

        @physics_test(checks=["trace_one"])
        def get_density_matrix() -> np.ndarray:
            return np.array([[0.5, 0], [0, 0.5]], dtype=complex)

        result = get_density_matrix()
        assert np.isclose(np.trace(result), 1.0)

    def test_trace_one_check_fails(self) -> None:
        """Trace check fails for wrong trace."""

        @physics_test(checks=["trace_one"])
        def get_wrong_trace() -> np.ndarray:
            return np.eye(2, dtype=complex)  # trace = 2

        with pytest.raises(ValidationError, match="trace"):
            get_wrong_trace()

    def test_positive_semidefinite_passes(self) -> None:
        """Positive semi-definite check passes for valid matrix."""

        @physics_test(checks=["positive_semidefinite"])
        def get_positive() -> np.ndarray:
            return np.array([[1, 0], [0, 0]], dtype=complex)

        result = get_positive()
        assert result.shape == (2, 2)

    def test_positive_semidefinite_fails(self) -> None:
        """Positive semi-definite check fails for negative eigenvalue."""

        @physics_test(checks=["positive_semidefinite"])
        def get_negative_eigenvalue() -> np.ndarray:
            return np.array([[0.5, 0.6], [0.6, 0.5]], dtype=complex)

        with pytest.raises(ValidationError, match="negative eigenvalue"):
            get_negative_eigenvalue()

    def test_normalization_check_passes(self) -> None:
        """Normalization check passes for unit vector."""

        @physics_test(checks=["normalization"])
        def get_normalized() -> np.ndarray:
            return np.array([1, 0, 0], dtype=complex)

        result = get_normalized()
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_normalization_check_fails(self) -> None:
        """Normalization check fails for non-unit vector."""

        @physics_test(checks=["normalization"])
        def get_unnormalized() -> np.ndarray:
            return np.array([1, 1, 1], dtype=complex)

        with pytest.raises(ValidationError, match="norm"):
            get_unnormalized()

    def test_probability_check_passes(self) -> None:
        """Probability check passes for valid probabilities."""

        @physics_test(checks=["probability"])
        def get_probs() -> np.ndarray:
            return np.array([0.25, 0.25, 0.5])

        result = get_probs()
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_probability_check_fails_negative(self) -> None:
        """Probability check fails for negative values."""

        @physics_test(checks=["probability"])
        def get_negative_prob() -> np.ndarray:
            return np.array([0.5, -0.1, 0.6])

        with pytest.raises(ValidationError, match="probability"):
            get_negative_prob()

    def test_multiple_checks(self) -> None:
        """Multiple checks can be applied."""

        @physics_test(checks=["hermiticity", "trace_one", "positive_semidefinite"])
        def get_valid_density_matrix() -> np.ndarray:
            return np.array([[0.5, 0], [0, 0.5]], dtype=complex)

        result = get_valid_density_matrix()
        assert result.shape == (2, 2)

    def test_unknown_check_raises(self) -> None:
        """Unknown check name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):

            @physics_test(checks=["unknown_check"])
            def func() -> np.ndarray:
                return np.eye(2)

    def test_no_checks_is_passthrough(self) -> None:
        """No checks means function is passthrough."""

        @physics_test(checks=[])
        def get_anything() -> np.ndarray:
            return np.array([[1, 2], [3, 4]])  # Not unitary, not Hermitian

        result = get_anything()
        assert result.shape == (2, 2)

    def test_non_array_return_is_passthrough(self) -> None:
        """Non-array return values are not validated."""

        @physics_test(checks=["unitarity"])
        def get_non_array() -> int:
            return 42

        result = get_non_array()
        assert result == 42

    def test_custom_tolerance(self) -> None:
        """Custom tolerance can be specified."""
        # Matrix that's almost unitary but not quite
        almost_unitary = np.array([[1, 0], [0, 1.001]], dtype=complex)

        @physics_test(checks=["unitarity"], rtol=1e-10)
        def strict_unitary() -> np.ndarray:
            return almost_unitary

        @physics_test(checks=["unitarity"], rtol=0.01)
        def relaxed_unitary() -> np.ndarray:
            return almost_unitary

        with pytest.raises(ValidationError):
            strict_unitary()

        # Relaxed should pass
        result = relaxed_unitary()
        assert result.shape == (2, 2)


class TestGetAvailableChecks:
    """Tests for get_available_checks()."""

    def test_returns_list(self) -> None:
        """Returns a list of strings."""
        checks = get_available_checks()
        assert isinstance(checks, list)
        assert all(isinstance(c, str) for c in checks)

    def test_contains_expected_checks(self) -> None:
        """Contains all expected physics checks."""
        checks = get_available_checks()
        expected = [
            "unitarity",
            "hermiticity",
            "trace_one",
            "positive_semidefinite",
            "normalization",
            "probability",
        ]
        for check in expected:
            assert check in checks
