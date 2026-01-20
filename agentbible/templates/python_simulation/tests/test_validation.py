"""Tests for simulation validation utilities.

These tests demonstrate the specification-before-code principle:
each test defines expected behavior for the validation functions.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.validation import (
    check_cfl_condition,
    check_conservation,
    check_convergence,
    check_grid_spacing,
    check_physical_bounds,
    check_stability,
    check_time_step,
)


class TestCheckConservation:
    """Tests for check_conservation function."""

    def test_conserved_quantity_passes(
        self, conserved_energy: NDArray[np.floating]
    ) -> None:
        """Conserved energy passes validation."""
        check_conservation(conserved_energy, rtol=1e-6)  # Should not raise

    def test_drifting_quantity_fails(
        self, drifting_energy: NDArray[np.floating]
    ) -> None:
        """Drifting energy fails validation."""
        with pytest.raises(ValueError, match="Conservation.*violated"):
            check_conservation(drifting_energy, rtol=1e-4)

    def test_single_value_passes(self) -> None:
        """Single value always passes (nothing to compare)."""
        check_conservation(np.array([100.0]))  # Should not raise

    def test_error_includes_name(self, drifting_energy: NDArray[np.floating]) -> None:
        """Error message includes name if provided."""
        with pytest.raises(ValueError, match="'energy'"):
            check_conservation(drifting_energy, rtol=1e-4, name="energy")


class TestCheckStability:
    """Tests for check_stability function."""

    def test_stable_solution_passes(self, positive_density: NDArray[np.floating]) -> None:
        """Stable solution passes validation."""
        check_stability(positive_density)  # Should not raise

    def test_nan_fails(self, unstable_solution: NDArray[np.floating]) -> None:
        """Solution with NaN fails validation."""
        with pytest.raises(ValueError, match="NaN"):
            check_stability(unstable_solution)

    def test_blowup_fails(self, blowup_solution: NDArray[np.floating]) -> None:
        """Solution with blowup fails validation."""
        with pytest.raises(ValueError, match="blowup"):
            check_stability(blowup_solution, max_value=1e10)

    def test_custom_max_value(self) -> None:
        """Custom max_value is respected."""
        data = np.array([1e5, 1e6, 1e7])
        check_stability(data, max_value=1e8)  # Should pass
        with pytest.raises(ValueError):
            check_stability(data, max_value=1e6)  # Should fail


class TestCheckConvergence:
    """Tests for check_convergence function."""

    def test_converged_passes(self, converged_residuals: NDArray[np.floating]) -> None:
        """Converged residuals pass validation."""
        check_convergence(converged_residuals, tol=1e-10)  # Should not raise

    def test_unconverged_fails(
        self, unconverged_residuals: NDArray[np.floating]
    ) -> None:
        """Unconverged residuals fail validation."""
        with pytest.raises(ValueError, match="did not converge"):
            check_convergence(unconverged_residuals, tol=1e-10)

    def test_empty_residuals_fails(self) -> None:
        """Empty residuals raise error."""
        with pytest.raises(ValueError, match="No residuals"):
            check_convergence(np.array([]))


class TestCheckPhysicalBounds:
    """Tests for check_physical_bounds function."""

    def test_positive_density_passes(
        self, positive_density: NDArray[np.floating]
    ) -> None:
        """Positive density passes min_val=0 check."""
        check_physical_bounds(positive_density, min_val=0)  # Should not raise

    def test_negative_density_fails(
        self, negative_density: NDArray[np.floating]
    ) -> None:
        """Negative density fails min_val=0 check."""
        with pytest.raises(ValueError, match="below minimum"):
            check_physical_bounds(negative_density, min_val=0)

    def test_max_bound_works(self) -> None:
        """Maximum bound is enforced."""
        data = np.array([0.5, 1.0, 1.5])
        with pytest.raises(ValueError, match="above maximum"):
            check_physical_bounds(data, max_val=1.0)


class TestCheckCFLCondition:
    """Tests for check_cfl_condition function."""

    def test_stable_cfl_passes(self, stable_cfl_params: dict) -> None:
        """Parameters within CFL limit pass."""
        cfl = check_cfl_condition(**stable_cfl_params)
        assert cfl < 1.0

    def test_unstable_cfl_fails(self, unstable_cfl_params: dict) -> None:
        """Parameters exceeding CFL limit fail."""
        with pytest.raises(ValueError, match="CFL condition.*violated"):
            check_cfl_condition(**unstable_cfl_params)

    def test_returns_cfl_number(self, stable_cfl_params: dict) -> None:
        """Function returns computed CFL number."""
        cfl = check_cfl_condition(**stable_cfl_params)
        expected = 5.0 * 0.001 / 0.01  # 0.5
        assert abs(cfl - expected) < 1e-10

    def test_velocity_array(self) -> None:
        """Works with velocity array."""
        velocities = np.array([1.0, 2.0, 3.0])
        cfl = check_cfl_condition(dt=0.01, dx=0.1, velocity=velocities)
        assert abs(cfl - 0.3) < 1e-10  # max(v) * dt / dx


class TestCheckTimeStep:
    """Tests for check_time_step function."""

    def test_valid_time_step_passes(self) -> None:
        """Valid time step passes."""
        check_time_step(dt=0.001, t_final=1.0)  # Should not raise

    def test_negative_dt_fails(self) -> None:
        """Negative dt fails."""
        with pytest.raises(ValueError, match="must be positive"):
            check_time_step(dt=-0.001, t_final=1.0)

    def test_dt_exceeds_tfinal_fails(self) -> None:
        """dt > t_final fails."""
        with pytest.raises(ValueError, match="exceeds final time"):
            check_time_step(dt=2.0, t_final=1.0)

    def test_too_many_steps_fails(self) -> None:
        """Too many steps raises warning."""
        with pytest.raises(ValueError, match="steps"):
            check_time_step(dt=1e-12, t_final=1.0)


class TestCheckGridSpacing:
    """Tests for check_grid_spacing function."""

    def test_uniform_grid_passes(self, uniform_grid_1d: NDArray[np.floating]) -> None:
        """Uniform grid passes spacing check."""
        check_grid_spacing(uniform_grid_1d)  # Should not raise

    def test_nonuniform_grid_passes(
        self, nonuniform_grid_1d: NDArray[np.floating]
    ) -> None:
        """Smoothly varying grid passes with default tolerance."""
        check_grid_spacing(nonuniform_grid_1d, min_ratio=0.5, max_ratio=2.0)

    def test_irregular_grid_fails(
        self, irregular_grid_1d: NDArray[np.floating]
    ) -> None:
        """Highly irregular grid fails."""
        with pytest.raises(ValueError, match="irregular"):
            check_grid_spacing(irregular_grid_1d, min_ratio=0.5, max_ratio=2.0)

    def test_non_increasing_fails(self) -> None:
        """Non-increasing grid fails."""
        grid = np.array([0.0, 0.5, 0.4, 1.0])  # Not increasing
        with pytest.raises(ValueError, match="strictly increasing"):
            check_grid_spacing(grid)
