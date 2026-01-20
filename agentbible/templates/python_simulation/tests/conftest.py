"""Pytest configuration and shared fixtures for simulation projects.

This file is automatically loaded by pytest and provides:
- Reproducibility: Fixed random seeds for deterministic tests
- Common fixtures: Sample grids, time series, and physical quantities
- Custom markers: @pytest.mark.slow, @pytest.mark.long_running, etc.
"""

import numpy as np
import pytest
from numpy.typing import NDArray


# ============================================================================
# Reproducibility: Set seeds before each test
# ============================================================================
@pytest.fixture(autouse=True)
def set_random_seeds() -> None:
    """Set random seeds for reproducibility in all tests.

    This fixture runs automatically before every test to ensure
    deterministic behavior.
    """
    np.random.seed(42)


# ============================================================================
# Grids
# ============================================================================
@pytest.fixture
def uniform_grid_1d() -> NDArray[np.floating]:
    """Uniform 1D grid from 0 to 1."""
    return np.linspace(0, 1, 101)


@pytest.fixture
def nonuniform_grid_1d() -> NDArray[np.floating]:
    """Non-uniform 1D grid (finer near boundaries)."""
    # Chebyshev-like distribution
    n = 51
    i = np.arange(n)
    x = 0.5 * (1 - np.cos(np.pi * i / (n - 1)))
    return x


@pytest.fixture
def irregular_grid_1d() -> NDArray[np.floating]:
    """Highly irregular 1D grid (for testing validation)."""
    return np.array([0.0, 0.1, 0.11, 0.5, 0.9, 1.0])


# ============================================================================
# Time Series
# ============================================================================
@pytest.fixture
def conserved_energy() -> NDArray[np.floating]:
    """Energy time series with small fluctuations (conserved)."""
    n = 100
    base = 100.0
    noise = np.random.randn(n) * 1e-8
    return base + noise


@pytest.fixture
def drifting_energy() -> NDArray[np.floating]:
    """Energy time series with drift (non-conserved)."""
    n = 100
    base = 100.0
    drift = np.linspace(0, 1, n)  # 1% drift
    return base + drift


@pytest.fixture
def unstable_solution() -> NDArray[np.floating]:
    """Solution with NaN values (unstable)."""
    x = np.linspace(0, 1, 50)
    solution = np.sin(2 * np.pi * x)
    solution[25] = np.nan
    return solution


@pytest.fixture
def blowup_solution() -> NDArray[np.floating]:
    """Solution with exponential blowup."""
    x = np.linspace(0, 1, 50)
    return np.exp(100 * x)  # Will exceed typical max_value


# ============================================================================
# Physical Fields
# ============================================================================
@pytest.fixture
def positive_density() -> NDArray[np.floating]:
    """Valid positive density field."""
    x = np.linspace(0, 1, 50)
    return 1.0 + 0.5 * np.sin(2 * np.pi * x)


@pytest.fixture
def negative_density() -> NDArray[np.floating]:
    """Invalid density field with negative values."""
    x = np.linspace(0, 1, 50)
    return 0.5 * np.sin(2 * np.pi * x)  # Has negative values


@pytest.fixture
def velocity_field() -> NDArray[np.floating]:
    """Sample velocity field."""
    x = np.linspace(0, 1, 50)
    return np.sin(np.pi * x)


# ============================================================================
# Residuals
# ============================================================================
@pytest.fixture
def converged_residuals() -> NDArray[np.floating]:
    """Residuals showing convergence."""
    n = 20
    return 1e-3 * (0.1 ** np.linspace(0, 1, n))  # Exponential decay to ~1e-13


@pytest.fixture
def unconverged_residuals() -> NDArray[np.floating]:
    """Residuals that don't converge."""
    n = 20
    return 1e-3 * np.ones(n)  # Stuck at 1e-3


@pytest.fixture
def stalled_residuals() -> NDArray[np.floating]:
    """Residuals that stall before convergence."""
    n = 20
    res = np.zeros(n)
    res[:10] = 1e-3 * (0.5 ** np.arange(10))  # Initial decrease
    res[10:] = res[9]  # Then stalls
    return res


# ============================================================================
# CFL Parameters
# ============================================================================
@pytest.fixture
def stable_cfl_params() -> dict:
    """Parameters satisfying CFL condition."""
    return {"dt": 0.001, "dx": 0.01, "velocity": 5.0}  # CFL = 0.5


@pytest.fixture
def unstable_cfl_params() -> dict:
    """Parameters violating CFL condition."""
    return {"dt": 0.01, "dx": 0.01, "velocity": 5.0}  # CFL = 5.0


# ============================================================================
# Tolerance Constants
# ============================================================================
@pytest.fixture
def tolerance() -> float:
    """Default numerical tolerance."""
    return 1e-10


@pytest.fixture
def conservation_tolerance() -> float:
    """Tolerance for conservation checks."""
    return 1e-6
