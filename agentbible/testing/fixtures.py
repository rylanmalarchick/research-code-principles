"""Pytest fixtures for reproducible physics testing.

Provides fixtures for:
- Deterministic random seeds
- Floating-point tolerances
- Quantum-specific tolerances
"""

from __future__ import annotations

import random
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# Default seed for reproducibility
DEFAULT_SEED = 42


def deterministic_seed(
    seed: int = DEFAULT_SEED,
) -> Generator[int, None, None]:
    """Fixture factory for deterministic random seeds.

    Sets numpy and Python random seeds for reproducibility.
    Also sets PyTorch seed if available.

    Usage as a fixture:
        @pytest.fixture
        def fixed_seed():
            yield from deterministic_seed(42)

    Or use the pre-configured fixture in conftest.py:
        def test_something(deterministic_seed):
            # Seeds are set to 42
            ...

    Args:
        seed: The seed value to use

    Yields:
        The seed value that was set
    """
    # Save current states
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_state = None

    try:
        import torch

        torch_state = torch.get_rng_state()
    except ImportError:
        pass

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

    try:
        yield seed
    finally:
        # Restore states
        np.random.set_state(numpy_state)
        random.setstate(python_state)

        if torch_state is not None:
            try:
                import torch

                torch.set_rng_state(torch_state)
            except ImportError:
                pass


def tolerance() -> dict[str, float]:
    """Standard tolerance for floating-point comparisons.

    Returns a dictionary suitable for numpy.allclose() or numpy.testing functions.

    Returns:
        Dictionary with 'rtol' and 'atol' keys

    Example:
        >>> tol = tolerance()
        >>> np.allclose(a, b, **tol)
    """
    return {"rtol": 1e-10, "atol": 1e-12}


def quantum_tolerance() -> dict[str, float]:
    """Relaxed tolerance for quantum calculations.

    Quantum simulations often have higher numerical error due to:
    - Large matrix operations
    - Eigenvalue decompositions
    - Iterative algorithms

    Returns:
        Dictionary with 'rtol' and 'atol' keys

    Example:
        >>> tol = quantum_tolerance()
        >>> np.allclose(rho, rho.conj().T, **tol)  # Check Hermitian
    """
    return {"rtol": 1e-6, "atol": 1e-8}


# Pre-configured pytest fixtures
# These can be imported into conftest.py:
#
# from agentbible.testing.fixtures import (
#     deterministic_seed_fixture,
#     tolerance_fixture,
#     quantum_tolerance_fixture,
# )


def deterministic_seed_fixture() -> Generator[int, None, None]:
    """Pytest fixture for deterministic seeds.

    Import and use in conftest.py:
        from agentbible.testing.fixtures import deterministic_seed_fixture
        deterministic_seed = pytest.fixture(deterministic_seed_fixture)
    """
    yield from deterministic_seed(DEFAULT_SEED)


def tolerance_fixture() -> dict[str, float]:
    """Pytest fixture for standard tolerance.

    Import and use in conftest.py:
        from agentbible.testing.fixtures import tolerance_fixture
        tolerance = pytest.fixture(tolerance_fixture)
    """
    return tolerance()


def quantum_tolerance_fixture() -> dict[str, float]:
    """Pytest fixture for quantum tolerance.

    Import and use in conftest.py:
        from agentbible.testing.fixtures import quantum_tolerance_fixture
        quantum_tolerance = pytest.fixture(quantum_tolerance_fixture)
    """
    return quantum_tolerance()
