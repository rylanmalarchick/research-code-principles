"""Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Reproducibility: Fixed random seeds for deterministic tests
- Common fixtures: Shared test data and objects
- Custom markers: @pytest.mark.slow, @pytest.mark.deterministic, etc.
"""

import numpy as np
import pytest


# ============================================================================
# Reproducibility: Set seeds before each test
# ============================================================================
@pytest.fixture(autouse=True)
def set_random_seeds() -> None:
    """Set random seeds for reproducibility in all tests.

    This fixture runs automatically before every test to ensure
    deterministic behavior. Document any seed changes.

    Seeds:
        numpy: 42
        python random: 42 (if using random module)
    """
    np.random.seed(42)


# ============================================================================
# Common Test Fixtures
# ============================================================================
@pytest.fixture
def pauli_x() -> np.ndarray:
    """Pauli X gate matrix."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


@pytest.fixture
def pauli_y() -> np.ndarray:
    """Pauli Y gate matrix."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


@pytest.fixture
def pauli_z() -> np.ndarray:
    """Pauli Z gate matrix."""
    return np.array([[1, 0], [0, -1]], dtype=complex)


@pytest.fixture
def hadamard() -> np.ndarray:
    """Hadamard gate matrix."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


@pytest.fixture
def identity_2x2() -> np.ndarray:
    """2x2 identity matrix."""
    return np.eye(2, dtype=complex)


@pytest.fixture
def zero_state() -> np.ndarray:
    """Qubit |0⟩ state."""
    return np.array([1, 0], dtype=complex)


@pytest.fixture
def one_state() -> np.ndarray:
    """Qubit |1⟩ state."""
    return np.array([0, 1], dtype=complex)


@pytest.fixture
def plus_state() -> np.ndarray:
    """Qubit |+⟩ = (|0⟩ + |1⟩)/√2 state."""
    return np.array([1, 1], dtype=complex) / np.sqrt(2)


@pytest.fixture
def minus_state() -> np.ndarray:
    """Qubit |-⟩ = (|0⟩ - |1⟩)/√2 state."""
    return np.array([1, -1], dtype=complex) / np.sqrt(2)


# ============================================================================
# Tolerance Constants
# ============================================================================
@pytest.fixture
def tolerance() -> float:
    """Default numerical tolerance for floating-point comparisons.

    Source: Machine epsilon for float64 is ~2.2e-16; 1e-10 provides
    margin for accumulated numerical error in typical operations.
    """
    return 1e-10
