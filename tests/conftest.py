"""Pytest configuration and shared fixtures for agentbible tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def identity_2x2() -> np.ndarray:
    """2x2 identity matrix."""
    return np.eye(2, dtype=complex)


@pytest.fixture
def identity_4x4() -> np.ndarray:
    """4x4 identity matrix."""
    return np.eye(4, dtype=complex)


@pytest.fixture
def hadamard_gate() -> np.ndarray:
    """Hadamard gate - a valid 2x2 unitary matrix."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


@pytest.fixture
def pauli_x() -> np.ndarray:
    """Pauli X gate - both unitary and Hermitian."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


@pytest.fixture
def pauli_y() -> np.ndarray:
    """Pauli Y gate - both unitary and Hermitian."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


@pytest.fixture
def pauli_z() -> np.ndarray:
    """Pauli Z gate - both unitary and Hermitian."""
    return np.array([[1, 0], [0, -1]], dtype=complex)


@pytest.fixture
def non_unitary_matrix() -> np.ndarray:
    """A matrix that is not unitary."""
    return np.array([[1, 1], [0, 1]], dtype=complex)


@pytest.fixture
def non_hermitian_matrix() -> np.ndarray:
    """A matrix that is not Hermitian."""
    return np.array([[1, 2], [3, 4]], dtype=complex)


@pytest.fixture
def pure_state_density_matrix() -> np.ndarray:
    """Valid density matrix for |0> state."""
    return np.array([[1, 0], [0, 0]], dtype=complex)


@pytest.fixture
def mixed_state_density_matrix() -> np.ndarray:
    """Valid density matrix for maximally mixed state."""
    return np.eye(2, dtype=complex) / 2


@pytest.fixture
def invalid_trace_density_matrix() -> np.ndarray:
    """Density matrix with wrong trace."""
    return np.eye(2, dtype=complex)  # trace = 2, not 1


@pytest.fixture
def negative_eigenvalue_matrix() -> np.ndarray:
    """Hermitian matrix with negative eigenvalue (not positive semi-definite)."""
    return np.array([[0.5, 0.6], [0.6, 0.5]], dtype=complex)


@pytest.fixture
def probability_array() -> np.ndarray:
    """Valid probability array (all values in [0, 1])."""
    return np.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def normalized_array() -> np.ndarray:
    """Array that sums to 1."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def non_normalized_array() -> np.ndarray:
    """Array that does not sum to 1."""
    return np.array([0.1, 0.2, 0.3, 0.5])


@pytest.fixture
def array_with_nan() -> np.ndarray:
    """Array containing NaN."""
    return np.array([1.0, np.nan, 3.0])


@pytest.fixture
def array_with_inf() -> np.ndarray:
    """Array containing Inf."""
    return np.array([1.0, np.inf, 3.0])
