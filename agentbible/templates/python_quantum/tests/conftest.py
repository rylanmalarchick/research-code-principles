"""Pytest configuration and shared fixtures for quantum computing.

This file is automatically loaded by pytest and provides:
- Reproducibility: Fixed random seeds for deterministic tests
- Common fixtures: Quantum gates and states
- Custom markers: @pytest.mark.slow, @pytest.mark.requires_qiskit, etc.
"""

from typing import List

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
    deterministic behavior. Document any seed changes.

    Seeds:
        numpy: 42
        python random: 42 (if using random module)
    """
    np.random.seed(42)


# ============================================================================
# Single-Qubit Gates
# ============================================================================
@pytest.fixture
def pauli_x() -> NDArray[np.complexfloating]:
    """Pauli X gate matrix."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


@pytest.fixture
def pauli_y() -> NDArray[np.complexfloating]:
    """Pauli Y gate matrix."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


@pytest.fixture
def pauli_z() -> NDArray[np.complexfloating]:
    """Pauli Z gate matrix."""
    return np.array([[1, 0], [0, -1]], dtype=complex)


@pytest.fixture
def hadamard() -> NDArray[np.complexfloating]:
    """Hadamard gate matrix."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


@pytest.fixture
def identity_2x2() -> NDArray[np.complexfloating]:
    """2x2 identity matrix."""
    return np.eye(2, dtype=complex)


@pytest.fixture
def s_gate() -> NDArray[np.complexfloating]:
    """S (phase) gate matrix."""
    return np.array([[1, 0], [0, 1j]], dtype=complex)


@pytest.fixture
def t_gate() -> NDArray[np.complexfloating]:
    """T gate matrix."""
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


# ============================================================================
# Two-Qubit Gates
# ============================================================================
@pytest.fixture
def cnot() -> NDArray[np.complexfloating]:
    """CNOT gate matrix (control on qubit 0, target on qubit 1)."""
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )


@pytest.fixture
def cz() -> NDArray[np.complexfloating]:
    """CZ gate matrix."""
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
    )


@pytest.fixture
def swap_gate() -> NDArray[np.complexfloating]:
    """SWAP gate matrix."""
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )


# ============================================================================
# Quantum States
# ============================================================================
@pytest.fixture
def zero_state() -> NDArray[np.complexfloating]:
    """Qubit |0> state."""
    return np.array([1, 0], dtype=complex)


@pytest.fixture
def one_state() -> NDArray[np.complexfloating]:
    """Qubit |1> state."""
    return np.array([0, 1], dtype=complex)


@pytest.fixture
def plus_state() -> NDArray[np.complexfloating]:
    """Qubit |+> = (|0> + |1>)/sqrt(2) state."""
    return np.array([1, 1], dtype=complex) / np.sqrt(2)


@pytest.fixture
def minus_state() -> NDArray[np.complexfloating]:
    """Qubit |-> = (|0> - |1>)/sqrt(2) state."""
    return np.array([1, -1], dtype=complex) / np.sqrt(2)


@pytest.fixture
def bell_state() -> NDArray[np.complexfloating]:
    """Bell state |Phi+> = (|00> + |11>)/sqrt(2)."""
    state = np.zeros(4, dtype=complex)
    state[0] = 1 / np.sqrt(2)  # |00>
    state[3] = 1 / np.sqrt(2)  # |11>
    return state


@pytest.fixture
def ghz_state_3() -> NDArray[np.complexfloating]:
    """3-qubit GHZ state (|000> + |111>)/sqrt(2)."""
    state = np.zeros(8, dtype=complex)
    state[0] = 1 / np.sqrt(2)  # |000>
    state[7] = 1 / np.sqrt(2)  # |111>
    return state


# ============================================================================
# Density Matrices
# ============================================================================
@pytest.fixture
def maximally_mixed_1q() -> np.ndarray:
    """Maximally mixed single-qubit state I/2."""
    return np.eye(2, dtype=complex) / 2


@pytest.fixture
def pure_zero_density() -> NDArray[np.complexfloating]:
    """Pure state density matrix |0><0|."""
    zero = np.array([1, 0], dtype=complex)
    return np.outer(zero, zero.conj())


# ============================================================================
# Kraus Operators for Common Channels
# ============================================================================
@pytest.fixture
def depolarizing_kraus(request: pytest.FixtureRequest) -> List[NDArray[np.complexfloating]]:
    """Kraus operators for single-qubit depolarizing channel.

    Use with: @pytest.mark.parametrize("p", [0.0, 0.1, 0.5])
    Default p=0.1 if not parametrized.
    """
    p = getattr(request, "param", 0.1)

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    return [
        np.sqrt(1 - 3 * p / 4) * I,
        np.sqrt(p / 4) * X,
        np.sqrt(p / 4) * Y,
        np.sqrt(p / 4) * Z,
    ]


@pytest.fixture
def amplitude_damping_kraus() -> List[NDArray[np.complexfloating]]:
    """Kraus operators for amplitude damping channel with gamma=0.1."""
    gamma = 0.1
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]


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


@pytest.fixture
def loose_tolerance() -> float:
    """Looser tolerance for operations with more numerical error."""
    return 1e-6
