"""Tests for physical validation utilities.

These tests demonstrate the specification-before-code principle:
each test defines expected behavior for the validation functions.
"""

import numpy as np
import pytest
from src.validation import (
    check_density_matrix,
    check_hermitian,
    check_normalized,
    check_probabilities,
    check_unitarity,
)


class TestCheckUnitarity:
    """Tests for check_unitarity function."""

    # Happy path: valid unitary matrices
    def test_pauli_x_is_unitary(self, pauli_x: np.ndarray) -> None:
        """Pauli X is unitary (X†X = I)."""
        check_unitarity(pauli_x)  # Should not raise

    def test_pauli_y_is_unitary(self, pauli_y: np.ndarray) -> None:
        """Pauli Y is unitary."""
        check_unitarity(pauli_y)

    def test_pauli_z_is_unitary(self, pauli_z: np.ndarray) -> None:
        """Pauli Z is unitary."""
        check_unitarity(pauli_z)

    def test_hadamard_is_unitary(self, hadamard: np.ndarray) -> None:
        """Hadamard is unitary."""
        check_unitarity(hadamard)

    def test_identity_is_unitary(self, identity_2x2: np.ndarray) -> None:
        """Identity is unitary."""
        check_unitarity(identity_2x2)

    # Edge cases
    def test_1x1_unitary(self) -> None:
        """1x1 phase matrix is unitary."""
        phase = np.array([[np.exp(1j * np.pi / 4)]], dtype=complex)
        check_unitarity(phase)

    def test_larger_unitary(self) -> None:
        """Random 4x4 unitary matrix from QR decomposition."""
        # Generate random unitary via QR decomposition
        random_matrix = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        q, _ = np.linalg.qr(random_matrix)
        check_unitarity(q)

    # Invalid inputs
    def test_non_unitary_raises(self) -> None:
        """Non-unitary matrix raises ValueError."""
        non_unitary = np.array([[1, 1], [0, 1]], dtype=complex)
        with pytest.raises(ValueError, match="not unitary"):
            check_unitarity(non_unitary)

    def test_non_square_raises(self) -> None:
        """Non-square matrix raises ValueError."""
        rectangular = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        with pytest.raises(ValueError, match="must be square"):
            check_unitarity(rectangular)

    def test_1d_array_raises(self) -> None:
        """1D array raises ValueError."""
        vector = np.array([1, 0, 0], dtype=complex)
        with pytest.raises(ValueError, match="must be 2D"):
            check_unitarity(vector)

    def test_error_message_includes_name(self) -> None:
        """Error message includes matrix name if provided."""
        non_unitary = np.array([[2, 0], [0, 1]], dtype=complex)
        with pytest.raises(ValueError, match="'MyGate'"):
            check_unitarity(non_unitary, name="MyGate")


class TestCheckHermitian:
    """Tests for check_hermitian function."""

    def test_pauli_matrices_are_hermitian(
        self, pauli_x: np.ndarray, pauli_y: np.ndarray, pauli_z: np.ndarray
    ) -> None:
        """All Pauli matrices are Hermitian."""
        check_hermitian(pauli_x)
        check_hermitian(pauli_y)
        check_hermitian(pauli_z)

    def test_real_symmetric_is_hermitian(self) -> None:
        """Real symmetric matrix is Hermitian."""
        real_symmetric = np.array([[1, 2], [2, 3]], dtype=complex)
        check_hermitian(real_symmetric)

    def test_non_hermitian_raises(self) -> None:
        """Non-Hermitian matrix raises ValueError."""
        non_hermitian = np.array([[0, 1], [0, 0]], dtype=complex)
        with pytest.raises(ValueError, match="not Hermitian"):
            check_hermitian(non_hermitian)


class TestCheckNormalized:
    """Tests for check_normalized function."""

    def test_computational_basis_normalized(
        self, zero_state: np.ndarray, one_state: np.ndarray
    ) -> None:
        """Computational basis states are normalized."""
        check_normalized(zero_state)
        check_normalized(one_state)

    def test_superposition_normalized(
        self, plus_state: np.ndarray, minus_state: np.ndarray
    ) -> None:
        """Superposition states are normalized."""
        check_normalized(plus_state)
        check_normalized(minus_state)

    def test_unnormalized_raises(self) -> None:
        """Unnormalized state raises ValueError."""
        unnormalized = np.array([1, 1], dtype=complex)  # norm = sqrt(2)
        with pytest.raises(ValueError, match="not normalized"):
            check_normalized(unnormalized)

    def test_zero_vector_raises(self) -> None:
        """Zero vector raises ValueError."""
        zero_vector = np.array([0, 0], dtype=complex)
        with pytest.raises(ValueError, match="not normalized"):
            check_normalized(zero_vector)


class TestCheckDensityMatrix:
    """Tests for check_density_matrix function."""

    def test_pure_state_density_matrix(self, zero_state: np.ndarray) -> None:
        """Pure state |0⟩⟨0| is a valid density matrix."""
        rho = np.outer(zero_state, zero_state.conj())
        check_density_matrix(rho)

    def test_maximally_mixed_state(self) -> None:
        """Maximally mixed state I/2 is a valid density matrix."""
        rho = np.eye(2, dtype=complex) / 2
        check_density_matrix(rho)

    def test_non_hermitian_raises(self) -> None:
        """Non-Hermitian matrix raises ValueError."""
        non_hermitian = np.array([[0.5, 1], [0, 0.5]], dtype=complex)
        with pytest.raises(ValueError, match="not Hermitian"):
            check_density_matrix(non_hermitian)

    def test_wrong_trace_raises(self) -> None:
        """Matrix with trace != 1 raises ValueError."""
        wrong_trace = np.eye(2, dtype=complex)  # trace = 2
        with pytest.raises(ValueError, match="trace"):
            check_density_matrix(wrong_trace)

    def test_negative_eigenvalue_raises(self) -> None:
        """Matrix with negative eigenvalue raises ValueError."""
        # Diagonal matrix with negative entry
        negative_eigenvalue = np.array([[1.5, 0], [0, -0.5]], dtype=complex)
        with pytest.raises(ValueError, match="negative eigenvalue"):
            check_density_matrix(negative_eigenvalue)


class TestCheckProbabilities:
    """Tests for check_probabilities function."""

    def test_valid_probabilities(self) -> None:
        """Valid probability distribution passes."""
        probs = np.array([0.25, 0.25, 0.5])
        check_probabilities(probs)

    def test_single_probability(self) -> None:
        """Single probability of 1.0 is valid."""
        probs = np.array([1.0])
        check_probabilities(probs)

    def test_uniform_distribution(self) -> None:
        """Uniform distribution is valid."""
        probs = np.ones(10) / 10
        check_probabilities(probs)

    def test_negative_probability_raises(self) -> None:
        """Negative probability raises ValueError."""
        probs = np.array([0.5, -0.1, 0.6])
        with pytest.raises(ValueError, match="negative"):
            check_probabilities(probs)

    def test_wrong_sum_raises(self) -> None:
        """Probabilities not summing to 1 raises ValueError."""
        probs = np.array([0.3, 0.3, 0.3])  # sum = 0.9
        with pytest.raises(ValueError, match="sum to"):
            check_probabilities(probs)
