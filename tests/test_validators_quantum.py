"""Tests for quantum validators."""

from __future__ import annotations

import numpy as np
import pytest

from agentbible.domains.quantum import (
    validate_density_matrix,
    validate_hermitian,
    validate_unitary,
)
from agentbible.errors import ValidationError


class TestValidateUnitary:
    """Tests for @validate_unitary decorator."""

    def test_valid_identity(self, identity_2x2: np.ndarray) -> None:
        """Identity matrix is unitary."""

        @validate_unitary
        def make_gate() -> np.ndarray:
            return identity_2x2

        result = make_gate()
        assert np.allclose(result, identity_2x2)

    def test_valid_hadamard(self, hadamard_gate: np.ndarray) -> None:
        """Hadamard gate is unitary."""

        @validate_unitary
        def make_gate() -> np.ndarray:
            return hadamard_gate

        result = make_gate()
        assert np.allclose(result, hadamard_gate)

    def test_valid_pauli_gates(
        self,
        pauli_x: np.ndarray,
        pauli_y: np.ndarray,
        pauli_z: np.ndarray,
    ) -> None:
        """Pauli gates are unitary."""

        @validate_unitary
        def make_x() -> np.ndarray:
            return pauli_x

        @validate_unitary
        def make_y() -> np.ndarray:
            return pauli_y

        @validate_unitary
        def make_z() -> np.ndarray:
            return pauli_z

        assert make_x() is not None
        assert make_y() is not None
        assert make_z() is not None

    def test_invalid_non_unitary(self, non_unitary_matrix: np.ndarray) -> None:
        """Non-unitary matrix raises ValidationError."""

        @validate_unitary
        def make_gate() -> np.ndarray:
            return non_unitary_matrix

        with pytest.raises(ValidationError) as exc_info:
            make_gate()

        assert "not unitary" in str(exc_info.value).lower()
        assert "U†U = I" in str(exc_info.value)

    def test_invalid_non_square(self) -> None:
        """Non-square matrix raises ValidationError."""

        @validate_unitary
        def make_gate() -> np.ndarray:
            return np.array([[1, 0, 0], [0, 1, 0]])

        with pytest.raises(ValidationError) as exc_info:
            make_gate()

        assert "Square matrix" in str(exc_info.value)

    def test_custom_tolerance(self) -> None:
        """Custom tolerance is respected."""
        # Create a matrix that's "almost" unitary
        almost_unitary = np.eye(2, dtype=complex)
        almost_unitary[0, 0] = 1.0001  # Slightly off

        @validate_unitary(rtol=1e-2)  # Loose tolerance
        def make_loose() -> np.ndarray:
            return almost_unitary

        @validate_unitary(rtol=1e-6)  # Strict tolerance
        def make_strict() -> np.ndarray:
            return almost_unitary

        # Loose tolerance should pass
        make_loose()

        # Strict tolerance should fail
        with pytest.raises(ValidationError):
            make_strict()

    def test_decorator_preserves_function_name(self) -> None:
        """Decorator preserves function metadata."""

        @validate_unitary
        def my_special_gate() -> np.ndarray:
            """My docstring."""
            return np.eye(2)

        assert my_special_gate.__name__ == "my_special_gate"
        assert my_special_gate.__doc__ == "My docstring."

    def test_error_includes_function_name(self, non_unitary_matrix: np.ndarray) -> None:
        """Error message includes the function name."""

        @validate_unitary
        def buggy_gate() -> np.ndarray:
            return non_unitary_matrix

        with pytest.raises(ValidationError) as exc_info:
            buggy_gate()

        assert "buggy_gate" in str(exc_info.value)


class TestValidateHermitian:
    """Tests for @validate_hermitian decorator."""

    def test_valid_identity(self, identity_2x2: np.ndarray) -> None:
        """Identity matrix is Hermitian."""

        @validate_hermitian
        def make_matrix() -> np.ndarray:
            return identity_2x2

        result = make_matrix()
        assert np.allclose(result, identity_2x2)

    def test_valid_pauli_gates(
        self,
        pauli_x: np.ndarray,
        pauli_y: np.ndarray,
        pauli_z: np.ndarray,
    ) -> None:
        """Pauli gates are Hermitian."""

        @validate_hermitian
        def make_x() -> np.ndarray:
            return pauli_x

        @validate_hermitian
        def make_y() -> np.ndarray:
            return pauli_y

        @validate_hermitian
        def make_z() -> np.ndarray:
            return pauli_z

        assert make_x() is not None
        assert make_y() is not None
        assert make_z() is not None

    def test_valid_real_symmetric(self) -> None:
        """Real symmetric matrix is Hermitian."""

        @validate_hermitian
        def make_matrix() -> np.ndarray:
            return np.array([[1, 2], [2, 3]], dtype=float)

        make_matrix()

    def test_invalid_non_hermitian(self, non_hermitian_matrix: np.ndarray) -> None:
        """Non-Hermitian matrix raises ValidationError."""

        @validate_hermitian
        def make_matrix() -> np.ndarray:
            return non_hermitian_matrix

        with pytest.raises(ValidationError) as exc_info:
            make_matrix()

        assert "not hermitian" in str(exc_info.value).lower()
        assert "H = H†" in str(exc_info.value)

    def test_invalid_non_square(self) -> None:
        """Non-square matrix raises ValidationError."""

        @validate_hermitian
        def make_matrix() -> np.ndarray:
            return np.array([[1, 0, 0], [0, 1, 0]])

        with pytest.raises(ValidationError) as exc_info:
            make_matrix()

        assert "Square matrix" in str(exc_info.value)

    def test_decorator_with_parentheses(self, identity_2x2: np.ndarray) -> None:
        """Decorator works with explicit parentheses."""

        @validate_hermitian()
        def make_matrix() -> np.ndarray:
            return identity_2x2

        make_matrix()

    def test_custom_tolerance(self) -> None:
        """Custom tolerance is respected."""
        almost_hermitian = np.array([[1, 0.0001], [0, 1]], dtype=complex)

        @validate_hermitian(atol=1e-3)  # Loose
        def make_loose() -> np.ndarray:
            return almost_hermitian

        @validate_hermitian(atol=1e-6)  # Strict
        def make_strict() -> np.ndarray:
            return almost_hermitian

        make_loose()

        with pytest.raises(ValidationError):
            make_strict()


class TestValidateDensityMatrix:
    """Tests for @validate_density_matrix decorator."""

    def test_valid_pure_state(self, pure_state_density_matrix: np.ndarray) -> None:
        """Pure state |0><0| is a valid density matrix."""

        @validate_density_matrix
        def make_state() -> np.ndarray:
            return pure_state_density_matrix

        result = make_state()
        assert np.allclose(result, pure_state_density_matrix)

    def test_valid_mixed_state(self, mixed_state_density_matrix: np.ndarray) -> None:
        """Maximally mixed state is a valid density matrix."""

        @validate_density_matrix
        def make_state() -> np.ndarray:
            return mixed_state_density_matrix

        make_state()

    def test_invalid_not_hermitian(self, non_hermitian_matrix: np.ndarray) -> None:
        """Non-Hermitian matrix fails validation."""
        # Normalize trace
        matrix = non_hermitian_matrix / np.trace(non_hermitian_matrix)

        @validate_density_matrix
        def make_state() -> np.ndarray:
            return matrix

        with pytest.raises(ValidationError) as exc_info:
            make_state()

        assert "hermitian" in str(exc_info.value).lower()

    def test_invalid_wrong_trace(
        self, invalid_trace_density_matrix: np.ndarray
    ) -> None:
        """Matrix with trace != 1 fails validation."""

        @validate_density_matrix
        def make_state() -> np.ndarray:
            return invalid_trace_density_matrix

        with pytest.raises(ValidationError) as exc_info:
            make_state()

        assert "trace" in str(exc_info.value).lower()

    def test_invalid_negative_eigenvalue(
        self, negative_eigenvalue_matrix: np.ndarray
    ) -> None:
        """Matrix with negative eigenvalue fails validation."""

        @validate_density_matrix
        def make_state() -> np.ndarray:
            return negative_eigenvalue_matrix

        with pytest.raises(ValidationError) as exc_info:
            make_state()

        assert "positive semi-definite" in str(exc_info.value).lower()

    def test_valid_bell_state(self) -> None:
        """Bell state density matrix is valid."""
        # |Φ+> = (|00> + |11>)/√2
        bell_ket = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        bell_rho = np.outer(bell_ket, bell_ket.conj())

        @validate_density_matrix
        def make_bell() -> np.ndarray:
            return bell_rho

        make_bell()
