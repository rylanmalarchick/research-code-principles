"""Tests for quantum gate module.

Demonstrates testing principles for research code:
- Happy path tests for each gate type
- Edge cases (zero angle, 2pi rotation, boundary values)
- Invalid input handling (negative qubit, NaN, duplicates)
- Physical property verification (X^2 = I, unitarity, eigenvalues)

Run with: pytest test_gate.py -v
"""

import math

import numpy as np
import pytest

from gate import Gate, is_unitary, UNITARITY_TOLERANCE


class TestGateConstruction:
    """Tests for Gate dataclass creation and validation."""

    def test_x_gate_creation(self):
        """X gate should be created with correct attributes."""
        gate = Gate.x(qubit=0)
        assert gate.gate_type == "X"
        assert gate.qubits == (0,)
        assert gate.parameter is None

    def test_z_gate_creation(self):
        """Z gate should be created with correct attributes."""
        gate = Gate.z(qubit=1)
        assert gate.gate_type == "Z"
        assert gate.qubits == (1,)
        assert gate.parameter is None

    def test_h_gate_creation(self):
        """H gate should be created with correct attributes."""
        gate = Gate.h(qubit=2)
        assert gate.gate_type == "H"
        assert gate.qubits == (2,)
        assert gate.parameter is None

    def test_rz_gate_creation(self):
        """RZ gate should store rotation angle."""
        gate = Gate.rz(qubit=0, angle=math.pi / 4)
        assert gate.gate_type == "RZ"
        assert gate.qubits == (0,)
        assert gate.parameter == math.pi / 4

    def test_cnot_gate_creation(self):
        """CNOT gate should store control and target qubits."""
        gate = Gate.cnot(control=0, target=1)
        assert gate.gate_type == "CNOT"
        assert gate.qubits == (0, 1)
        assert gate.parameter is None

    def test_gate_is_immutable(self):
        """Gate should be frozen dataclass."""
        gate = Gate.x(qubit=0)
        with pytest.raises(AttributeError):
            gate.qubits = (1,)  # type: ignore


class TestInvalidInputs:
    """Fail-fast validation tests."""

    def test_negative_qubit_raises(self):
        """Negative qubit index should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Gate.x(qubit=-1)

    def test_nan_parameter_raises(self):
        """NaN parameter should raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            Gate.rz(qubit=0, angle=float("nan"))

    def test_inf_parameter_raises(self):
        """Infinite parameter should raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            Gate.rz(qubit=0, angle=float("inf"))

    def test_cnot_same_qubit_raises(self):
        """CNOT with same control and target should raise."""
        with pytest.raises(ValueError, match="different"):
            Gate.cnot(control=0, target=0)

    def test_duplicate_qubits_raises(self):
        """Duplicate qubit indices should raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate"):
            Gate(gate_type="TEST", qubits=(0, 0))

    def test_empty_qubits_raises(self):
        """Empty qubit tuple should raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            Gate(gate_type="TEST", qubits=())

    def test_string_parameter_raises(self):
        """Non-numeric parameter should raise TypeError."""
        with pytest.raises(TypeError, match="numeric"):
            Gate(gate_type="RZ", qubits=(0,), parameter="pi")  # type: ignore


class TestMatrixGeneration:
    """Tests for to_matrix() correctness."""

    def test_x_matrix(self):
        """X gate matrix should be [[0,1],[1,0]]."""
        matrix = Gate.x(qubit=0).to_matrix()
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_z_matrix(self):
        """Z gate matrix should be [[1,0],[0,-1]]."""
        matrix = Gate.z(qubit=0).to_matrix()
        expected = np.array([[1, 0], [0, -1]], dtype=complex)
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_h_matrix(self):
        """H gate matrix should be (1/sqrt2)*[[1,1],[1,-1]]."""
        matrix = Gate.h(qubit=0).to_matrix()
        inv_sqrt2 = 1 / math.sqrt(2)
        expected = np.array(
            [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=complex
        )
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_rz_matrix_pi_over_4(self):
        """RZ(pi/4) should give correct diagonal phases."""
        matrix = Gate.rz(qubit=0, angle=math.pi / 4).to_matrix()
        half_angle = math.pi / 8
        expected = np.array(
            [[np.exp(-1j * half_angle), 0], [0, np.exp(1j * half_angle)]],
            dtype=complex,
        )
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_cnot_matrix(self):
        """CNOT matrix should swap |10> and |11>."""
        matrix = Gate.cnot(control=0, target=1).to_matrix()
        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_unknown_gate_type_raises(self):
        """Unknown gate type should raise ValueError."""
        # Bypass factory to create invalid gate type
        gate = Gate(gate_type="UNKNOWN", qubits=(0,))
        with pytest.raises(ValueError, match="Unknown gate type"):
            gate.to_matrix()


class TestEdgeCases:
    """Edge case tests for boundary values."""

    def test_rz_zero_angle_is_identity(self):
        """RZ(0) should be identity matrix."""
        matrix = Gate.rz(qubit=0, angle=0.0).to_matrix()
        np.testing.assert_array_almost_equal(matrix, np.eye(2, dtype=complex))

    def test_rz_2pi_is_negative_identity(self):
        """RZ(2*pi) should be -I (global phase)."""
        matrix = Gate.rz(qubit=0, angle=2 * math.pi).to_matrix()
        # RZ(2pi) = [[exp(-i*pi), 0], [0, exp(i*pi)]] = [[-1, 0], [0, -1]]
        expected = -np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_rz_negative_angle(self):
        """RZ with negative angle should work correctly."""
        matrix = Gate.rz(qubit=0, angle=-math.pi / 2).to_matrix()
        assert is_unitary(matrix)

    def test_large_qubit_index(self):
        """Large qubit indices should be accepted."""
        gate = Gate.x(qubit=1000)
        assert gate.qubits == (1000,)
        # Matrix generation still works
        matrix = gate.to_matrix()
        assert matrix.shape == (2, 2)


class TestPhysicalProperties:
    """Tests verifying quantum mechanical correctness."""

    def test_x_squared_is_identity(self):
        """X^2 = I (Pauli gate property)."""
        x_matrix = Gate.x(qubit=0).to_matrix()
        x_squared = x_matrix @ x_matrix
        np.testing.assert_array_almost_equal(x_squared, np.eye(2, dtype=complex))

    def test_z_squared_is_identity(self):
        """Z^2 = I (Pauli gate property)."""
        z_matrix = Gate.z(qubit=0).to_matrix()
        z_squared = z_matrix @ z_matrix
        np.testing.assert_array_almost_equal(z_squared, np.eye(2, dtype=complex))

    def test_h_squared_is_identity(self):
        """H^2 = I (Hadamard is its own inverse)."""
        h_matrix = Gate.h(qubit=0).to_matrix()
        h_squared = h_matrix @ h_matrix
        np.testing.assert_array_almost_equal(h_squared, np.eye(2, dtype=complex))

    def test_all_gates_are_unitary(self):
        """All gate matrices must be unitary."""
        gates = [
            Gate.x(qubit=0),
            Gate.z(qubit=0),
            Gate.h(qubit=0),
            Gate.rz(qubit=0, angle=math.pi / 3),
            Gate.cnot(control=0, target=1),
        ]
        for gate in gates:
            matrix = gate.to_matrix()
            assert is_unitary(matrix), f"{gate} is not unitary"

    def test_xz_anticommute(self):
        """XZ = -ZX (Pauli anticommutation)."""
        x = Gate.x(qubit=0).to_matrix()
        z = Gate.z(qubit=0).to_matrix()
        xz = x @ z
        zx = z @ x
        np.testing.assert_array_almost_equal(xz, -1 * zx)

    def test_h_creates_superposition(self):
        """H|0> = (|0> + |1>)/sqrt(2)."""
        h = Gate.h(qubit=0).to_matrix()
        zero_state = np.array([1, 0], dtype=complex)
        result = h @ zero_state
        expected = np.array([1, 1], dtype=complex) / math.sqrt(2)
        np.testing.assert_array_almost_equal(result, expected)


class TestIsUnitaryHelper:
    """Tests for the is_unitary utility function."""

    def test_identity_is_unitary(self):
        """Identity matrix should be unitary."""
        assert is_unitary(np.eye(2, dtype=complex))

    def test_non_square_is_not_unitary(self):
        """Non-square matrix should not be unitary."""
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        assert not is_unitary(matrix)

    def test_scaled_unitary_is_not_unitary(self):
        """2*I is not unitary (breaks normalization)."""
        assert not is_unitary(2 * np.eye(2, dtype=complex))

    def test_custom_tolerance(self):
        """Custom tolerance should be respected."""
        # Create a slightly non-unitary matrix
        matrix = np.eye(2, dtype=complex)
        matrix[0, 0] = 1.0 + 1e-8
        assert is_unitary(matrix, tol=1e-6)
        assert not is_unitary(matrix, tol=1e-10)
