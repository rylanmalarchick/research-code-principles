"""Quantum gate representation with validation and matrix generation.

This module demonstrates the five research code principles:
1. Correctness First - rigorous input validation and physical constraints
2. Simple Beats Clever - straightforward dataclass design
3. Make It Inspectable - clear repr, logging, intermediate values
4. Fail Fast and Loud - explicit exceptions with physics context
5. Reproducibility Is Sacred - deterministic matrix generation

References:
    Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum
    Information (10th Anniversary ed.). Cambridge University Press.
    Chapter 4: Quantum Circuits.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
UNITARITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Gate:
    """Immutable representation of a quantum gate.

    Attributes:
        gate_type: Name of the gate (e.g., "X", "Z", "H", "RZ", "CNOT").
        qubits: Tuple of qubit indices this gate acts on.
        parameter: Optional rotation angle in radians (for parametric gates).

    Example:
        >>> gate = Gate.rz(qubit=0, angle=math.pi / 4)
        >>> matrix = gate.to_matrix()
        >>> assert is_unitary(matrix)
    """

    gate_type: str
    qubits: tuple[int, ...]
    parameter: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate gate construction. Fail fast on invalid inputs."""
        self._validate_qubits()
        self._validate_parameter()

    def _validate_qubits(self) -> None:
        """Check qubit indices are valid non-negative integers."""
        if not self.qubits:
            raise ValueError("Gate must act on at least one qubit.")
        for q in self.qubits:
            if not isinstance(q, int) or q < 0:
                raise ValueError(
                    f"Qubit index must be non-negative integer, got {q!r}."
                )
        if len(self.qubits) != len(set(self.qubits)):
            raise ValueError(f"Duplicate qubit indices: {self.qubits}.")

    def _validate_parameter(self) -> None:
        """Check rotation parameter is finite when present."""
        if self.parameter is not None:
            if not isinstance(self.parameter, (int, float)):
                raise TypeError(
                    f"Parameter must be numeric, got {type(self.parameter).__name__}."
                )
            if math.isnan(self.parameter) or math.isinf(self.parameter):
                raise ValueError(
                    f"Parameter must be finite, got {self.parameter}."
                )

    # === Factory Methods ===
    # These provide the standard gates with proper documentation.

    @classmethod
    def x(cls, qubit: int) -> Gate:
        """Pauli-X gate (bit flip).

        Matrix: [[0, 1], [1, 0]]

        Reference: Nielsen & Chuang, Eq. 4.2
        """
        return cls(gate_type="X", qubits=(qubit,))

    @classmethod
    def z(cls, qubit: int) -> Gate:
        """Pauli-Z gate (phase flip).

        Matrix: [[1, 0], [0, -1]]

        Reference: Nielsen & Chuang, Eq. 4.3
        """
        return cls(gate_type="Z", qubits=(qubit,))

    @classmethod
    def h(cls, qubit: int) -> Gate:
        """Hadamard gate (creates superposition).

        Matrix: (1/sqrt(2)) * [[1, 1], [1, -1]]

        Reference: Nielsen & Chuang, Eq. 4.4
        """
        return cls(gate_type="H", qubits=(qubit,))

    @classmethod
    def rz(cls, qubit: int, angle: float) -> Gate:
        """Rotation around Z-axis by given angle.

        Matrix: [[exp(-i*angle/2), 0], [0, exp(i*angle/2)]]

        Args:
            qubit: Index of qubit to rotate.
            angle: Rotation angle in radians.

        Reference: Nielsen & Chuang, Eq. 4.6
        """
        return cls(gate_type="RZ", qubits=(qubit,), parameter=angle)

    @classmethod
    def cnot(cls, control: int, target: int) -> Gate:
        """Controlled-NOT gate (entangling gate).

        Flips target qubit if control qubit is |1>.

        Args:
            control: Index of control qubit.
            target: Index of target qubit.

        Reference: Nielsen & Chuang, Section 4.6
        """
        if control == target:
            raise ValueError(
                f"Control and target must be different, both are {control}."
            )
        return cls(gate_type="CNOT", qubits=(control, target))

    # === Matrix Generation ===

    def to_matrix(self) -> np.ndarray:
        """Generate the unitary matrix for this gate.

        Returns:
            Complex numpy array representing the gate's unitary matrix.

        Raises:
            ValueError: If the gate type is unknown.
            RuntimeError: If the generated matrix fails unitarity check.
        """
        logger.debug("Generating matrix for %s", self)

        matrix = self._compute_matrix()
        self._verify_unitarity(matrix)

        return matrix

    def _compute_matrix(self) -> np.ndarray:
        """Compute the raw matrix for this gate type."""
        if self.gate_type == "X":
            return np.array([[0, 1], [1, 0]], dtype=complex)

        if self.gate_type == "Z":
            return np.array([[1, 0], [0, -1]], dtype=complex)

        if self.gate_type == "H":
            inv_sqrt2 = 1 / math.sqrt(2)
            return np.array(
                [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=complex
            )

        if self.gate_type == "RZ":
            assert self.parameter is not None  # Guaranteed by factory
            half_angle = self.parameter / 2
            return np.array(
                [
                    [np.exp(-1j * half_angle), 0],
                    [0, np.exp(1j * half_angle)],
                ],
                dtype=complex,
            )

        if self.gate_type == "CNOT":
            # |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=complex,
            )

        raise ValueError(f"Unknown gate type: {self.gate_type!r}")

    def _verify_unitarity(self, matrix: np.ndarray) -> None:
        """Verify that matrix is unitary: U @ U^dagger = I.

        This is a critical physical constraint. Quantum gates must preserve
        probability normalization.

        Reference: Nielsen & Chuang, Section 2.2.1
        """
        product = matrix @ matrix.conj().T
        identity = np.eye(matrix.shape[0], dtype=complex)
        deviation = np.max(np.abs(product - identity))

        if deviation > UNITARITY_TOLERANCE:
            raise RuntimeError(
                f"Unitarity violation: max deviation {deviation:.2e} "
                f"exceeds tolerance {UNITARITY_TOLERANCE:.2e}. "
                f"This indicates a bug in matrix generation for {self}."
            )

        logger.debug("Unitarity verified, max deviation: %.2e", deviation)


def is_unitary(matrix: np.ndarray, tol: float = UNITARITY_TOLERANCE) -> bool:
    """Check if a matrix is unitary within tolerance.

    Useful for testing and validation.

    Args:
        matrix: Square numpy array to check.
        tol: Maximum allowed deviation from identity.

    Returns:
        True if matrix satisfies U @ U^dagger = I within tolerance.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    product = matrix @ matrix.conj().T
    identity = np.eye(matrix.shape[0], dtype=complex)
    return np.max(np.abs(product - identity)) <= tol
