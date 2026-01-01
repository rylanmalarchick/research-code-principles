"""Physical validation utilities for research code.

This module provides helper functions for validating physical constraints
commonly encountered in quantum computing and numerical research.

Author: Rylan Malarchick
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def check_unitarity(
    matrix: NDArray[np.complexfloating],
    tolerance: float = 1e-10,
    name: Optional[str] = None,
) -> None:
    """Validate that a matrix is unitary (U†U = I).

    Args:
        matrix: Square complex matrix to validate.
        tolerance: Numerical tolerance for comparison.
        name: Optional name for error messages.

    Raises:
        ValueError: If matrix is not square or not unitary.

    Example:
        >>> U = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X
        >>> check_unitarity(U)  # No error
        >>> check_unitarity(np.array([[1, 1], [0, 1]]))  # Raises ValueError
    """
    label = f" '{name}'" if name else ""

    if matrix.ndim != 2:
        raise ValueError(f"Matrix{label} must be 2D, got shape {matrix.shape}")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix{label} must be square, got shape {matrix.shape}")

    identity = np.eye(matrix.shape[0], dtype=complex)
    product = matrix.conj().T @ matrix

    if not np.allclose(product, identity, atol=tolerance):
        max_error = np.max(np.abs(product - identity))
        raise ValueError(
            f"Matrix{label} is not unitary: max|U†U - I| = {max_error:.2e} "
            f"(tolerance: {tolerance:.2e})"
        )


def check_hermitian(
    matrix: NDArray[np.complexfloating],
    tolerance: float = 1e-10,
    name: Optional[str] = None,
) -> None:
    """Validate that a matrix is Hermitian (H = H†).

    Args:
        matrix: Square complex matrix to validate.
        tolerance: Numerical tolerance for comparison.
        name: Optional name for error messages.

    Raises:
        ValueError: If matrix is not square or not Hermitian.
    """
    label = f" '{name}'" if name else ""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix{label} must be square, got shape {matrix.shape}")

    if not np.allclose(matrix, matrix.conj().T, atol=tolerance):
        max_error = np.max(np.abs(matrix - matrix.conj().T))
        raise ValueError(
            f"Matrix{label} is not Hermitian: max|H - H†| = {max_error:.2e}"
        )


def check_normalized(
    state: NDArray[np.complexfloating],
    tolerance: float = 1e-10,
    name: Optional[str] = None,
) -> None:
    """Validate that a quantum state is normalized (⟨ψ|ψ⟩ = 1).

    Args:
        state: Complex state vector.
        tolerance: Numerical tolerance for comparison.
        name: Optional name for error messages.

    Raises:
        ValueError: If state is not normalized.
    """
    label = f" '{name}'" if name else ""
    norm = np.linalg.norm(state)

    if not np.isclose(norm, 1.0, atol=tolerance):
        raise ValueError(
            f"State{label} is not normalized: |⟨ψ|ψ⟩| = {norm:.6f} (expected 1.0)"
        )


def check_density_matrix(
    rho: NDArray[np.complexfloating],
    tolerance: float = 1e-10,
    name: Optional[str] = None,
) -> None:
    """Validate that a matrix is a valid density matrix.

    Checks:
    1. Hermitian: ρ = ρ†
    2. Trace one: Tr(ρ) = 1
    3. Positive semidefinite: all eigenvalues ≥ 0

    Args:
        rho: Square complex matrix to validate.
        tolerance: Numerical tolerance for comparison.
        name: Optional name for error messages.

    Raises:
        ValueError: If any density matrix property is violated.
    """
    label = f" '{name}'" if name else ""

    # Check Hermitian
    check_hermitian(rho, tolerance, name)

    # Check trace = 1
    trace = np.trace(rho)
    if not np.isclose(trace, 1.0, atol=tolerance):
        raise ValueError(f"Density matrix{label} trace = {trace:.6f} (expected 1.0)")

    # Check positive semidefinite
    eigenvalues = np.linalg.eigvalsh(rho)
    min_eigenvalue = np.min(eigenvalues)
    if min_eigenvalue < -tolerance:
        raise ValueError(
            f"Density matrix{label} has negative eigenvalue: {min_eigenvalue:.2e}"
        )


def check_probabilities(
    probs: NDArray[np.floating],
    tolerance: float = 1e-10,
    name: Optional[str] = None,
) -> None:
    """Validate that an array represents valid probabilities.

    Checks:
    1. All values non-negative
    2. Sum equals 1

    Args:
        probs: Array of probability values.
        tolerance: Numerical tolerance for comparison.
        name: Optional name for error messages.

    Raises:
        ValueError: If probabilities are invalid.
    """
    label = f" '{name}'" if name else ""

    if np.any(probs < -tolerance):
        min_val = np.min(probs)
        raise ValueError(f"Probabilities{label} contain negative value: {min_val:.2e}")

    total = np.sum(probs)
    if not np.isclose(total, 1.0, atol=tolerance):
        raise ValueError(f"Probabilities{label} sum to {total:.6f} (expected 1.0)")
