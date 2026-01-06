"""Quantum computing validation domain.

Provides validators for quantum mechanical properties:
- Unitarity: U†U = I (quantum gates)
- Hermiticity: H = H† (observables/Hamiltonians)
- Density matrices: Hermitian, trace 1, positive semi-definite

Example:
    >>> from agentbible.domains.quantum import validate_unitary, validate_hermitian
    >>> import numpy as np
    >>>
    >>> @validate_unitary
    ... def create_hadamard():
    ...     return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    >>>
    >>> @validate_hermitian
    ... def create_pauli_z():
    ...     return np.array([[1, 0], [0, -1]])

All validators support validation levels:
    - "debug" (default): Full validation with all physics checks
    - "lite": Only NaN/Inf sanity checks (fast)
    - "off": Skip validation entirely (DANGEROUS - for benchmarking only)

Control via decorator parameter or AGENTBIBLE_VALIDATION_LEVEL environment variable.
"""

from __future__ import annotations

from agentbible.domains.quantum.errors import (
    DensityMatrixError,
    HermiticityError,
    PositivityError,
    TraceError,
    UnitarityError,
)
from agentbible.domains.quantum.validators import (
    validate_density_matrix,
    validate_hermitian,
    validate_unitary,
)

__all__ = [
    # Validators
    "validate_unitary",
    "validate_hermitian",
    "validate_density_matrix",
    # Errors
    "UnitarityError",
    "HermiticityError",
    "DensityMatrixError",
    "TraceError",
    "PositivityError",
]
