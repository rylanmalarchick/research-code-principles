"""Physics validation decorators for research code.

This module provides decorators that validate function outputs against
physical constraints like unitarity, hermiticity, and probability bounds.

Example:
    >>> from agentbible.validators import validate_unitary
    >>> import numpy as np
    >>>
    >>> @validate_unitary
    ... def create_gate():
    ...     return np.eye(2)
    >>>
    >>> gate = create_gate()  # Validates automatically
"""

from agentbible.errors import (
    BoundsError,
    DensityMatrixError,
    HermiticityError,
    NonFiniteError,
    NormalizationError,
    PhysicsConstraintError,
    PositivityError,
    ProbabilityBoundsError,
    StateVectorNormError,
    TraceError,
    UnitarityError,
    ValidationError,
)
from agentbible.validators.bounds import (
    validate_finite,
    validate_non_negative,
    validate_positive,
    validate_range,
)
from agentbible.validators.probability import (
    validate_normalized,
    validate_probabilities,
    validate_probability,
)
from agentbible.validators.quantum import (
    validate_density_matrix,
    validate_hermitian,
    validate_unitary,
)

__all__ = [
    # Base errors
    "ValidationError",
    "PhysicsConstraintError",
    # Quantum errors
    "UnitarityError",
    "HermiticityError",
    "DensityMatrixError",
    "TraceError",
    "PositivityError",
    # Probability errors
    "ProbabilityBoundsError",
    "NormalizationError",
    "StateVectorNormError",
    # Numerical errors
    "NonFiniteError",
    "BoundsError",
    # Quantum validators
    "validate_unitary",
    "validate_hermitian",
    "validate_density_matrix",
    # Probability validators
    "validate_probability",
    "validate_probabilities",
    "validate_normalized",
    # Bounds validators
    "validate_positive",
    "validate_non_negative",
    "validate_range",
    "validate_finite",
]
