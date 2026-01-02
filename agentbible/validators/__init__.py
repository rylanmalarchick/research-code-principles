"""Physics validation decorators for research code.

This module provides decorators that validate function outputs against
physical constraints like unitarity, hermiticity, and probability bounds.

Validation levels:
    - "debug" (default): Full validation with all physics checks
    - "lite": Only NaN/Inf sanity checks (fast)
    - "off": Skip validation entirely (DANGEROUS - for benchmarking only)

Control validation level via:
    - Decorator parameter: @validate_unitary(level="lite")
    - Environment variable: AGENTBIBLE_VALIDATION_LEVEL=off

Example:
    >>> from agentbible.validators import validate_unitary
    >>> import numpy as np
    >>>
    >>> @validate_unitary
    ... def create_gate():
    ...     return np.eye(2)
    >>>
    >>> gate = create_gate()  # Validates automatically

    >>> @validate_unitary(level="lite")  # Only NaN/Inf check (faster)
    ... def create_gate_fast():
    ...     return np.eye(2)
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
from agentbible.validators.base import (
    ENV_VALIDATION_LEVEL,
    ValidationLevel,
    get_validation_level,
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
    # Validation level control
    "ValidationLevel",
    "get_validation_level",
    "ENV_VALIDATION_LEVEL",
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
