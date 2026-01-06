"""Core physics validation decorators for research code.

This module provides decorators that validate function outputs against
physical constraints like probability bounds, normalization, and finiteness.

For domain-specific validators (e.g., quantum), import from the domain:
    from agentbible.domains.quantum import validate_unitary, validate_hermitian

Validation levels:
    - "debug" (default): Full validation with all physics checks
    - "lite": Only NaN/Inf sanity checks (fast)
    - "off": Skip validation entirely (DANGEROUS - for benchmarking only)

Control validation level via:
    - Decorator parameter: @validate_finite(level="lite")
    - Environment variable: AGENTBIBLE_VALIDATION_LEVEL=off

Example:
    >>> from agentbible.validators import validate_finite, validate_normalized
    >>> import numpy as np
    >>>
    >>> @validate_finite
    ... def compute_result():
    ...     return np.array([1.0, 2.0, 3.0])
    >>>
    >>> @validate_normalized
    ... def get_distribution():
    ...     return np.array([0.25, 0.25, 0.25, 0.25])
"""

from agentbible.errors import (
    BoundsError,
    NonFiniteError,
    NormalizationError,
    PhysicsConstraintError,
    ProbabilityBoundsError,
    StateVectorNormError,
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

__all__ = [
    # Validation level control
    "ValidationLevel",
    "get_validation_level",
    "ENV_VALIDATION_LEVEL",
    # Base errors
    "ValidationError",
    "PhysicsConstraintError",
    # Probability errors
    "ProbabilityBoundsError",
    "NormalizationError",
    "StateVectorNormError",
    # Numerical errors
    "NonFiniteError",
    "BoundsError",
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
