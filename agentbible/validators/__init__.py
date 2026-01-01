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

from agentbible.validators.base import ValidationError
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
    # Base
    "ValidationError",
    # Quantum
    "validate_unitary",
    "validate_hermitian",
    "validate_density_matrix",
    # Probability
    "validate_probability",
    "validate_probabilities",
    "validate_normalized",
    # Bounds
    "validate_positive",
    "validate_non_negative",
    "validate_range",
    "validate_finite",
]
