"""AgentBible: Production-grade infrastructure for AI-assisted research software.

This package provides:
- Validators: Physics validation decorators (bounds, probability, normalization)
- Provenance: HDF5-based reproducibility tracking
- Testing: Physics-aware pytest fixtures and decorators
- CLI: Project scaffolding and context management (bible init, bible context)

For domain-specific validators (e.g., quantum), import from the domain:
    from agentbible.domains.quantum import validate_unitary, validate_hermitian

Example:
    >>> from agentbible import validate_finite, validate_normalized
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

from __future__ import annotations

__version__ = "0.3.0"
__author__ = "Rylan Malarchick"
__email__ = "rylan1012@gmail.com"

# Public API - core errors
from agentbible.errors import (
    BoundsError,
    NonFiniteError,
    NormalizationError,
    PhysicsConstraintError,
    ProbabilityBoundsError,
    StateVectorNormError,
    ValidationError,
)

# Public API - core validators
from agentbible.validators import (
    validate_finite,
    validate_non_negative,
    validate_normalized,
    validate_positive,
    validate_probabilities,
    validate_probability,
    validate_range,
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    # Base exceptions
    "ValidationError",
    "PhysicsConstraintError",
    # Core exceptions
    "ProbabilityBoundsError",
    "NormalizationError",
    "StateVectorNormError",
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
