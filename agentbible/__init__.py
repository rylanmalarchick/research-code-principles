"""AgentBible: Production-grade infrastructure for AI-assisted research software.

This package provides:
- Validators: Physics validation decorators (bounds, probability, normalization)
- Array Checks: Direct validation functions for data pipelines
- Provenance: HDF5-based reproducibility tracking
- Testing: Physics-aware pytest fixtures and decorators
- CLI: Project scaffolding and context management (bible init, bible context)

For domain-specific validators (e.g., quantum), import from the domain:
    from agentbible.domains.quantum import validate_unitary, validate_hermitian

For ML-specific validators (e.g., data leakage), import from the ML domain:
    from agentbible.domains.ml import check_no_leakage, check_temporal_autocorrelation

Example (decorators):
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

Example (array checks):
    >>> from agentbible import check_finite, check_positive, check_range
    >>> import numpy as np
    >>>
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> check_finite(arr, name="temperature")
    >>> check_positive(arr, name="temperature")
    >>> check_range(arr, 0, 100, name="temperature")
    >>>
    >>> # Warn instead of raise (strict=False)
    >>> check_positive(arr, name="x", strict=False)  # Warns if invalid
"""

from __future__ import annotations

__version__ = "0.4.0"
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

# Public API - core validators (decorators)
from agentbible.validators import (
    validate_finite,
    validate_non_negative,
    validate_normalized,
    validate_positive,
    validate_probabilities,
    validate_probability,
    validate_range,
)

# Public API - array checks (direct validation functions)
from agentbible.validators import (
    check_finite,
    check_non_negative,
    check_normalized,
    check_positive,
    check_probabilities,
    check_probability,
    check_range,
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
    # Probability validators (decorators)
    "validate_probability",
    "validate_probabilities",
    "validate_normalized",
    # Bounds validators (decorators)
    "validate_positive",
    "validate_non_negative",
    "validate_range",
    "validate_finite",
    # Array checks (direct validation functions)
    "check_finite",
    "check_positive",
    "check_non_negative",
    "check_range",
    "check_probability",
    "check_probabilities",
    "check_normalized",
]
