"""Core physics validation decorators and array checks for research code.

This module provides two types of validators:

1. **Decorators** (@validate_*): Validate function outputs automatically
2. **Array checks** (check_*): Validate arrays directly in pipelines

For domain-specific validators (e.g., quantum), import from the domain:
    from agentbible.domains.quantum import validate_unitary, validate_hermitian

Validation levels (for decorators):
    - "debug" (default): Full validation with all physics checks
    - "lite": Only NaN/Inf sanity checks (fast)
    - "off": Skip validation entirely (DANGEROUS - for benchmarking only)

Control validation level via:
    - Decorator parameter: @validate_finite(level="lite")
    - Environment variable: AGENTBIBLE_VALIDATION_LEVEL=off

Strict mode (for array checks):
    - strict=True (default): Raise exceptions on validation failure
    - strict=False: Issue warnings instead (useful for exploratory work)

Example (decorators):
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

Example (array checks):
    >>> from agentbible.validators import check_finite, check_positive
    >>> import numpy as np
    >>>
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> check_finite(arr, name="temperature")
    >>> check_positive(arr, name="temperature")
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
from agentbible.validators.arrays import (
    check_finite,
    check_non_negative,
    check_normalized,
    check_positive,
    check_probabilities,
    check_probability,
    check_range,
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
from agentbible.validators.pipeline import (
    ValidationPipeline,
    ValidationResult,
    create_distribution_pipeline,
    create_numeric_pipeline,
    create_positive_pipeline,
    create_probability_pipeline,
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
    # Probability validators (decorators)
    "validate_probability",
    "validate_probabilities",
    "validate_normalized",
    # Bounds validators (decorators)
    "validate_positive",
    "validate_non_negative",
    "validate_range",
    "validate_finite",
    # Array checks (direct functions)
    "check_finite",
    "check_positive",
    "check_non_negative",
    "check_range",
    "check_probability",
    "check_probabilities",
    "check_normalized",
    # Pipeline
    "ValidationPipeline",
    "ValidationResult",
    "create_numeric_pipeline",
    "create_positive_pipeline",
    "create_probability_pipeline",
    "create_distribution_pipeline",
]
