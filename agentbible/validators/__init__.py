"""Core validation decorators and array checks for scientific numerical code.

This module provides two types of validators:

1. **Decorators** (@validate_*): Validate function outputs automatically
2. **Array checks** (check_*): Validate arrays directly in pipelines

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
    DensityMatrixError,
    HermiticityError,
    NonFiniteError,
    NormalizationError,
    PhysicsConstraintError,
    PositiveDefiniteError,
    PositiveSemidefiniteError,
    ProbabilityBoundsError,
    StateVectorNormError,
    SymmetryError,
    TraceError,
    UnitarityError,
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
from agentbible.validators.matrix import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    check_density_matrix,
    check_hermitian,
    check_positive_definite,
    check_positive_semidefinite,
    check_symmetric,
    check_unitary,
    validate_density_matrix,
    validate_hermitian,
    validate_positive_definite,
    validate_positive_semidefinite,
    validate_symmetric,
    validate_unitary,
)
from agentbible.validators.pipeline import (
    ValidationPipeline,
    ValidationResult,
    create_distribution_pipeline,
    create_numeric_pipeline,
    create_positive_pipeline,
    create_probability_pipeline,
)
from agentbible.validators.probability import (
    validate_normalized,
    validate_probabilities,
    validate_probability,
)

check_finite_array = check_finite
check_positive_array = check_positive
check_non_negative_array = check_non_negative
check_probability_array = check_probabilities
check_normalized_l1 = check_normalized
validate_normalized_l1 = validate_normalized

__all__ = [
    "DEFAULT_RTOL",
    "DEFAULT_ATOL",
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
    # Matrix errors
    "SymmetryError",
    "HermiticityError",
    "UnitarityError",
    "PositiveDefiniteError",
    "PositiveSemidefiniteError",
    "TraceError",
    "DensityMatrixError",
    # Probability validators (decorators)
    "validate_probability",
    "validate_probabilities",
    "validate_normalized",
    "validate_normalized_l1",
    # Bounds validators (decorators)
    "validate_positive",
    "validate_non_negative",
    "validate_range",
    "validate_finite",
    # Matrix validators (decorators)
    "validate_symmetric",
    "validate_hermitian",
    "validate_unitary",
    "validate_positive_definite",
    "validate_positive_semidefinite",
    "validate_density_matrix",
    # Array checks (direct functions)
    "check_finite",
    "check_finite_array",
    "check_positive",
    "check_positive_array",
    "check_non_negative",
    "check_non_negative_array",
    "check_range",
    "check_probability",
    "check_probabilities",
    "check_normalized",
    "check_probability_array",
    "check_normalized_l1",
    # Matrix checks (direct functions)
    "check_symmetric",
    "check_hermitian",
    "check_unitary",
    "check_positive_definite",
    "check_positive_semidefinite",
    "check_density_matrix",
    # Pipeline
    "ValidationPipeline",
    "ValidationResult",
    "create_numeric_pipeline",
    "create_positive_pipeline",
    "create_probability_pipeline",
    "create_distribution_pipeline",
]
