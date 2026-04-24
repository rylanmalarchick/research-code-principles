"""AgentBible public package surface."""

from __future__ import annotations

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
from agentbible.validators import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    check_density_matrix,
    check_finite,
    check_finite_array,
    check_hermitian,
    check_non_negative,
    check_non_negative_array,
    check_normalized,
    check_normalized_l1,
    check_positive,
    check_positive_array,
    check_positive_definite,
    check_positive_semidefinite,
    check_probabilities,
    check_probability,
    check_probability_array,
    check_range,
    check_symmetric,
    check_unitary,
    validate_density_matrix,
    validate_finite,
    validate_hermitian,
    validate_non_negative,
    validate_normalized,
    validate_normalized_l1,
    validate_positive,
    validate_positive_definite,
    validate_positive_semidefinite,
    validate_probabilities,
    validate_probability,
    validate_range,
    validate_symmetric,
    validate_unitary,
)

__version__ = "1.0.0"
__author__ = "Rylan Malarchick"
__email__ = "rylan1012@gmail.com"
SPEC_VERSION = "1.0"

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "SPEC_VERSION",
    "DEFAULT_RTOL",
    "DEFAULT_ATOL",
    # Base exceptions
    "ValidationError",
    "PhysicsConstraintError",
    # Core exceptions
    "ProbabilityBoundsError",
    "NormalizationError",
    "StateVectorNormError",
    "NonFiniteError",
    "BoundsError",
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
    # Array checks (direct validation functions)
    "check_finite",
    "check_finite_array",
    "check_positive",
    "check_positive_array",
    "check_non_negative",
    "check_non_negative_array",
    "check_range",
    "check_probability",
    "check_probabilities",
    "check_probability_array",
    "check_normalized",
    "check_normalized_l1",
    # Matrix checks (direct validation functions)
    "check_symmetric",
    "check_hermitian",
    "check_unitary",
    "check_positive_definite",
    "check_positive_semidefinite",
    "check_density_matrix",
]
