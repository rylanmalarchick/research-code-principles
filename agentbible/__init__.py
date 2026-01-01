"""AgentBible: Production-grade infrastructure for AI-assisted research software.

This package provides:
- Validators: Physics validation decorators (@validate_unitary, @validate_hermitian, etc.)
- CLI: Project scaffolding and context management (bible init, bible context)
- Testing: Physics-aware pytest fixtures and decorators

Example:
    >>> from agentbible.validators import validate_unitary
    >>> import numpy as np
    >>>
    >>> @validate_unitary
    ... def create_hadamard():
    ...     return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    >>>
    >>> H = create_hadamard()  # Validates automatically
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Rylan Malarchick"
__email__ = "rylan1012@gmail.com"

# Public API - validators
from agentbible.validators import (
    ValidationError,
    validate_density_matrix,
    validate_finite,
    validate_hermitian,
    validate_non_negative,
    validate_normalized,
    validate_positive,
    validate_probabilities,
    validate_probability,
    validate_range,
    validate_unitary,
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    # Exceptions
    "ValidationError",
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
