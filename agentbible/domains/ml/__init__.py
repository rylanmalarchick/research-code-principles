"""Machine learning validation domain.

Provides validators for ML-specific properties:
- Data leakage detection: Prevent forbidden features that leak target info
- Temporal autocorrelation: Detect high correlation that invalidates random CV
- Coverage validation: Verify prediction intervals achieve target coverage
- Exchangeability checks: Validate conformal prediction assumptions

Example:
    >>> from agentbible.domains.ml import check_no_leakage, check_coverage
    >>> import numpy as np
    >>>
    >>> # Direct array checks
    >>> FORBIDDEN = {"target_lagged", "inversion_height"}
    >>> check_no_leakage(["blh", "t2m", "rh"], forbidden=FORBIDDEN)
    >>>
    >>> # Coverage validation
    >>> check_coverage(y_true, y_lower, y_upper, target=0.90)

All validators support validation levels:
    - "debug" (default): Full validation with all checks
    - "lite": Only basic sanity checks (fast)
    - "off": Skip validation entirely (DANGEROUS - for benchmarking only)

Control via decorator parameter or AGENTBIBLE_VALIDATION_LEVEL environment variable.
"""

from __future__ import annotations

from agentbible.domains.ml.checks import (
    check_coverage,
    check_exchangeability,
    check_no_leakage,
    check_temporal_autocorrelation,
)
from agentbible.domains.ml.errors import (
    AutocorrelationWarning,
    CoverageError,
    CVStrategyWarning,
    DataLeakageError,
    ExchangeabilityWarning,
    MLValidationError,
)
from agentbible.domains.ml.validators import (
    validate_cv_strategy,
    validate_no_leakage,
)

__all__ = [
    # Check functions (direct array validation)
    "check_no_leakage",
    "check_temporal_autocorrelation",
    "check_coverage",
    "check_exchangeability",
    # Decorators
    "validate_no_leakage",
    "validate_cv_strategy",
    # Errors
    "MLValidationError",
    "DataLeakageError",
    "CoverageError",
    # Warnings
    "ExchangeabilityWarning",
    "AutocorrelationWarning",
    "CVStrategyWarning",
]
