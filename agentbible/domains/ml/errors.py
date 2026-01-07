"""ML-specific validation errors with academic references.

These errors are raised when ML-specific validation checks fail, such as
data leakage detection, temporal autocorrelation issues, or coverage problems.
"""

from __future__ import annotations

from agentbible.errors import PhysicsConstraintError


class MLValidationError(PhysicsConstraintError):
    """Base class for ML validation errors.

    All ML-specific errors inherit from this class, making it easy
    to catch any ML-related validation failure.
    """

    pass


class DataLeakageError(MLValidationError):
    """Raised when forbidden features that leak target information are detected.

    Data leakage occurs when features contain information that would not
    be available at prediction time, or when features directly encode
    the target variable.

    Common causes:
        - Target-derived features (e.g., inversion_height = CBH - BLH)
        - Future data leaking into training features
        - Test set information leaking into training

    Example:
        >>> from agentbible.domains.ml import check_no_leakage
        >>> FORBIDDEN = {"inversion_height", "target_lagged"}
        >>> check_no_leakage(["blh", "t2m", "inversion_height"], forbidden=FORBIDDEN)
        DataLeakageError: Forbidden features found: {'inversion_height'}
    """

    REFERENCE = (
        "Kaufman et al., 'Leakage in Data Mining: Formulation, Detection, and "
        "Avoidance', ACM TKDD, 2011. Kapoor & Narayanan, 'Leakage and the "
        "Reproducibility Crisis in ML-based Science', Patterns, 2023."
    )
    GUIDANCE = (
        "Remove forbidden features from your dataset. These features encode "
        "or derive from the target variable, leading to artificially inflated "
        "performance metrics that won't generalize to new data.\n"
        "    - Check feature engineering for target-derived variables\n"
        "    - Ensure temporal ordering is respected (no future data)\n"
        "    - Verify train/test split before any preprocessing"
    )


class CoverageError(MLValidationError):
    """Raised when prediction intervals don't achieve target coverage.

    Coverage is the proportion of true values that fall within the
    predicted intervals. For valid uncertainty quantification, actual
    coverage should match the target (e.g., 90% intervals should
    contain ~90% of true values).

    Example:
        >>> from agentbible.domains.ml import check_coverage
        >>> check_coverage(y_true, y_lower, y_upper, target=0.90, tolerance=0.05)
        CoverageError: Coverage 0.72 is below target 0.90 (tolerance 0.05)
    """

    REFERENCE = (
        "Vovk et al., 'Algorithmic Learning in a Random World', Springer, 2005. "
        "Barber et al., 'Conformal Prediction Under Covariate Shift', NeurIPS, 2019."
    )
    GUIDANCE = (
        "Actual coverage doesn't match target. Common causes:\n"
        "    - Exchangeability assumption violated (temporal autocorrelation, "
        "domain shift)\n"
        "    - Model miscalibration\n"
        "    - Distribution shift between calibration and test sets\n"
        "Consider: per-group calibration, conditional coverage methods, "
        "or adaptive conformal prediction."
    )


class ExchangeabilityWarning(UserWarning):
    """Warning issued when data may violate the exchangeability assumption.

    Exchangeability is required for valid conformal prediction. It is
    violated by:
        - High temporal autocorrelation
        - Domain/distribution shift
        - Non-stationary processes

    This is a warning rather than an error because exchangeability
    violations don't always invalidate the analysis, but they should
    be documented and considered.
    """

    pass


class AutocorrelationWarning(UserWarning):
    """Warning issued when high temporal autocorrelation is detected.

    High autocorrelation (e.g., lag-1 > 0.5) means consecutive samples
    are highly correlated. This can:
        - Inflate random K-fold CV metrics
        - Violate independence assumptions
        - Lead to overly optimistic performance estimates

    Use time-series CV, per-group CV, or walk-forward validation instead.
    """

    pass


class CVStrategyWarning(UserWarning):
    """Warning issued when CV strategy may be inappropriate for the data.

    Issued when:
        - High autocorrelation detected with random CV
        - Small groups used with group CV
        - Other CV strategy mismatches
    """

    pass


__all__ = [
    "MLValidationError",
    "DataLeakageError",
    "CoverageError",
    "ExchangeabilityWarning",
    "AutocorrelationWarning",
    "CVStrategyWarning",
]
