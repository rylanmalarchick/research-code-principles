"""Atmospheric science validation errors with academic references.

These errors are raised when atmospheric physics validation checks fail,
such as cloud base height constraints, boundary layer height limits,
or cloud layer consistency violations.
"""

from __future__ import annotations

from agentbible.errors import PhysicsConstraintError


class AtmosphericValidationError(PhysicsConstraintError):
    """Base class for atmospheric science validation errors.

    All atmospheric-specific errors inherit from this class, making it
    easy to catch any atmospheric-related validation failure.
    """

    pass


class CloudBaseHeightError(AtmosphericValidationError):
    """Raised when cloud base height (CBH) is outside valid range.

    Cloud base height should typically be:
    - Above the surface (> 0 m AGL)
    - Below the tropopause (typically < 12000 m)
    - For low clouds: 0-2000 m
    - For mid clouds: 2000-6000 m
    - For high clouds: 6000-12000 m

    Example:
        >>> from agentbible.domains.atmospheric import check_cloud_base_height
        >>> check_cloud_base_height(np.array([-100, 500, 1000]))
        CloudBaseHeightError: Cloud base height contains invalid values
    """

    REFERENCE = (
        "World Meteorological Organization, 'International Cloud Atlas', 2017. "
        "Stull, 'An Introduction to Boundary Layer Meteorology', Springer, 1988."
    )
    GUIDANCE = (
        "Cloud base height should be positive and below the tropopause. "
        "Common issues:\n"
        "    - Negative values indicate measurement errors or below-surface values\n"
        "    - Values > 12 km are unlikely for most cloud types\n"
        "    - Check sensor calibration and data quality flags"
    )


class BoundaryLayerHeightError(AtmosphericValidationError):
    """Raised when boundary layer height (BLH) is outside valid range.

    The atmospheric boundary layer height (BLH) typically ranges from:
    - Nighttime stable BL: 50-500 m
    - Daytime convective BL: 500-3000 m
    - Extreme convective conditions: up to 5000 m

    Example:
        >>> from agentbible.domains.atmospheric import check_boundary_layer_height
        >>> check_boundary_layer_height(np.array([100, 2000, 8000]))
        BoundaryLayerHeightError: Boundary layer height exceeds typical limits
    """

    REFERENCE = (
        "Stull, 'An Introduction to Boundary Layer Meteorology', Springer, 1988. "
        "Seibert et al., 'Review and intercomparison of operational methods for "
        "the determination of the mixing height', Atmos. Environ., 2000."
    )
    GUIDANCE = (
        "Boundary layer height should typically be 50-5000 m. "
        "Extreme values may indicate:\n"
        "    - Algorithm artifacts in retrieval\n"
        "    - Free tropospheric air mass misidentified as BL\n"
        "    - Very deep convective conditions (rare)"
    )


class LiftingCondensationLevelError(AtmosphericValidationError):
    """Raised when lifting condensation level (LCL) is outside valid range.

    The LCL is the height at which a parcel becomes saturated when lifted
    adiabatically. It should be:
    - Above the surface (> 0 m AGL)
    - Typically below 4000 m in most conditions
    - Equal to or above the surface dewpoint depression

    Example:
        >>> from agentbible.domains.atmospheric import check_lifting_condensation_level
        >>> check_lifting_condensation_level(np.array([500, 2000, -100]))
        LiftingCondensationLevelError: LCL contains invalid values
    """

    REFERENCE = (
        "Bolton, 'The computation of equivalent potential temperature', "
        "Mon. Wea. Rev., 1980. "
        "Romps, 'Exact expression for the lifting condensation level', "
        "J. Atmos. Sci., 2017."
    )
    GUIDANCE = (
        "LCL should be positive and typically below 4000 m. Check:\n"
        "    - Temperature and dewpoint input data quality\n"
        "    - Surface pressure values\n"
        "    - Calculation method (Romps 2017 is recommended)"
    )


class CloudLayerConsistencyError(AtmosphericValidationError):
    """Raised when cloud layer relationships are physically inconsistent.

    Cloud layers should satisfy physical constraints:
    - Cloud base < cloud top
    - No overlapping layers (unless multi-layer clouds intended)
    - Higher layers have higher bases than lower layers

    Example:
        >>> from agentbible.domains.atmospheric import check_cloud_layer_consistency
        >>> check_cloud_layer_consistency(base=1000, top=800)  # Inverted
        CloudLayerConsistencyError: Cloud base must be below cloud top
    """

    REFERENCE = (
        "WMO, 'International Cloud Atlas', 2017. "
        "Intrieri et al., 'An annual cycle of Arctic cloud characteristics', "
        "JGR Atmospheres, 2002."
    )
    GUIDANCE = (
        "Cloud layers must satisfy physical constraints:\n"
        "    - Base must be below top\n"
        "    - Layer thickness should be positive\n"
        "    - Check for data ordering issues or unit mismatches"
    )


class TemperatureInversionError(AtmosphericValidationError):
    """Raised when temperature inversion properties are invalid.

    Temperature inversions should satisfy:
    - Inversion base < inversion top
    - Temperature at top > temperature at base (by definition)
    - Inversion strength typically < 20 K

    Example:
        >>> from agentbible.domains.atmospheric import check_temperature_inversion
        >>> check_temperature_inversion(base_temp=300, top_temp=290)
        TemperatureInversionError: Not a valid inversion (temp decreasing with height)
    """

    REFERENCE = (
        "Stull, 'An Introduction to Boundary Layer Meteorology', Springer, 1988. "
        "Seidel et al., 'Climatological characteristics of the Arctic mean boundary layer', "
        "JGR Atmospheres, 2012."
    )
    GUIDANCE = (
        "Temperature inversions must have temperature increasing with height:\n"
        "    - Ensure top temperature > base temperature\n"
        "    - Typical inversion strengths are 1-10 K\n"
        "    - Strengths > 20 K are rare and should be verified"
    )


class RelativeHumidityWarning(UserWarning):
    """Warning for relative humidity values near physical limits.

    Relative humidity should be in [0, 100]%, but due to measurement
    uncertainty and supersaturation, values slightly outside this range
    may occur.
    """

    pass


class AtmosphericStabilityWarning(UserWarning):
    """Warning for unusual atmospheric stability conditions.

    Issued when atmospheric stability parameters suggest unusual but
    not necessarily invalid conditions.
    """

    pass


__all__ = [
    "AtmosphericValidationError",
    "CloudBaseHeightError",
    "BoundaryLayerHeightError",
    "LiftingCondensationLevelError",
    "CloudLayerConsistencyError",
    "TemperatureInversionError",
    "RelativeHumidityWarning",
    "AtmosphericStabilityWarning",
]
