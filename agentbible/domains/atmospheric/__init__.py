"""Atmospheric science validation domain.

Provides validators for atmospheric physics properties:
- Cloud base height (CBH): Valid altitude ranges
- Boundary layer height (BLH): Physically plausible limits
- Lifting condensation level (LCL): Thermodynamic constraints
- Cloud layer consistency: Base < top relationships
- Relative humidity: 0-100% bounds (with supersaturation option)
- Temperature inversions: Physical consistency checks

Example:
    >>> from agentbible.domains.atmospheric import (
    ...     check_cloud_base_height,
    ...     check_boundary_layer_height,
    ...     check_cloud_layer_consistency,
    ... )
    >>> import numpy as np
    >>>
    >>> # Validate cloud base heights
    >>> cbh = np.array([500, 1000, 1500])  # meters AGL
    >>> check_cloud_base_height(cbh, name="cbh")
    >>>
    >>> # Validate cloud layer consistency
    >>> base = np.array([500, 2000])
    >>> top = np.array([1500, 3000])
    >>> check_cloud_layer_consistency(base, top, name="stratus")

All validators support strict mode:
    - strict=True (default): Raise exceptions on validation failure
    - strict=False: Issue warnings instead (useful for exploratory work)
"""

from __future__ import annotations

from agentbible.domains.atmospheric.checks import (
    check_boundary_layer_height,
    check_cloud_base_height,
    check_cloud_layer_consistency,
    check_lifting_condensation_level,
    check_relative_humidity,
    check_temperature_inversion,
)
from agentbible.domains.atmospheric.errors import (
    AtmosphericStabilityWarning,
    AtmosphericValidationError,
    BoundaryLayerHeightError,
    CloudBaseHeightError,
    CloudLayerConsistencyError,
    LiftingCondensationLevelError,
    RelativeHumidityWarning,
    TemperatureInversionError,
)

__all__ = [
    # Check functions
    "check_cloud_base_height",
    "check_boundary_layer_height",
    "check_lifting_condensation_level",
    "check_cloud_layer_consistency",
    "check_relative_humidity",
    "check_temperature_inversion",
    # Errors
    "AtmosphericValidationError",
    "CloudBaseHeightError",
    "BoundaryLayerHeightError",
    "LiftingCondensationLevelError",
    "CloudLayerConsistencyError",
    "TemperatureInversionError",
    # Warnings
    "RelativeHumidityWarning",
    "AtmosphericStabilityWarning",
]
