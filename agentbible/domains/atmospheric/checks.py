"""Direct atmospheric validation functions for data pipelines.

These validation functions check atmospheric science constraints
such as cloud base height ranges, boundary layer height limits,
and cloud layer consistency.

All functions:
- Take data as the first argument
- Take a `name` parameter for clear error messages
- Return the input data unchanged (for chaining)
- Raise appropriate errors when validation fails

Strict Mode:
    By default, these functions raise exceptions on validation failure.
    Set `strict=False` to issue warnings instead (useful for exploratory work).

Example:
    >>> import numpy as np
    >>> from agentbible.domains.atmospheric import (
    ...     check_cloud_base_height,
    ...     check_boundary_layer_height,
    ... )
    >>>
    >>> cbh = np.array([500, 1000, 1500])  # meters
    >>> check_cloud_base_height(cbh, name="cbh")
    >>>
    >>> blh = np.array([200, 800, 1200])  # meters
    >>> check_boundary_layer_height(blh, name="blh")
"""

from __future__ import annotations

import warnings
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike

from agentbible.domains.atmospheric.errors import (
    BoundaryLayerHeightError,
    CloudBaseHeightError,
    CloudLayerConsistencyError,
    LiftingCondensationLevelError,
    RelativeHumidityWarning,
    TemperatureInversionError,
)

# Type variable for array-like inputs
T = TypeVar("T", bound=ArrayLike)


def check_cloud_base_height(
    arr: T,
    *,
    min_height: float = 0.0,
    max_height: float = 12000.0,
    name: str = "cloud_base_height",
    strict: bool = True,
) -> T:
    """Validate that cloud base heights are within physical limits.

    Cloud base height should typically be:
    - Above the surface (> 0 m AGL)
    - Below the tropopause (typically < 12000 m)

    Args:
        arr: Array of cloud base heights in meters AGL.
        min_height: Minimum valid height (default 0 m).
        max_height: Maximum valid height (default 12000 m).
        name: Descriptive name for error messages.
        strict: If True (default), raise error on invalid values.
            If False, issue warning instead.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        CloudBaseHeightError: If any values are outside valid range.

    Example:
        >>> import numpy as np
        >>> from agentbible.domains.atmospheric import check_cloud_base_height
        >>> cbh = np.array([500, 1000, 1500])
        >>> check_cloud_base_height(cbh, name="cbh")  # Passes
    """
    np_arr = np.asarray(arr)

    # Check for NaN/Inf first
    if not np.all(np.isfinite(np_arr)):
        nan_count = int(np.sum(np.isnan(np_arr)))
        inf_count = int(np.sum(np.isinf(np_arr)))
        details = []
        if nan_count > 0:
            details.append(f"{nan_count} NaN")
        if inf_count > 0:
            details.append(f"{inf_count} Inf")

        error = CloudBaseHeightError(
            f"Array '{name}' contains non-finite values",
            expected="All finite cloud base heights",
            got=", ".join(details),
            function_name=f"check_cloud_base_height(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
            return arr

    min_val = float(np.min(np_arr))
    max_val = float(np.max(np_arr))

    if min_val < min_height:
        error = CloudBaseHeightError(
            f"Array '{name}' contains heights below surface",
            expected=f"Cloud base height >= {min_height} m",
            got=f"min = {min_val} m",
            function_name=f"check_cloud_base_height(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
            return arr

    if max_val > max_height:
        error = CloudBaseHeightError(
            f"Array '{name}' contains heights above tropopause",
            expected=f"Cloud base height <= {max_height} m",
            got=f"max = {max_val} m",
            function_name=f"check_cloud_base_height(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return arr


def check_boundary_layer_height(
    arr: T,
    *,
    min_height: float = 10.0,
    max_height: float = 5000.0,
    name: str = "boundary_layer_height",
    strict: bool = True,
) -> T:
    """Validate that boundary layer heights are within physical limits.

    The atmospheric boundary layer height typically ranges from:
    - Nighttime stable BL: 50-500 m
    - Daytime convective BL: 500-3000 m
    - Extreme convective conditions: up to 5000 m

    Args:
        arr: Array of boundary layer heights in meters AGL.
        min_height: Minimum valid height (default 10 m).
        max_height: Maximum valid height (default 5000 m).
        name: Descriptive name for error messages.
        strict: If True (default), raise error on invalid values.
            If False, issue warning instead.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        BoundaryLayerHeightError: If any values are outside valid range.

    Example:
        >>> import numpy as np
        >>> from agentbible.domains.atmospheric import check_boundary_layer_height
        >>> blh = np.array([200, 800, 1200])
        >>> check_boundary_layer_height(blh, name="blh")  # Passes
    """
    np_arr = np.asarray(arr)

    # Check for NaN/Inf first
    if not np.all(np.isfinite(np_arr)):
        nan_count = int(np.sum(np.isnan(np_arr)))
        inf_count = int(np.sum(np.isinf(np_arr)))
        details = []
        if nan_count > 0:
            details.append(f"{nan_count} NaN")
        if inf_count > 0:
            details.append(f"{inf_count} Inf")

        error = BoundaryLayerHeightError(
            f"Array '{name}' contains non-finite values",
            expected="All finite boundary layer heights",
            got=", ".join(details),
            function_name=f"check_boundary_layer_height(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
            return arr

    min_val = float(np.min(np_arr))
    max_val = float(np.max(np_arr))

    if min_val < min_height:
        error = BoundaryLayerHeightError(
            f"Array '{name}' contains implausibly low BLH values",
            expected=f"Boundary layer height >= {min_height} m",
            got=f"min = {min_val} m",
            function_name=f"check_boundary_layer_height(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
            return arr

    if max_val > max_height:
        error = BoundaryLayerHeightError(
            f"Array '{name}' contains implausibly high BLH values",
            expected=f"Boundary layer height <= {max_height} m",
            got=f"max = {max_val} m",
            function_name=f"check_boundary_layer_height(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return arr


def check_lifting_condensation_level(
    arr: T,
    *,
    min_height: float = 0.0,
    max_height: float = 5000.0,
    name: str = "lcl",
    strict: bool = True,
) -> T:
    """Validate that lifting condensation level heights are within limits.

    The LCL is the height at which a parcel becomes saturated when lifted
    adiabatically. It should be:
    - Above the surface (> 0 m AGL)
    - Typically below 5000 m in most conditions

    Args:
        arr: Array of LCL heights in meters AGL.
        min_height: Minimum valid height (default 0 m).
        max_height: Maximum valid height (default 5000 m).
        name: Descriptive name for error messages.
        strict: If True (default), raise error on invalid values.
            If False, issue warning instead.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        LiftingCondensationLevelError: If any values are outside valid range.

    Example:
        >>> import numpy as np
        >>> from agentbible.domains.atmospheric import check_lifting_condensation_level
        >>> lcl = np.array([500, 1000, 1500])
        >>> check_lifting_condensation_level(lcl, name="lcl")  # Passes
    """
    np_arr = np.asarray(arr)

    # Check for NaN/Inf first
    if not np.all(np.isfinite(np_arr)):
        error = LiftingCondensationLevelError(
            f"Array '{name}' contains non-finite values",
            expected="All finite LCL heights",
            got="Contains NaN or Inf",
            function_name=f"check_lifting_condensation_level(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
            return arr

    min_val = float(np.min(np_arr))
    max_val = float(np.max(np_arr))

    if min_val < min_height:
        error = LiftingCondensationLevelError(
            f"Array '{name}' contains negative LCL values",
            expected=f"LCL >= {min_height} m",
            got=f"min = {min_val} m",
            function_name=f"check_lifting_condensation_level(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
            return arr

    if max_val > max_height:
        error = LiftingCondensationLevelError(
            f"Array '{name}' contains implausibly high LCL values",
            expected=f"LCL <= {max_height} m",
            got=f"max = {max_val} m",
            function_name=f"check_lifting_condensation_level(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return arr


def check_cloud_layer_consistency(
    cloud_base: ArrayLike,
    cloud_top: ArrayLike,
    *,
    name: str = "cloud_layer",
    strict: bool = True,
) -> tuple[ArrayLike, ArrayLike]:
    """Validate that cloud base is below cloud top.

    Cloud layers must satisfy the physical constraint that the base
    is below the top.

    Args:
        cloud_base: Array of cloud base heights in meters.
        cloud_top: Array of cloud top heights in meters.
        name: Descriptive name for error messages.
        strict: If True (default), raise error on invalid layers.
            If False, issue warning instead.

    Returns:
        Tuple of (cloud_base, cloud_top) unchanged.

    Raises:
        CloudLayerConsistencyError: If any base >= top.

    Example:
        >>> import numpy as np
        >>> from agentbible.domains.atmospheric import check_cloud_layer_consistency
        >>> base = np.array([500, 1000])
        >>> top = np.array([1500, 2000])
        >>> check_cloud_layer_consistency(base, top)  # Passes
    """
    base_arr = np.asarray(cloud_base)
    top_arr = np.asarray(cloud_top)

    if base_arr.shape != top_arr.shape:
        raise ValueError(
            f"Shape mismatch: cloud_base {base_arr.shape}, cloud_top {top_arr.shape}"
        )

    # Check for inverted layers
    inverted = base_arr >= top_arr
    if np.any(inverted):
        n_inverted = int(np.sum(inverted))
        error = CloudLayerConsistencyError(
            f"Cloud layer '{name}' has {n_inverted} inverted layer(s)",
            expected="Cloud base < cloud top for all layers",
            got=f"{n_inverted} layers with base >= top",
            function_name=f"check_cloud_layer_consistency(name='{name}')",
        )

        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return (cloud_base, cloud_top)


def check_relative_humidity(
    arr: T,
    *,
    allow_supersaturation: bool = False,
    max_supersaturation: float = 105.0,
    name: str = "relative_humidity",
    strict: bool = True,
) -> T:
    """Validate that relative humidity is within physical limits.

    Relative humidity should be in [0, 100]%, though slight supersaturation
    (up to ~105%) can occur in clouds.

    Args:
        arr: Array of relative humidity values in percent (0-100).
        allow_supersaturation: Allow values slightly > 100%.
        max_supersaturation: Maximum allowed value if supersaturation allowed.
        name: Descriptive name for error messages.
        strict: If True (default), raise error on invalid values.
            If False, issue warning instead.

    Returns:
        The input array unchanged (allows chaining).

    Raises:
        ValueError: If values are outside [0, max] range.

    Example:
        >>> import numpy as np
        >>> from agentbible.domains.atmospheric import check_relative_humidity
        >>> rh = np.array([50, 75, 90])
        >>> check_relative_humidity(rh, name="rh")  # Passes
    """
    np_arr = np.asarray(arr)

    max_allowed = max_supersaturation if allow_supersaturation else 100.0

    min_val = float(np.min(np_arr))
    max_val = float(np.max(np_arr))

    if min_val < 0:
        msg = (
            f"Relative humidity '{name}' contains negative values "
            f"(min = {min_val}%). RH must be >= 0%."
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RelativeHumidityWarning, stacklevel=2)
            return arr

    if max_val > max_allowed:
        msg = (
            f"Relative humidity '{name}' exceeds {max_allowed}% "
            f"(max = {max_val}%). Check for measurement errors."
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RelativeHumidityWarning, stacklevel=2)

    return arr


def check_temperature_inversion(
    base_height: ArrayLike,
    top_height: ArrayLike,
    base_temp: ArrayLike,
    top_temp: ArrayLike,
    *,
    max_strength: float = 25.0,
    name: str = "inversion",
    strict: bool = True,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Validate temperature inversion properties.

    Temperature inversions must satisfy:
    - Inversion base < inversion top
    - Temperature at top > temperature at base
    - Inversion strength typically < 20 K

    Args:
        base_height: Heights of inversion bases in meters.
        top_height: Heights of inversion tops in meters.
        base_temp: Temperatures at inversion bases in K.
        top_temp: Temperatures at inversion tops in K.
        max_strength: Maximum allowed inversion strength in K.
        name: Descriptive name for error messages.
        strict: If True (default), raise error on invalid inversions.

    Returns:
        Tuple of all input arrays unchanged.

    Raises:
        TemperatureInversionError: If inversion properties are invalid.
    """
    base_h = np.asarray(base_height)
    top_h = np.asarray(top_height)
    base_t = np.asarray(base_temp)
    top_t = np.asarray(top_temp)

    # Check height ordering
    if np.any(base_h >= top_h):
        error = TemperatureInversionError(
            f"Inversion '{name}' has base >= top",
            expected="Inversion base height < top height",
            got="One or more inversions with base >= top",
            function_name=f"check_temperature_inversion(name='{name}')",
        )
        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
            return (base_height, top_height, base_temp, top_temp)

    # Check temperature ordering (top should be warmer for inversion)
    strength = top_t - base_t
    if np.any(strength <= 0):
        error = TemperatureInversionError(
            f"Inversion '{name}' has temperature decreasing with height",
            expected="Temperature at top > temperature at base",
            got="One or more inversions with non-positive temperature difference",
            function_name=f"check_temperature_inversion(name='{name}')",
        )
        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
            return (base_height, top_height, base_temp, top_temp)

    # Check strength
    if np.any(strength > max_strength):
        max_observed = float(np.max(strength))
        error = TemperatureInversionError(
            f"Inversion '{name}' has unusually strong inversion",
            expected=f"Inversion strength <= {max_strength} K",
            got=f"max strength = {max_observed:.1f} K",
            function_name=f"check_temperature_inversion(name='{name}')",
        )
        if strict:
            raise error
        else:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return (base_height, top_height, base_temp, top_temp)


__all__ = [
    "check_cloud_base_height",
    "check_boundary_layer_height",
    "check_lifting_condensation_level",
    "check_cloud_layer_consistency",
    "check_relative_humidity",
    "check_temperature_inversion",
]
