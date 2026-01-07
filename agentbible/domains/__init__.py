"""Domain-specific validation plugins for AgentBible.

Each domain provides specialized validators for a specific scientific field.
Domains are optional and can be imported explicitly when needed.

Available domains:
    - quantum: Quantum computing validators (unitarity, hermiticity, density matrices)
    - ml: Machine learning validators (data leakage, coverage, exchangeability)
    - atmospheric: Atmospheric science validators (CBH, BLH, cloud layers)

Example:
    >>> from agentbible.domains.quantum import validate_unitary
    >>> import numpy as np
    >>>
    >>> @validate_unitary
    ... def create_hadamard():
    ...     return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    >>> from agentbible.domains.ml import check_no_leakage, check_coverage
    >>> FORBIDDEN = {"target_lagged", "inversion_height"}
    >>> check_no_leakage(["blh", "t2m"], forbidden=FORBIDDEN)

    >>> from agentbible.domains.atmospheric import check_cloud_base_height
    >>> check_cloud_base_height(np.array([500, 1000, 1500]), name="cbh")

Future domains (contributions welcome):
    - chemistry: Molecular structure and thermodynamics validators
    - fluids: CFD and fluid dynamics validators
    - electromagnetism: Maxwell's equations validators
"""

from __future__ import annotations

# Domains are imported explicitly by the user, not auto-loaded here.
# This keeps the core package lightweight and allows lazy loading.

__all__: list[str] = []
