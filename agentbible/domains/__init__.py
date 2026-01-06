"""Domain-specific validation plugins for AgentBible.

Each domain provides specialized validators for a specific scientific field.
Domains are optional and can be imported explicitly when needed.

Available domains:
    - quantum: Quantum computing validators (unitarity, hermiticity, density matrices)

Example:
    >>> from agentbible.domains.quantum import validate_unitary
    >>> import numpy as np
    >>>
    >>> @validate_unitary
    ... def create_hadamard():
    ...     return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

Future domains (contributions welcome):
    - chemistry: Molecular structure and thermodynamics validators
    - fluids: CFD and fluid dynamics validators
    - electromagnetism: Maxwell's equations validators
"""

from __future__ import annotations

# Domains are imported explicitly by the user, not auto-loaded here.
# This keeps the core package lightweight and allows lazy loading.

__all__: list[str] = []
