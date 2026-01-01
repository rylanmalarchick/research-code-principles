"""Provenance tracking for research data.

Provides functions to save and load data with full reproducibility metadata.
"""

from __future__ import annotations

from agentbible.provenance.hdf5 import (
    get_provenance_metadata,
    load_with_metadata,
    save_with_metadata,
)

__all__ = [
    "save_with_metadata",
    "load_with_metadata",
    "get_provenance_metadata",
]
