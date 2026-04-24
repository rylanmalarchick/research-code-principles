"""HDF5 helpers built on the shared provenance record model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agentbible.provenance.record import get_provenance_metadata

Metadata = dict[str, Any]


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "h5py is required for HDF5 provenance. Install with: pip install agentbible[hdf5]"
        ) from exc
    return h5py


def save_with_metadata(
    filepath: str | Path,
    data: dict[str, np.ndarray],
    description: str = "",
    extra: dict[str, Any] | None = None,
    include_pip_freeze: bool = True,
    include_hardware: bool = True,
) -> None:
    """Save numpy arrays to HDF5 with provenance metadata."""
    if not data:
        raise ValueError("data dictionary cannot be empty")
    for name, array in data.items():
        if not isinstance(array, np.ndarray):
            raise ValueError(f"Value for '{name}' must be a numpy array")
    h5py = _require_h5py()
    metadata = get_provenance_metadata(
        description,
        extra,
        include_pip_freeze,
        include_hardware,
    )
    with h5py.File(Path(filepath), "w") as handle:
        for name, array in data.items():
            handle.create_dataset(name, data=array)
        handle.attrs["provenance"] = json.dumps(metadata)
        handle.attrs["provenance_version"] = "2.0"


def load_with_metadata(filepath: str | Path) -> tuple[dict[str, np.ndarray], Metadata]:
    """Load numpy arrays and provenance metadata from HDF5."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    h5py = _require_h5py()
    data: dict[str, np.ndarray] = {}
    metadata: Metadata = {}
    with h5py.File(path, "r") as handle:
        for name in handle:
            data[name] = np.array(handle[name])
        if "provenance" in handle.attrs:
            metadata = json.loads(handle.attrs["provenance"])
    return data, metadata
