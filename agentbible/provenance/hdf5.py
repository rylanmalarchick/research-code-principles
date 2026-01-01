"""HDF5 provenance tracking for research data.

Save and load numpy arrays with full reproducibility metadata including:
- Git commit SHA, branch, and dirty status
- Random seeds (numpy, python, torch)
- Timestamps and hardware info
- Package versions
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Type alias for metadata
Metadata = dict[str, Any]


def _get_git_info() -> dict[str, Any]:
    """Get git repository information."""
    git_info: dict[str, Any] = {
        "git_sha": None,
        "git_branch": None,
        "git_dirty": None,
    }

    try:
        # Get current commit SHA
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["git_sha"] = result.stdout.strip()

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["git_branch"] = result.stdout.strip()

        # Check if repo is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["git_dirty"] = len(result.stdout.strip()) > 0

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return git_info


def _get_numpy_seed() -> int | None:
    """Try to get the numpy random seed if available."""
    # numpy doesn't expose the seed directly, but we can get the state
    # This returns None since we can't reliably get the original seed
    # Users should set seeds explicitly and we'll capture them
    return None


def _get_python_seed() -> int | None:
    """Try to get the Python random seed if available."""
    # Python's random module doesn't expose the seed
    return None


def _get_torch_seed() -> int | None:
    """Try to get the PyTorch random seed if available."""
    try:
        import torch

        # torch.initial_seed() returns the seed used to initialize the generator
        return int(torch.initial_seed())
    except (ImportError, RuntimeError):
        return None


def _get_package_versions() -> dict[str, str]:
    """Get versions of key scientific packages."""
    packages: dict[str, str] = {}

    # Always include numpy
    packages["numpy"] = np.__version__

    # Try to get other common packages
    optional_packages = ["scipy", "h5py", "torch", "jax", "pandas"]
    for pkg_name in optional_packages:
        try:
            module = __import__(pkg_name)
            packages[pkg_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            pass

    return packages


def get_provenance_metadata(
    description: str = "",
    extra: dict[str, Any] | None = None,
) -> Metadata:
    """Generate provenance metadata dictionary.

    Args:
        description: User-provided description of the data
        extra: Additional metadata to include

    Returns:
        Dictionary containing provenance metadata
    """
    metadata: Metadata = {
        # User description
        "description": description,
        # Timestamp
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # Git info
        **_get_git_info(),
        # Random seeds
        "numpy_seed": _get_numpy_seed(),
        "python_seed": _get_python_seed(),
        "torch_seed": _get_torch_seed(),
        # System info
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        # Package versions
        "packages": _get_package_versions(),
    }

    # Add extra metadata if provided
    if extra:
        metadata["extra"] = extra

    return metadata


def save_with_metadata(
    filepath: str | Path,
    data: dict[str, np.ndarray],
    description: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Save numpy arrays to HDF5 with provenance metadata.

    Args:
        filepath: Path to save the HDF5 file
        data: Dictionary mapping dataset names to numpy arrays
        description: User-provided description of the data
        extra: Additional metadata to include

    Raises:
        ImportError: If h5py is not installed
        ValueError: If data is empty or contains non-array values

    Example:
        >>> import numpy as np
        >>> from agentbible.provenance import save_with_metadata
        >>> save_with_metadata(
        ...     "results.h5",
        ...     {"density_matrix": np.eye(2), "eigenvalues": np.array([1, 0])},
        ...     description="Ground state calculation",
        ... )
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 provenance. "
            "Install with: pip install agentbible[hdf5]"
        ) from e

    if not data:
        raise ValueError("data dictionary cannot be empty")

    filepath = Path(filepath)
    metadata = get_provenance_metadata(description, extra)

    with h5py.File(filepath, "w") as f:
        # Save each array as a dataset
        for name, array in data.items():
            if not isinstance(array, np.ndarray):
                raise ValueError(f"Value for '{name}' must be a numpy array")
            f.create_dataset(name, data=array)

        # Save metadata as JSON string in root attributes
        f.attrs["provenance"] = json.dumps(metadata)
        f.attrs["provenance_version"] = "1.0"


def load_with_metadata(
    filepath: str | Path,
) -> tuple[dict[str, np.ndarray], Metadata]:
    """Load numpy arrays from HDF5 with provenance metadata.

    Args:
        filepath: Path to the HDF5 file

    Returns:
        Tuple of (data_dict, metadata_dict)

    Raises:
        ImportError: If h5py is not installed
        FileNotFoundError: If file doesn't exist

    Example:
        >>> from agentbible.provenance import load_with_metadata
        >>> data, metadata = load_with_metadata("results.h5")
        >>> print(metadata["git_sha"])
        'a1b2c3d4...'
        >>> print(data["density_matrix"].shape)
        (2, 2)
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 provenance. "
            "Install with: pip install agentbible[hdf5]"
        ) from e

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data: dict[str, np.ndarray] = {}
    metadata: Metadata = {}

    with h5py.File(filepath, "r") as f:
        # Load all datasets
        for name in f.keys():
            data[name] = np.array(f[name])

        # Load metadata from attributes
        if "provenance" in f.attrs:
            metadata = json.loads(f.attrs["provenance"])

    return data, metadata
