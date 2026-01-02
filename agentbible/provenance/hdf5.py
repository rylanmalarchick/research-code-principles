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
import platform
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
        "git_diff": None,
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
            is_dirty = len(result.stdout.strip()) > 0
            git_info["git_dirty"] = is_dirty

            # If dirty, capture the diff for reproducibility
            if is_dirty:
                diff_result = subprocess.run(
                    ["git", "diff", "--no-color"],
                    capture_output=True,
                    text=True,
                    timeout=30,  # Longer timeout for potentially large diffs
                )
                if diff_result.returncode == 0:
                    diff_text = diff_result.stdout
                    # Limit diff size to 100KB to avoid bloated HDF5 files
                    max_diff_size = 100_000
                    if len(diff_text) > max_diff_size:
                        git_info["git_diff"] = (
                            diff_text[:max_diff_size]
                            + f"\n... [TRUNCATED: diff was {len(diff_text)} bytes] ..."
                        )
                        git_info["git_diff_truncated"] = True
                    else:
                        git_info["git_diff"] = diff_text
                        git_info["git_diff_truncated"] = False

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


def _get_pip_freeze() -> list[str]:
    """Get full pip freeze output for exact reproducibility.

    Returns:
        List of package specs (e.g., ["numpy==1.24.0", "scipy==1.11.0"])
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Filter out empty lines and editable installs (too long)
            lines = result.stdout.strip().split("\n")
            return [line for line in lines if line and not line.startswith("-e ")]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return []


def _get_hardware_info() -> dict[str, Any]:
    """Get hardware information for reproducibility.

    Returns:
        Dictionary with CPU, memory, and GPU info
    """
    hw_info: dict[str, Any] = {
        "cpu_model": None,
        "cpu_count_physical": None,
        "cpu_count_logical": None,
        "memory_total_gb": None,
        "gpu_info": None,
    }

    # CPU info
    try:
        import os

        hw_info["cpu_count_logical"] = os.cpu_count()

        # Try to get physical core count
        try:
            import multiprocessing

            hw_info["cpu_count_physical"] = multiprocessing.cpu_count()
        except Exception:
            pass

        # Try to get CPU model from /proc/cpuinfo (Linux)
        try:
            with Path("/proc/cpuinfo").open() as f:
                for line in f:
                    if line.startswith("model name"):
                        hw_info["cpu_model"] = line.split(":")[1].strip()
                        break
        except (FileNotFoundError, OSError):
            # Try platform.processor() as fallback
            processor = platform.processor()
            if processor:
                hw_info["cpu_model"] = processor

    except Exception:
        pass

    # Memory info
    try:
        # Try to get memory from /proc/meminfo (Linux)
        with Path("/proc/meminfo").open() as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Convert from KB to GB
                    mem_kb = int(line.split()[1])
                    hw_info["memory_total_gb"] = round(mem_kb / (1024 * 1024), 2)
                    break
    except (FileNotFoundError, OSError, ValueError):
        pass

    # GPU info (try nvidia-smi for NVIDIA GPUs, then check torch)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) >= 3:
                    gpus.append(
                        {
                            "name": parts[0].strip(),
                            "memory": parts[1].strip(),
                            "driver_version": parts[2].strip(),
                        }
                    )
                elif len(parts) >= 2:
                    gpus.append({"name": parts[0].strip(), "memory": parts[1].strip()})
            if gpus:
                hw_info["gpu_info"] = gpus
                # Also get CUDA version if available
                cuda_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if cuda_result.returncode == 0 and cuda_result.stdout.strip():
                    hw_info["cuda_version"] = cuda_result.stdout.strip().split("\n")[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # Try torch.cuda as fallback
        try:
            import torch

            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append(
                        {
                            "name": props.name,
                            "memory": f"{props.total_memory / (1024**3):.1f} GB",
                        }
                    )
                if gpus:
                    hw_info["gpu_info"] = gpus
        except (ImportError, RuntimeError):
            pass

    return hw_info


def get_provenance_metadata(
    description: str = "",
    extra: dict[str, Any] | None = None,
    include_pip_freeze: bool = True,
    include_hardware: bool = True,
) -> Metadata:
    """Generate provenance metadata dictionary.

    Args:
        description: User-provided description of the data
        extra: Additional metadata to include
        include_pip_freeze: Whether to include full pip freeze output
        include_hardware: Whether to include hardware info

    Returns:
        Dictionary containing provenance metadata
    """
    metadata: Metadata = {
        # User description
        "description": description,
        # Timestamp
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # Git info (includes git_diff if repo is dirty)
        **_get_git_info(),
        # Random seeds
        "numpy_seed": _get_numpy_seed(),
        "python_seed": _get_python_seed(),
        "torch_seed": _get_torch_seed(),
        # System info
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        # Package versions (key packages for quick reference)
        "packages": _get_package_versions(),
    }

    # Full pip freeze for exact reproducibility
    if include_pip_freeze:
        metadata["pip_freeze"] = _get_pip_freeze()

    # Hardware info for understanding performance context
    if include_hardware:
        metadata["hardware"] = _get_hardware_info()

    # Add extra metadata if provided
    if extra:
        metadata["extra"] = extra

    return metadata


def save_with_metadata(
    filepath: str | Path,
    data: dict[str, np.ndarray],
    description: str = "",
    extra: dict[str, Any] | None = None,
    include_pip_freeze: bool = True,
    include_hardware: bool = True,
) -> None:
    """Save numpy arrays to HDF5 with provenance metadata.

    Args:
        filepath: Path to save the HDF5 file
        data: Dictionary mapping dataset names to numpy arrays
        description: User-provided description of the data
        extra: Additional metadata to include
        include_pip_freeze: Whether to include full pip freeze output
        include_hardware: Whether to include hardware info

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
    metadata = get_provenance_metadata(
        description, extra, include_pip_freeze, include_hardware
    )

    with h5py.File(filepath, "w") as f:
        # Save each array as a dataset
        for name, array in data.items():
            if not isinstance(array, np.ndarray):
                raise ValueError(f"Value for '{name}' must be a numpy array")
            f.create_dataset(name, data=array)

        # Save metadata as JSON string in root attributes
        f.attrs["provenance"] = json.dumps(metadata)
        f.attrs["provenance_version"] = (
            "2.0"  # v2.0 adds pip_freeze, hardware, git_diff
        )


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
        for name in f:
            data[name] = np.array(f[name])

        # Load metadata from attributes
        if "provenance" in f.attrs:
            metadata = json.loads(f.attrs["provenance"])

    return data, metadata
