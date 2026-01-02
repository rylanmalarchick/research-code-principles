# API Reference: Provenance

::: agentbible.provenance

## Core Functions

### save_with_metadata

```python
from agentbible.provenance import save_with_metadata

save_with_metadata(
    path: str,
    data: Dict[str, np.ndarray],
    description: str = "",
    custom_metadata: Optional[Dict] = None,
) -> None
```

Save numpy arrays to HDF5 with full provenance metadata.

**Parameters:**

- `path` (str): Output file path (.h5 or .hdf5)
- `data` (Dict[str, np.ndarray]): Dictionary of arrays to save
- `description` (str): Human-readable description of the data
- `custom_metadata` (Dict, optional): Additional metadata to include

**Example:**

```python
save_with_metadata(
    "results.h5",
    {"matrix": np.eye(4), "vector": np.zeros(4)},
    description="Test matrices",
    custom_metadata={"experiment_id": "exp_001"}
)
```

---

### load_with_metadata

```python
from agentbible.provenance import load_with_metadata

data, metadata = load_with_metadata(path: str) -> Tuple[Dict, Dict]
```

Load data and provenance metadata from HDF5 file.

**Parameters:**

- `path` (str): Input file path

**Returns:**

- `data` (Dict[str, np.ndarray]): Dictionary of loaded arrays
- `metadata` (Dict): Provenance metadata

**Example:**

```python
data, meta = load_with_metadata("results.h5")
print(data["matrix"].shape)
print(meta["git_sha"])
print(meta["timestamp"])
```

---

### get_provenance_metadata

```python
from agentbible.provenance import get_provenance_metadata

metadata = get_provenance_metadata(
    description: str = "",
    include_diff: bool = False,
) -> Dict
```

Generate provenance metadata without saving.

**Parameters:**

- `description` (str): Description to include
- `include_diff` (bool): Include git diff if repo is dirty

**Returns:**

Dictionary with keys:

- `git_sha`: Commit SHA
- `git_branch`: Branch name
- `git_dirty`: Whether there are uncommitted changes
- `git_diff`: Diff content (if dirty and include_diff=True)
- `timestamp`: ISO 8601 UTC timestamp
- `hostname`: Machine hostname
- `platform`: OS information
- `python_version`: Python version
- `packages`: Dict of installed package versions
- `description`: Provided description

---

### set_all_seeds

```python
from agentbible.provenance import set_all_seeds

set_all_seeds(seed: int) -> None
```

Set random seeds for numpy, Python random, and torch (if available).

**Parameters:**

- `seed` (int): Seed value

**Example:**

```python
set_all_seeds(42)
# Now numpy.random, random, and torch are seeded
```

---

## Inspection Functions

### inspect_provenance

```python
from agentbible.provenance import inspect_provenance

inspect_provenance(
    path: str,
    return_dict: bool = False,
) -> Optional[Dict]
```

Display or return provenance information from an HDF5 file.

**Parameters:**

- `path` (str): Path to HDF5 file
- `return_dict` (bool): If True, return dict instead of printing

**Example:**

```python
# Print formatted output
inspect_provenance("results.h5")

# Get as dictionary
info = inspect_provenance("results.h5", return_dict=True)
```

---

## Metadata Schema

The provenance metadata dictionary contains:

```python
{
    # Git information
    "git_sha": "a1b2c3d4e5f6...",
    "git_branch": "main",
    "git_dirty": False,
    "git_diff": "",  # Only if dirty
    
    # Timing
    "timestamp": "2026-01-01T12:00:00+00:00",
    "timezone": "UTC",
    
    # Random seeds
    "numpy_seed": 42,  # If capturable
    "python_seed": 42,
    "torch_seed": 42,  # If torch installed
    
    # System
    "hostname": "workstation",
    "platform": "Linux-5.15.0-x86_64",
    "python_version": "3.12.0",
    "cpu_count": 8,
    
    # Packages
    "packages": {
        "numpy": "1.26.0",
        "scipy": "1.11.0",
        "h5py": "3.10.0",
        "agentbible": "0.1.1",
    },
    
    # User-provided
    "description": "Experiment results",
    "custom": {...},  # custom_metadata
}
```

---

## HDF5 File Structure

Files saved with `save_with_metadata` have this structure:

```
file.h5
├── data/
│   ├── array_name_1      # numpy array
│   ├── array_name_2      # numpy array
│   └── ...
└── provenance/
    ├── git_sha           # string attribute
    ├── git_branch        # string attribute
    ├── timestamp         # string attribute
    ├── packages          # JSON-encoded string
    └── ...
```
