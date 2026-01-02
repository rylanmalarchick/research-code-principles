# Provenance Tracking

AgentBible's provenance module captures complete reproducibility metadata when saving data.

## Why Provenance Matters

Research results must be reproducible. When you save a file, you need to know:

- What code version produced it?
- When was it created?
- What random seeds were used?
- What package versions were installed?

AgentBible captures all of this automatically.

## Basic Usage

### Saving Data

```python
from agentbible.provenance import save_with_metadata
import numpy as np

# Create some data
result = np.random.rand(100, 100)
eigenvalues = np.linalg.eigvalsh(result @ result.T)

# Save with full provenance
save_with_metadata(
    "results.h5",
    {"matrix": result, "eigenvalues": eigenvalues},
    description="Random matrix eigenvalue analysis",
)
```

### Loading Data

```python
from agentbible.provenance import load_with_metadata

# Load data and metadata
data, metadata = load_with_metadata("results.h5")

print(data["matrix"].shape)        # (100, 100)
print(metadata["git_sha"])         # "a1b2c3d..."
print(metadata["timestamp"])       # "2026-01-01T12:00:00+00:00"
print(metadata["packages"])        # {"numpy": "1.26.0", ...}
```

## Captured Metadata

### Git Information

| Field | Description |
|-------|-------------|
| `git_sha` | Full commit SHA |
| `git_branch` | Current branch name |
| `git_dirty` | True if uncommitted changes |
| `git_diff` | Diff of uncommitted changes (if dirty) |

### Timing

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 UTC timestamp |
| `timezone` | Local timezone |

### Random Seeds

| Field | Description |
|-------|-------------|
| `numpy_seed` | NumPy random state (if capturable) |
| `python_seed` | Python random seed |
| `torch_seed` | PyTorch seed (if torch installed) |

### System Information

| Field | Description |
|-------|-------------|
| `hostname` | Machine hostname |
| `platform` | OS and version |
| `python_version` | Python version |
| `cpu_count` | Number of CPUs |

### Package Versions

Automatically captures versions of installed packages:

- numpy, scipy, h5py
- torch, tensorflow (if installed)
- agentbible itself

## Advanced Usage

### Custom Metadata

Add your own metadata:

```python
save_with_metadata(
    "results.h5",
    {"data": array},
    description="Experiment results",
    custom_metadata={
        "experiment_id": "exp_001",
        "parameters": {"learning_rate": 0.01, "epochs": 100},
        "notes": "First successful run",
    }
)
```

### Specifying Seeds

Explicitly set seeds for reproducibility:

```python
from agentbible.provenance import save_with_metadata, set_all_seeds

# Set all random seeds
set_all_seeds(42)

# Run computation
result = run_simulation()

# Save - seeds are captured automatically
save_with_metadata("results.h5", {"result": result})
```

### Getting Metadata Without Saving

```python
from agentbible.provenance import get_provenance_metadata

metadata = get_provenance_metadata(
    description="Current experiment state"
)

print(metadata["git_sha"])
print(metadata["packages"])
```

## File Format

Data is stored in HDF5 format:

```
results.h5
├── data/
│   ├── matrix          # Your numpy arrays
│   └── eigenvalues
└── provenance/
    ├── git_sha
    ├── timestamp
    ├── packages        # JSON-encoded
    └── ...
```

## Programmatic Access

### Inspecting Files

```python
from agentbible.provenance import inspect_provenance

# Print formatted provenance info
inspect_provenance("results.h5")

# Get as dictionary
info = inspect_provenance("results.h5", return_dict=True)
```

### CLI Inspection

```bash
bible validate results.h5 --show-provenance
```

## Best Practices

### 1. Always Use Descriptions

```python
save_with_metadata(
    "results.h5",
    data,
    description="VQE ground state, H2 molecule, 1.4 Angstrom bond length",
)
```

### 2. Commit Before Saving

Ensure your code is committed so `git_sha` points to reproducible code:

```python
if metadata["git_dirty"]:
    warnings.warn("Uncommitted changes - results may not be reproducible")
```

### 3. Use Explicit Seeds

```python
from agentbible.provenance import set_all_seeds

SEED = 42
set_all_seeds(SEED)

# Now results are reproducible
```

### 4. Version Your Data Files

Include version in filename or metadata:

```python
save_with_metadata(
    f"results_v{VERSION}.h5",
    data,
    custom_metadata={"data_version": VERSION}
)
```

## Reproducing Results

To reproduce a result from a saved file:

```python
from agentbible.provenance import load_with_metadata, set_all_seeds
import subprocess

# Load the file
data, meta = load_with_metadata("results.h5")

# Check out the exact code version
subprocess.run(["git", "checkout", meta["git_sha"]])

# Set the same seeds
set_all_seeds(meta.get("numpy_seed", 42))

# Install same package versions
# pip install numpy==1.26.0 ...

# Re-run
result = run_simulation()
assert np.allclose(result, data["result"])
```
