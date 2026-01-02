# CLI Reference

The `bible` command provides project management and validation tools.

## Global Options

```bash
bible --help     # Show help
bible --version  # Show version
```

## Commands

### `bible init`

Create a new project from a template.

```bash
bible init <project-name> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project-name` | Name of the project directory to create |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--template`, `-t` | `python-scientific` | Template to use |
| `--author` | Git config | Author name |
| `--email` | Git config | Author email |
| `--description` | Generic | Project description |
| `--no-git` | False | Skip git initialization |
| `--no-venv` | False | Skip virtual environment |
| `--force` | False | Overwrite existing directory |

**Available Templates:**

- `python-scientific` - Python project with ruff, mypy, pytest
- `cpp-hpc-cuda` - C++ project with CMake, GoogleTest

**Examples:**

```bash
# Basic Python project
bible init my-project

# C++ project with custom author
bible init hpc-solver --template cpp-hpc-cuda --author "Jane Doe"

# Python project without git/venv
bible init quick-test --no-git --no-venv

# Force overwrite existing
bible init my-project --force
```

---

### `bible validate`

Validate physics constraints in data files.

```bash
bible validate <file> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `file` | Path to data file (.npy, .npz, .h5, .hdf5) |

**Options:**

| Option | Description |
|--------|-------------|
| `--check`, `-c` | Constraint to check (can repeat) |
| `--key`, `-k` | Key/dataset to validate (for .npz/.h5) |
| `--tolerance`, `-tol` | Numerical tolerance |
| `--verbose`, `-v` | Show detailed output |

**Available Checks:**

| Check | Validates |
|-------|-----------|
| `unitarity` | U @ U.H = I |
| `hermiticity` | A = A.H |
| `trace` | Tr(A) = 1 |
| `positivity` | All eigenvalues ≥ 0 |
| `normalization` | Sum = 1 or norm = 1 |
| `all` | Run all applicable checks |

**Examples:**

```bash
# Check unitarity of a matrix
bible validate gate.npy --check unitarity

# Multiple checks
bible validate density.h5 -c hermiticity -c trace -c positivity

# All checks with verbose output
bible validate results.h5 --check all --verbose

# Specific dataset in HDF5
bible validate results.h5 --key matrices/hamiltonian --check hermiticity

# Custom tolerance
bible validate approx_unitary.npy -c unitarity --tolerance 1e-6
```

**Output:**

```
$ bible validate gate.npy --check unitarity
✓ unitarity: PASSED (deviation: 1.2e-15)

$ bible validate bad_gate.npy --check unitarity
✗ unitarity: FAILED (deviation: 0.42)
  Matrix is not unitary. Maximum deviation from identity: 0.42
```

---

### `bible context`

Generate AI context for the current project.

```bash
bible context [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--all` | Load all context files |
| `--query`, `-q` | Search for specific context |
| `--dir`, `-d` | Context directory |

**Examples:**

```bash
# Auto-detect and load default context
bible context

# Load all context from directory
bible context --all ./agent_docs

# Search for specific topic
bible context --query "error handling"
```

!!! note
    This command wraps the `opencode-context` tool if installed.

---

### `bible info`

Show installation and environment information.

```bash
bible info
```

**Output:**

```
AgentBible v0.1.1

Installation:
  Location: /path/to/.venv/lib/python3.12/site-packages/agentbible
  Python: 3.12.0

Optional Dependencies:
  h5py: 3.10.0 ✓
  torch: not installed
  chromadb: not installed

Templates:
  python-scientific: available
  cpp-hpc-cuda: available
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Validation failed |
| 3 | File not found |
| 4 | Invalid arguments |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AGENTBIBLE_SKIP_VALIDATION` | Set to "1" to skip runtime validation |
| `AGENTBIBLE_VERBOSE` | Set to "1" for verbose output |

## Shell Completion

Enable shell completion:

```bash
# Bash
eval "$(_BIBLE_COMPLETE=bash_source bible)"

# Zsh
eval "$(_BIBLE_COMPLETE=zsh_source bible)"

# Fish
_BIBLE_COMPLETE=fish_source bible | source
```

Add to your shell profile for persistence.
