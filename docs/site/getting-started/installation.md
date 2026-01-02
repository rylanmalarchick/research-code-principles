# Installation

## Requirements

- Python 3.9 or higher
- pip (Python package installer)

## Basic Installation

Install AgentBible from PyPI:

```bash
pip install agentbible
```

This installs the core package with validators and CLI.

## Optional Dependencies

### HDF5 Provenance Support

For saving data with full reproducibility metadata:

```bash
pip install agentbible[hdf5]
```

This adds `h5py` for HDF5 file support.

### Vector Context (Experimental)

For AI context management with embeddings:

```bash
pip install agentbible[vector]
```

This adds `chromadb`, `openai`, and `tiktoken`.

### Full Development Install

For development with all dependencies:

```bash
pip install agentbible[all]
```

## Verify Installation

After installation, verify it works:

```bash
# Check CLI
bible --version
bible info

# Check Python import
python -c "from agentbible import validate_unitary; print('OK')"
```

## Development Setup

For contributing or modifying AgentBible:

```bash
# Clone the repository
git clone https://github.com/rylanmalarchick/research-code-principles
cd research-code-principles

# Option 1: Use bootstrap script
./bootstrap.sh

# Option 2: Manual setup
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,hdf5]"

# Run tests
pytest tests/ -v
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade agentbible
```

## Troubleshooting

### Import Errors

If you get import errors for optional dependencies:

```bash
# For HDF5 features
pip install h5py>=3.0

# For torch seed capture (optional)
pip install torch
```

### CLI Not Found

If `bible` command is not found after installation:

```bash
# Check if scripts directory is in PATH
python -m agentbible.cli.main --help

# Or reinstall
pip install --force-reinstall agentbible
```
