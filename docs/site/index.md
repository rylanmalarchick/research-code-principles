# AgentBible

[![CI](https://github.com/rylanmalarchick/research-code-principles/actions/workflows/ci.yml/badge.svg)](https://github.com/rylanmalarchick/research-code-principles/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/agentbible.svg)](https://badge.fury.io/py/agentbible)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/gh/rylanmalarchick/research-code-principles/branch/main/graph/badge.svg)](https://codecov.io/gh/rylanmalarchick/research-code-principles)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade infrastructure for AI-assisted research software.**

AgentBible provides physics validation decorators, project scaffolding, provenance tracking, and testing utilities for scientific Python projects.

## Why AgentBible?

Research code is different. It needs to be correct, reproducible, and maintainable by both humans and AI agents. AgentBible provides:

- **Physics Validators** - Catch quantum/physics errors at runtime
- **Provenance Tracking** - Full reproducibility metadata in HDF5 files
- **Project Templates** - Start with best practices baked in
- **AI-Native Design** - `.cursorrules` and context management for AI coding

## Quick Start

```bash
# Install
pip install agentbible

# Create a new project
bible init my-quantum-sim --template python-scientific

# Use validators in your code
```

```python
from agentbible import validate_unitary
import numpy as np

@validate_unitary
def create_hadamard():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

H = create_hadamard()  # Validated automatically
```

## Features

<div class="grid cards" markdown>

-   :material-check-circle:{ .lg .middle } __Physics Validators__

    ---

    Decorators for quantum/physics validation: unitarity, hermiticity, density matrices, probabilities, and more.

    [:octicons-arrow-right-24: Validators](guide/validators.md)

-   :material-file-document:{ .lg .middle } __Provenance Tracking__

    ---

    Save numpy arrays to HDF5 with git SHA, timestamps, seeds, and package versions automatically captured.

    [:octicons-arrow-right-24: Provenance](guide/provenance.md)

-   :material-test-tube:{ .lg .middle } __Testing Utilities__

    ---

    Physics-aware test decorators and fixtures for reproducible scientific testing.

    [:octicons-arrow-right-24: Testing](guide/testing.md)

-   :material-folder-plus:{ .lg .middle } __Project Templates__

    ---

    Pre-configured Python and C++ templates with linting, testing, and CI already set up.

    [:octicons-arrow-right-24: Templates](getting-started/templates.md)

</div>

## The 5 Principles

1. **Correctness First** — Physical accuracy is non-negotiable
2. **Specification Before Code** — Tests define the contract
3. **Fail Fast with Clarity** — Validate inputs, descriptive errors
4. **Simplicity by Design** — Functions ≤50 lines, single responsibility
5. **Infrastructure Enables Speed** — CI, tests, linting from day one

## Installation

```bash
# Basic install
pip install agentbible

# With HDF5 provenance support
pip install agentbible[hdf5]

# Full development install
pip install agentbible[all]
```

## License

MIT — Use and adapt freely.
