# Architecture: Target Directory Structure

**Purpose:** Define the complete directory structure for the research-code-principles repository after all sprints are complete.

---

## Current vs Target

### Legend
- ✅ Exists and complete
- 🔄 Exists, needs updates
- ⬜ To be created

---

## Target Structure (v3.0)

```
research-code-principles/
│
├── .cursorrules                    ✅ AI agent enforcement rules
├── .devcontainer/                  ✅ VS Code dev container
│   ├── devcontainer.json
│   ├── Dockerfile
│   └── post-create.sh
│
├── .github/                        ✅ GitHub config
│   └── workflows/
│       ├── ci.yml                  ✅ Main CI pipeline
│       ├── publish.yml             ⬜ PyPI trusted publishing
│       └── security.yml            ⬜ pip-audit vulnerability scan
│
├── .gitignore                      ✅
├── .pre-commit-config.yaml         ✅ Pre-commit hooks
├── bootstrap.sh                    ✅ One-command setup script
├── CHANGELOG.md                    ⬜ Release history
├── CONTRIBUTING.md                 ✅
├── LICENSE                         ✅ MIT
├── README.md                       🔄 Update for v3.0
├── SECURITY.md                     ⬜ Security policy
│
├── agent_docs/                     🔄 Meta-docs for repo development
│   ├── SCOPE_OF_WORK.md            ✅ Overall vision
│   ├── sprint-1-foundation.md      ✅ Sprint 1 details
│   ├── sprint-2-package.md         ✅ Sprint 2 details
│   ├── ARCHITECTURE.md             ✅ This file
│   └── README.md                   ⬜ How to use agent_docs
│
├── agent_prompts/                  ✅ Modular AI prompts
│   ├── README.md                   ✅
│   ├── core-principles.md          ✅
│   ├── test-generation.md          ✅
│   ├── physics-validation.md       ✅
│   ├── kernel-optimization.md      ✅
│   ├── code-review.md              ✅
│   └── error-handling.md           ✅
│
├── agentbible/                     ✅ Python package (pip installable)
│   ├── __init__.py
│   ├── __main__.py                 # python -m agentbible
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py                 # Click entry point (skeleton)
│   ├── validators/
│   │   ├── __init__.py
│   │   ├── base.py                 # ValidationError, utilities
│   │   ├── quantum.py              # @validate_unitary, @validate_hermitian
│   │   ├── probability.py          # @validate_probability, @validate_normalized
│   │   └── bounds.py               # @validate_positive, @validate_range
│   │   └── bounds.py               # @validate_positive, @validate_range
│   ├── provenance/
│   │   ├── __init__.py
│   │   └── hdf5.py                 # save_with_metadata(), git SHA embedding
│   ├── testing/
│   │   ├── __init__.py
│   │   ├── decorators.py           # @physics_test
│   │   └── fixtures.py             # deterministic_seed, tolerance
│   └── templates/                  # Embedded templates for bible init
│       ├── python_research/        # Copy of templates/python_research
│       └── cpp_hpc/                # Copy of templates/cpp_hpc
│
├── docs/                           ✅ Deep-dive documentation
│   ├── philosophy.md               ✅ Research code principles (theory)
│   ├── agent-coding-context.md     ✅ Full AI context (~500 lines)
│   ├── style-guide-reference.md    ✅ Exhaustive style guide
│   ├── repo-standards.md           ✅ Git/CI workflow standards
│   └── prompting-research-code.md  ✅ Prompt engineering patterns
│
├── examples/                       🔄 Working examples
│   ├── quantum-gate-example/       ✅ Python quantum gate with tests
│   └── hpc-vqe-benchmark/          ✅ 117x speedup demonstration
│
├── opencode-context/               ✅ Vector-based context retrieval
│   ├── README.md                   ✅
│   ├── requirements.txt            ✅
│   ├── config.example.yaml         ✅
│   ├── bin/
│   │   ├── oc-context              ✅
│   │   └── oc-update               ✅
│   └── oc_lib/
│       ├── __init__.py             ✅
│       ├── config.py               ✅
│       ├── embed.py                ✅
│       └── retrieve.py             ✅
│
├── scripts/                        ✅ Utility scripts
│   └── map_repo.sh                 ✅ Repository structure mapper
│
├── templates/                      ✅ Clonable project starters
│   ├── python_research/            ✅ Pre-configured Python
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   ├── tests/
│   │   └── .cursorrules            ⬜ Add template-specific rules
│   └── cpp_hpc/                    ✅ Pre-configured C++/CUDA
│       ├── CMakeLists.txt
│       ├── include/
│       ├── src/
│       ├── tests/
│       └── .cursorrules            ⬜ Add template-specific rules
│
├── tests/                          ✅ Package tests
│   ├── conftest.py                 ✅ Shared fixtures
│   ├── test_validators_quantum.py  ✅ Quantum validator tests
│   ├── test_validators_probability.py ✅ Probability validator tests
│   ├── test_validators_bounds.py   ✅ Bounds validator tests
│   ├── test_cli.py                 ✅ CLI smoke tests
│   └── test_provenance.py          ⬜ Test HDF5 metadata (Sprint 4)

└── pyproject.toml                  ✅ Package definition
```

---

## Key Directories Explained

### agentbible/

The pip-installable Python package. Contains:

- **cli/** - Command-line interface (`bible init`, `bible context`, `bible validate`)
- **validators/** - Decorators for physics validation (`@validate_unitary`)
- **provenance/** - HDF5 metadata embedding for reproducibility
- **testing/** - pytest fixtures and decorators for physics tests
- **templates/** - Embedded copies of project templates for `bible init`

### agent_docs/

Meta-documentation for developing the repository itself. These are embedded in the vector DB and loaded with `oc-context --all ./agent_docs` when working on the repo.

### agent_prompts/

Modular prompt snippets for AI sessions. Users concatenate what they need:
```bash
cat agent_prompts/core-principles.md agent_prompts/physics-validation.md
```

### opencode-context/

The vector-based context retrieval tool. Already complete. Will be wrapped by `bible context` in the CLI.

### templates/

Clonable project starters. Users copy these to start new projects:
```bash
cp -r templates/python_research ~/my-project
# or with CLI:
bible init my-project --template python-scientific
```

---

## File Responsibilities

### Root Level

| File | Purpose |
|------|---------|
| `.cursorrules` | AI agent enforcement (loaded by Cursor/Claude Code) |
| `bootstrap.sh` | One-command dev environment setup |
| `pyproject.toml` | Package metadata and build config |
| `SECURITY.md` | Security policy and vulnerability disclosure |
| `CHANGELOG.md` | Version history and release notes |

### GitHub Workflows

| File | Purpose |
|------|---------|
| `ci.yml` | Run tests, linting, type checking on PR |
| `publish.yml` | Publish to PyPI on release (trusted publishing) |
| `security.yml` | Run pip-audit for dependency vulnerabilities |

---

## Package Entry Points

After `pip install agentbible`:

```bash
# CLI commands
bible init my-project --template python-scientific
bible context --all ./agent_docs
bible validate state.npy --check unitarity

# Python imports
from agentbible.validators import validate_unitary, validate_hermitian
from agentbible.provenance import save_with_metadata
from agentbible.testing import physics_test, deterministic_seed
```

---

## Configuration Files

### pyproject.toml (Target)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "agentbible"
version = "0.1.0"
description = "Production-grade infrastructure for AI-assisted research software"
readme = "README.md"
license = "MIT"
authors = [
    { name = "Rylan Malarchick", email = "rylan1012@gmail.com" }
]
requires-python = ">=3.9"
dependencies = [
    "click>=8.0",
    "rich>=13.0",
    "pyyaml>=6.0",
    "numpy>=1.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
    "pre-commit>=3.0",
]
hdf5 = [
    "h5py>=3.0",
]
vector = [
    "chromadb>=0.4",
    "openai>=1.0",
    "tiktoken>=0.5",
]

[project.scripts]
bible = "agentbible.cli.main:cli"

[tool.hatch.build.targets.wheel]
packages = ["agentbible"]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]

[tool.mypy]
strict = true
```

---

## Migration Notes

### What Stays the Same
- All existing documentation content preserved
- Templates structure unchanged
- Examples unchanged
- agent_prompts unchanged

### What Changes
- Add `agentbible/` package
- Add root-level infrastructure files
- Add `.devcontainer/`
- Update README for v3.0
- Add pyproject.toml for package

### What Gets Wrapped
- `opencode-context/` functionality wrapped by `bible context`
- Templates embedded in package for `bible init`
