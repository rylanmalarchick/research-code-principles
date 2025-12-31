# Research Code Principles: Scope of Work

**Project:** research-code-principles → agentbible  
**Author:** Rylan Malarchick  
**Created:** December 2025  
**Status:** Sprint 1 - Foundation

---

## Executive Summary

Transform `research-code-principles` from a documentation repository into a **production-grade developer toolkit** with:

1. **`agentbible`** - Pip-installable Python package with validators, decorators, and CLI
2. **`bible` CLI** - Project scaffolding and context management
3. **AI-native architecture** - `.cursorrules`, embedded prompts, vector-based retrieval
4. **Invisible infrastructure** - One-command setup that "just works"

The repository name stays `research-code-principles` on GitHub; the package/brand is `agentbible`.

---

## Target Users

| User Type | What They Need |
|-----------|----------------|
| **AI agent users** (Claude Code, Cursor, etc.) | `.cursorrules`, modular prompts, context management |
| **PhD students / Postdocs** | Templates, validators, "how do I test physics code?" |
| **Research labs** | Enforceable standards, CI/CD patterns, reproducibility |
| **Rylan** | Quantum/CUDA stack template, personal workflow optimization |

---

## Current State (v2.0)

Already complete:
- ✅ `docs/` - Philosophy, style guide, agent context, prompting guide
- ✅ `templates/python_research/` - Pre-configured Python starter
- ✅ `templates/cpp_hpc/` - Pre-configured C++/CUDA starter
- ✅ `agent_prompts/` - Modular prompt snippets (6 files)
- ✅ `scripts/map_repo.sh` - Repository structure mapper
- ✅ `examples/quantum-gate-example/` - Working example
- ✅ `.pre-commit-config.yaml` - Pre-commit hooks
- ✅ `.github/workflows/` - CI/CD templates
- ✅ `opencode-context/` - Vector-based context retrieval tool

---

## Target State (v3.0)

### New Components

| Component | Description |
|-----------|-------------|
| `agentbible/` | Pip-installable Python package |
| `bible` CLI | `bible init`, `bible context`, `bible validate` |
| `.cursorrules` | Aggressive AI agent instructions |
| `bootstrap.sh` | One-command dev environment setup |
| `.devcontainer/` | VS Code dev container (Python + optional CUDA) |
| `SECURITY.md` | Security policy and disclosure process |

### Package Structure

```
agentbible/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── main.py              # Click CLI entry point
│   ├── init.py              # bible init command
│   ├── context.py           # bible context command (wraps oc-context)
│   └── validate.py          # bible validate command
├── validators/
│   ├── __init__.py
│   ├── quantum.py           # @validate_unitary, @validate_hermitian
│   ├── probability.py       # @validate_probability, @validate_normalized
│   └── bounds.py            # @validate_positive, @validate_range
├── provenance/
│   ├── __init__.py
│   └── hdf5.py              # save_with_metadata(), load_with_metadata()
├── testing/
│   ├── __init__.py
│   ├── decorators.py        # @physics_test
│   └── fixtures.py          # deterministic_seed, tolerance fixtures
└── templates/               # Embedded copies for bible init
    ├── python_research/
    └── cpp_hpc/
```

### CLI Commands

```bash
# Initialize new project from template
bible init my-project --template python-scientific
bible init my-project --template cpp-hpc-cuda

# Generate context for AI session
bible context                     # Auto-detect project, load defaults
bible context --all ./agent_docs  # Load entire directory
bible context --query "error handling"

# Validate physics constraints
bible validate state.npy --check unitarity
bible validate results.h5 --check all
```

---

## Sprint Overview

| Sprint | Focus | Duration | Status |
|--------|-------|----------|--------|
| **1** | Foundation (bootstrap, .cursorrules, .devcontainer) | 2-3 sessions | **Active** |
| **2** | Package Core (agentbible/, validators, CLI skeleton) | 3-4 sessions | Planned |
| **3** | CLI Scaffolding (bible init with templates) | 2-3 sessions | Planned |
| **4** | Provenance & Testing (HDF5, pytest-rcp) | 2-3 sessions | Planned |
| **5** | CI/CD & Security (pip-audit, trusted publishing) | 1-2 sessions | Planned |
| **6** | Documentation & Polish (docs site, badges) | 1-2 sessions | Planned |

---

## Sprint 1: Foundation

**Goal:** Make the repo immediately usable with aggressive AI enforcement

### Deliverables

1. **`bootstrap.sh`**
   - OS detection (Linux, macOS, WSL)
   - Python venv creation
   - Dependency installation
   - Pre-commit hook setup
   - Vector DB initialization (oc-update)
   - `--minimal` flag for docs-only setup

2. **`.cursorrules`**
   - Aggressive enforcement (REFUSE patterns)
   - Rule of 50 (functions ≤50 lines)
   - Mandatory test specification
   - Physical validation requirements
   - Citation requirements for equations

3. **`.devcontainer/`**
   - Base Python development container
   - CUDA variant (optional)
   - Pre-installed tools (ruff, mypy, pytest)
   - OpenCode context manager ready

4. **`agent_docs/`** (this directory)
   - `SCOPE_OF_WORK.md` (this file)
   - `sprint-1-foundation.md`
   - `ARCHITECTURE.md` - Target directory structure

5. **Config Updates**
   - Add `agent_docs/` to opencode config
   - Embed new docs in vector DB

### Success Criteria

- [ ] `./bootstrap.sh` runs successfully on fresh clone
- [ ] `.cursorrules` loaded by Cursor/Claude Code
- [ ] `.devcontainer/` opens in VS Code
- [ ] `oc-context --all ./agent_docs` returns sprint docs

---

## Sprint 2: Package Core

**Goal:** Pip-installable `agentbible` with validators

### Deliverables

1. **Package structure** - `agentbible/` with proper `pyproject.toml`
2. **Validators** - `@validate_unitary`, `@validate_trace`, `@validate_bounds`
3. **CLI skeleton** - `bible --help` works
4. **Tests** - 100% coverage on validators

### Success Criteria

- [ ] `pip install -e .` works
- [ ] `bible --help` shows available commands
- [ ] Validators catch invalid quantum states
- [ ] Tests pass with 100% coverage on core

---

## Sprint 3: CLI Scaffolding

**Goal:** `bible init` creates production-ready projects

### Deliverables

1. **`bible init`** - Copy templates with customization
2. **Template embedding** - Templates bundled in package
3. **Post-init hooks** - Auto-run git init, venv creation

### Success Criteria

- [ ] `bible init my-project --template python-scientific` creates working project
- [ ] Generated project passes `pytest` immediately
- [ ] `.cursorrules` included in generated project

---

## Sprint 4: Provenance & Testing

**Goal:** Research-grade data tracking and physics testing

### Deliverables

1. **HDF5 provenance** - `save_with_metadata()` embeds git SHA, seeds, hardware
2. **pytest-rcp plugin** - Physics assertions, multi-check support
3. **Fixtures** - `deterministic_seed`, `tolerance`

### Success Criteria

- [ ] HDF5 files contain full provenance metadata
- [ ] `@physics_test(checks=['trace', 'hermiticity'])` works
- [ ] Example in `examples/` demonstrates usage

---

## Sprint 5: CI/CD & Security

**Goal:** Production-ready package publishing

### Deliverables

1. **Trusted publishing** - GitHub Actions → PyPI without stored credentials
2. **`pip-audit`** - Dependency vulnerability scanning in CI
3. **`SECURITY.md`** - Disclosure policy
4. **Version automation** - Semantic versioning from git tags

### Success Criteria

- [ ] Package publishes to PyPI via GitHub release
- [ ] CI fails on known vulnerabilities
- [ ] 2FA enabled on PyPI

---

## Sprint 6: Documentation & Polish

**Goal:** Professional presentation

### Deliverables

1. **MkDocs site** - Hosted on GitHub Pages
2. **Badges** - Build status, coverage, PyPI version
3. **CHANGELOG.md** - Release history
4. **Video/GIF** - Quick demo of workflow

### Success Criteria

- [ ] Docs site live at research-code-principles.github.io or similar
- [ ] README has badges
- [ ] Demo shows `bible init` → `bible context` → AI session

---

## Non-Goals (Deferred)

- Hardware Abstraction Layer (HAL) - Too ambitious for now
- Performance dashboard - Nice to have, not core
- Conan/CPM alternatives - vcpkg is sufficient
- GUI - CLI only

---

## Security Considerations

| Risk | Mitigation |
|------|------------|
| PyPI credential compromise | Trusted publishing via GitHub Actions |
| Dependency vulnerabilities | `pip-audit` in CI, Dependabot alerts |
| API key exposure | Never log keys; document secrets handling |
| Arbitrary code execution | Templates are inert; no post-install scripts |

---

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package name | `agentbible` | Memorable, available on PyPI |
| CLI framework | Click | Standard, good UX |
| Embedding model | OpenAI text-embedding-3-small | Best quality; local fallback available |
| Vector DB | ChromaDB | Local, no server, SQLite-based |
| Build system | Hatch/pyproject.toml | Modern Python packaging |
| C++ package manager | vcpkg | Best CUDA support |

---

## References

- OpenCode Context Manager: `opencode-context/README.md`
- Research Code Principles: `docs/philosophy.md`
- Style Guide: `docs/style-guide-reference.md`
- Agent Prompts: `agent_prompts/`
