# Changelog

All notable changes to AgentBible will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-01

### Added

#### Package Core
- `agentbible` pip-installable package with `bible` CLI command
- Physics validation decorators:
  - `@validate_unitary` - Validates U @ U.H = I
  - `@validate_hermitian` - Validates A = A.H
  - `@validate_density_matrix` - Validates trace=1, positive semi-definite, Hermitian
  - `@validate_probability` - Validates value in [0, 1]
  - `@validate_probabilities` - Validates array of probabilities
  - `@validate_normalized` - Validates sum/norm = 1
  - `@validate_positive`, `@validate_non_negative`, `@validate_range`, `@validate_finite`

#### CLI Commands
- `bible init <name>` - Create new project from template
  - `--template python-scientific` (default) or `cpp-hpc-cuda`
  - `--author`, `--email`, `--description` options
  - `--no-git`, `--no-venv`, `--force` flags
- `bible validate <file>` - Validate physics constraints in data files
  - Supports `.npy`, `.npz`, `.h5`/`.hdf5` files
  - `--check unitarity/hermiticity/trace/positivity/normalization/all`
- `bible context` - Generate AI context (wraps opencode-context)
- `bible info` - Show installation information

#### Provenance Tracking
- `save_with_metadata()` - Save numpy arrays to HDF5 with full provenance
  - Git SHA, branch, dirty status
  - Timestamps (UTC ISO 8601)
  - Random seeds (numpy, python, torch)
  - System info (hostname, platform, Python version)
  - Package versions
- `load_with_metadata()` - Load data with provenance metadata
- `get_provenance_metadata()` - Generate metadata dictionary

#### Testing Utilities
- `@physics_test(checks=[...])` - Decorator for physics-aware tests
- `deterministic_seed` - Pytest fixture for reproducible tests
- `tolerance` / `quantum_tolerance` - Pytest fixtures for comparisons

#### Templates
- `python-scientific` template embedded in package
  - Pre-configured pyproject.toml with ruff, mypy, pytest
  - Physics validation helpers in `src/validation.py`
  - 28 example tests
  - `.cursorrules` for AI agents

#### Infrastructure
- `bootstrap.sh` - One-command development environment setup
- `.cursorrules` - AI agent coding rules
- `.devcontainer/` - VS Code dev container configuration
- GitHub Actions CI/CD with:
  - Multi-Python testing (3.9-3.12)
  - Linting (ruff) and type checking (mypy)
  - Security scanning (pip-audit)
  - PyPI trusted publishing

#### Documentation
- `SECURITY.md` - Security policy and vulnerability reporting
- `CHANGELOG.md` - This file
- Sprint planning documents in `agent_docs/`

### Developer Notes

This is the initial release combining 6 development sprints:
1. Foundation (bootstrap, .cursorrules, .devcontainer)
2. Package Core (validators, CLI skeleton)
3. CLI Scaffolding (bible init, templates)
4. Provenance & Testing (HDF5 metadata, @physics_test)
5. CI/CD & Security (GitHub Actions, pip-audit, trusted publishing)
6. Documentation & Polish (README, CHANGELOG, badges)

[Unreleased]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rylanmalarchick/research-code-principles/releases/tag/v0.1.0
