# Changelog

All notable changes to AgentBible will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-01-02

### Added

#### CI/CD Automation
- `bible ci status` command to show GitHub Actions workflow status
- `bible ci verify` command to verify CI is passing
  - `--wait` flag to wait for in-progress runs to complete
  - Returns proper exit codes for scripting (0 = pass, 1 = fail)
- `bible ci release` command for full automated release flow
  - Bumps version in `pyproject.toml` and `__init__.py`
  - Creates git tag and pushes to remote
  - Waits for CI to pass
  - Creates GitHub release with auto-generated notes
  - `--draft` flag for draft releases
  - `--no-verify` to skip CI verification

#### Agent CI/CD Guidance
- Enhanced `.cursorrules` template with comprehensive CI/CD rules
- `gh` CLI patterns for common operations
- Documentation of what CAN vs CANNOT be automated
- Common CI failure troubleshooting guide

### Changed

- Updated ROADMAP.md to reflect v0.2.0 and v0.2.1 completion

## [0.2.0] - 2026-01-01

### Added

#### Validation Levels
- `ValidationLevel` enum with three levels: `debug`, `lite`, `off`
- `AGENTBIBLE_VALIDATION_LEVEL` environment variable for global control
- All validators now support `level` parameter:
  - `debug` (default): Full physics validation
  - `lite`: Only NaN/Inf checks (fast, catches numerical instability)
  - `off`: Skip all validation (with warning) - for benchmarking only
- `get_validation_level()` helper function

#### Code Audit CLI
- `bible audit` command for code quality checks
  - Rule of 50: Detects functions exceeding configurable line limit
  - Docstring presence: Checks public functions/classes for docstrings
  - Type hints: Validates function signatures have type annotations
- JSON output for CI integration (`--json` flag)
- Strict mode (`--strict`) promotes warnings to errors
- Exclude patterns support (`-e`, `--exclude`)
- Custom line limits (`--max-lines`)

#### Documentation
- `ROADMAP.md` with v0.2.0, v0.3.0, v0.4.0 planned features
- Improved README with problem-first framing, before/after examples

#### Provenance
- CUDA driver version capture in hardware metadata

### Changed

- README rewritten with "Why AgentBible Exists" problem-first framing
- Added before/after code example showing bug detection
- Added "Who This Is For" section (researchers, quantum/ML developers, PhD students)

### Fixed

- Test assertion for probability error message (unicode vs ASCII)

## [0.1.1] - 2026-01-01

### Added

#### Documentation Site
- MkDocs documentation site with Material theme
- Getting Started guides (installation, quickstart, templates)
- User guides for validators, provenance, testing, and CLI
- API reference documentation
- Philosophy and style guide pages
- GitHub Pages deployment workflow (`.github/workflows/docs.yml`)

#### CI/CD Enhancements
- Dependabot configuration for automated dependency updates (`.github/dependabot.yml`)
- Sprint 5 CI/CD documentation (`agent_docs/sprint-5-cicd.md`)

#### Templates
- C++ HPC/CUDA template (`cpp-hpc-cuda`) with CMake, GoogleTest support
- Template test coverage

### Changed

- README badges now include PyPI version, codecov coverage, and docs link
- Documentation section in README links to hosted docs site
- Updated SCOPE_OF_WORK.md to reflect Sprint 1-6 completion

### Developer Notes

This release completes the initial 6-sprint development cycle:
1. Foundation (bootstrap, .cursorrules, .devcontainer) 
2. Package Core (validators, CLI skeleton)
3. CLI Scaffolding (bible init, templates)
4. Provenance & Testing (HDF5 metadata, @physics_test)
5. CI/CD & Security (GitHub Actions, pip-audit, trusted publishing)
6. Documentation & Polish (MkDocs site, badges)

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

[Unreleased]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rylanmalarchick/research-code-principles/releases/tag/v0.1.0
