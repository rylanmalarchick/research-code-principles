# Changelog

All notable changes to AgentBible will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-01-19

### Added

#### New CLI Commands
- `bible scaffold` command for generating module stubs with docstrings and type hints
  - `--class` option to generate a class with given name
  - `--dataclass` option to generate a dataclass
  - `--methods` and `--fields` options for customization
  - `--functions` option to generate standalone functions
  - Automatic test file generation (skip with `--no-test`)
- `bible retrofit` command to add AgentBible structure to existing projects
  - Detects project type (Python, C++) automatically
  - `--cursorrules`, `--precommit`, `--validators`, `--conftest`, `--agent-docs` options
  - Interactive mode for guided setup
  - `--all` flag to add all components at once
- `bible check-coverage` command for quick pytest-cov coverage checks
  - `--threshold` and `--fail-under` options
  - `--html` flag for HTML report generation
- `bible report` command for HDF5 provenance report generation
  - `--format text|markdown|json` output formats
  - `--output` option to write to file
  - Extracts and formats provenance metadata from HDF5 files
- `bible registry` command group for AI agent integration
  - `bible registry show` - Display current registry configuration
  - `bible registry init` - Create default `agent_registry.yaml`
  - `bible registry check <path>` - Check files use required validators

#### New Research Templates
- `python-quantum` template for quantum computing projects
  - Pre-configured unitarity, hermiticity validators
  - Example gates.py and circuits.py modules
  - Quantum-specific test fixtures
- `python-ml` template for machine learning projects
  - Data leakage detection helpers
  - Reproducibility utilities (seed management, model checkpointing)
  - ML-specific validation patterns
- `python-simulation` template for numerical simulation projects
  - HDF5 I/O helpers with provenance tracking
  - Conservation law validators
  - Simulation-specific test fixtures

#### Jupyter Integration
- `agentbible.jupyter` module for notebook support
  - `%load_ext agentbible.jupyter` to load the extension
  - `%agentbible` line magic for quick validation
  - `%%validate` cell magic for validating cell outputs
  - Graceful fallback when IPython not available

#### Pre-commit Integration
- `.pre-commit-hooks.yaml` in repo root
- AgentBible can now be used as a pre-commit hook

#### Enhanced Error Messages
- `PhysicsConstraintError` now includes `reference` and `guidance` fields
- All physics errors include academic references and fix suggestions
- `to_dict()` method for serializing errors to JSON

### Changed

- Version bumped to 0.5.0
- Updated `bible info` to display `bible registry` command
- Updated `bible info` to display `bible report` command
- Updated templates `__init__.py` with all new template registrations

## [0.4.0] - 2026-01-06

### Added

#### Array Validators (Direct Check Functions)
- New `check_*` functions for direct array validation in pipelines:
  - `check_finite(arr, name, strict)` - Validate no NaN/Inf
  - `check_positive(arr, name, atol, strict)` - Validate all values > 0
  - `check_non_negative(arr, name, atol, strict)` - Validate all values >= 0
  - `check_range(arr, min_val, max_val, name, inclusive, atol, strict)` - Validate range
  - `check_probability(value, name, atol, strict)` - Validate scalar in [0, 1]
  - `check_probabilities(arr, name, atol, strict)` - Validate array in [0, 1]
  - `check_normalized(arr, name, axis, rtol, atol, strict)` - Validate sums to 1
- All check functions support `strict=False` to warn instead of raise

#### Validation Pipeline
- New `ValidationPipeline` class for composing multiple validators
- Pre-built pipelines: `create_numeric_pipeline`, `create_positive_pipeline`,
  `create_probability_pipeline`, `create_distribution_pipeline`
- `ValidationResult` dataclass for collecting all validation outcomes
- Context manager `ValidationPipeline.strict_mode()` for temporary override

#### ML Domain (`agentbible.domains.ml`)
- New domain for machine learning validation:
  - `check_no_leakage(feature_names, forbidden, strict)` - Detect data leakage
  - `check_temporal_autocorrelation(arr, max_autocorr, strict)` - Detect high autocorrelation
  - `check_coverage(y_true, y_lower, y_upper, target, tolerance, strict)` - Validate prediction intervals
  - `check_exchangeability(residuals, max_autocorr, max_drift, strict)` - Check conformal assumptions
- Decorators:
  - `@validate_no_leakage(forbidden, feature_names_arg)` - Prevent forbidden features
  - `@validate_cv_strategy(max_autocorr, target_arg, warn_only)` - Warn on random CV issues
- Errors: `DataLeakageError`, `CoverageError`
- Warnings: `ExchangeabilityWarning`, `AutocorrelationWarning`, `CVStrategyWarning`

#### Atmospheric Domain (`agentbible.domains.atmospheric`)
- New domain for atmospheric science validation:
  - `check_cloud_base_height(arr, min_height, max_height, strict)` - Validate CBH
  - `check_boundary_layer_height(arr, min_height, max_height, strict)` - Validate BLH
  - `check_lifting_condensation_level(arr, min_height, max_height, strict)` - Validate LCL
  - `check_cloud_layer_consistency(cloud_base, cloud_top, strict)` - Validate base < top
  - `check_relative_humidity(arr, allow_supersaturation, strict)` - Validate RH in [0, 100]
  - `check_temperature_inversion(base_height, top_height, base_temp, top_temp, strict)` - Validate inversions
- Errors: `CloudBaseHeightError`, `BoundaryLayerHeightError`, `LiftingCondensationLevelError`,
  `CloudLayerConsistencyError`, `TemperatureInversionError`
- Warnings: `RelativeHumidityWarning`, `AtmosphericStabilityWarning`

### Changed

- Version bumped to 0.4.0
- Updated domains `__init__.py` to document ML and atmospheric domains

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

[Unreleased]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.2.1...v0.4.0
[0.2.1]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/rylanmalarchick/research-code-principles/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rylanmalarchick/research-code-principles/releases/tag/v0.1.0
