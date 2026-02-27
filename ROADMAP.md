# AgentBible Roadmap

This document outlines the development roadmap for AgentBible.

## v0.6.0 (February 2026) - Evidence-Based Minimal Context

**Status:** Released (Current)

Motivated by arxiv:2602.11988 ("Evaluating AGENTS.md for Coding Agents"), which found that
LLM-generated context files reduce task success by 2-3% and increase cost by 20-23%, while
developer-written minimal files improve success by ~4%.

### Features
- [x] `bible audit context [file]` — Score AGENTS.md / .cursorrules for minimal context compliance
  - Detects codebase overviews, workflow checklists, and long code blocks
  - Tightness score (0-100); exits 0 if score ≥ 70 (CI-friendly gate)
  - JSON output mode for CI integration
  - Auto-detects AGENTS.md or .cursorrules in current directory
- [x] `bible generate-agents-md` — Generate minimal, evidence-based AGENTS.md
  - Flags: `--domain [quantum|ml|atmospheric|none]`, `--test-cmd`, `--coverage`, `--stdout`
  - Output ≤ 20 lines; generated file passes its own `bible audit context`
- [x] `bible audit` converted to group with `code` and `context` subcommands
- [x] `bible init` now also generates minimal AGENTS.md alongside .cursorrules
- [x] Slimmed `.cursorrules` templates (~13 lines, down from ~134)
- [x] `bible context --all` deprecated with migration path to `--query`
  (removal target: v0.7.0)

### Reference
arxiv:2602.11988 — "Evaluating AGENTS.md for Coding Agents"

---

## v0.2.1 - v0.5.0 (January 2026)

**Status:** Released

### v0.2.0 Features
- Physics validators (`@validate_unitary`, `@validate_density_matrix`, etc.)
- Provenance metadata capture (git, seeds, packages, hardware)
- Project templates (Python Scientific, C++ HPC/CUDA)
- Basic CLI (`bible init`, `bible validate`, `bible context`, `bible info`)
- MkDocs documentation site
- GitHub Actions CI/CD with trusted PyPI publishing
- Dependabot for automated dependency updates

### v0.2.0 Features (NEW)
- [x] Conditional validation levels (`debug`, `lite`, `off`) for performance tuning
- [x] Environment variable `AGENTBIBLE_VALIDATION_LEVEL` for global control
- [x] CUDA driver version capture in hardware metadata
- [x] `bible audit` command - Self-verify code against AgentBible principles
  - Rule of 50 (function line length)
  - Docstring presence check
  - Type hints verification
  - JSON output for CI integration

### v0.2.1 Features (NEW)
- [x] `bible ci status` - Show GitHub Actions workflow status
- [x] `bible ci verify` - Verify CI is passing (with --wait option)
- [x] `bible ci release` - Full automated release flow
- [x] Enhanced `.cursorrules` with CI/CD guidance for AI agents
- [x] `gh` CLI integration patterns and best practices

## v0.7.0 (Q2 2026) - Context Cleanup

**Status:** Planned

### Breaking Changes
- [ ] Remove `bible context --all` (deprecated in v0.6.0)
  - Migration: use `bible context --query "your topic"` for task-specific retrieval

---

## v0.3.0 (Q2 2026) - Ecosystem Integration

**Status:** Planned

### CLI Enhancements
- [ ] `bible report` command - Generate scientific health reports from HDF5 provenance

### AI Agent Integration
- [ ] `agent_registry.yaml` - Map file patterns to required validators
- [ ] Enhanced `.cursorrules` with registry integration

### C++ Template Modernization
- [ ] Modular CMake with target-based architecture
- [ ] CPM.cmake for dependency management
- [ ] `DeviceTensor` class template for GPU computation

### Performance
- [ ] pytest-benchmark integration
- [ ] Baseline benchmarks for all validators
- [ ] Performance regression tracking in CI

### Jupyter Integration
- [ ] `%agentbible` magic for notebook validation
- [ ] Automatic provenance capture for notebook outputs
- [ ] Interactive validation error display

### Enhanced Error Messages
- [ ] Physics references with links to relevant papers/textbooks
- [ ] Suggested fixes based on common error patterns

## v0.4.0 (Q3 2026) - Enterprise & Scale

**Status:** Planned

### Example Gallery
- [ ] 5+ real-world project examples
- [ ] Quantum optimization tutorial
- [ ] ML training reproducibility example
- [ ] HPC simulation workflow

### Cloud Provenance Backend (Optional)
- [ ] Hosted service for provenance storage
- [ ] Team sharing and collaboration features
- [ ] Experiment tracking dashboard

### Multi-Language Support
- [ ] Julia template and validators
- [ ] Rust template for performance-critical code
- [ ] Language-agnostic provenance format

### Advanced Validation
- [ ] Custom validator DSL
- [ ] Composite validators (e.g., "valid quantum circuit")
- [ ] Async validation for large datasets

## Future Ideas (Backlog)

These are ideas under consideration but not yet scheduled:

- **Interactive Validation Dashboard** - Web UI for exploring validation results
- **arXiv Integration** - Automatic paper submission metadata
- **Slurm/PBS Integration** - HPC job provenance capture
- **Differential Testing** - Compare outputs across code versions
- **GPU Memory Profiling** - Integrated CUDA memory tracking
- **Quantum Hardware Integration** - Validate against real quantum devices

## Contributing

Want to help build the future of AgentBible? See [CONTRIBUTING.md](CONTRIBUTING.md).

Have a feature request? [Open an issue](https://github.com/rylanmalarchick/research-code-principles/issues/new).

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.
