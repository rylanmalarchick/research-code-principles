# Research Code Principles

**Production-grade infrastructure for AI-assisted research software.**

Clone this repository. Copy a template. Start building research code that meets the same standards as production systems.

## Quick Start

### 1. Copy a Starter Template

**Python project:**
```bash
cp -r templates/python_research ~/my-project
cd ~/my-project
pip install -e ".[dev]"
pytest  # Run tests
```

**C++/CUDA project:**
```bash
cp -r templates/cpp_hpc ~/my-project
cd ~/my-project
cmake -B build && cmake --build build
ctest --test-dir build  # Run tests
```

### 2. Load Context for AI Sessions

Load modular prompts instead of giant context files:

```bash
# Just the essentials (~50 lines)
cat agent_prompts/core-principles.md

# Add physics validation for quantum work
cat agent_prompts/core-principles.md agent_prompts/physics-validation.md
```

### 3. Generate Repository Map

Understand any codebase structure without reading every file:

```bash
./scripts/map_repo.sh > repo_map.txt
```

## What's Included

```
research-code-principles/
├── templates/                  # Clonable project starters
│   ├── python_research/        # Pre-configured Python environment
│   └── cpp_hpc/                # Pre-configured C++/CUDA environment
├── agent_prompts/              # Modular AI context snippets
│   ├── core-principles.md      # 5 principles (~50 lines)
│   ├── test-generation.md      # Writing research tests
│   ├── physics-validation.md   # Quantum constraints
│   ├── kernel-optimization.md  # CUDA best practices
│   ├── code-review.md          # Quick checklist
│   └── error-handling.md       # Fail-fast patterns
├── .github/workflows/ci.yml    # CI/CD template
├── .pre-commit-config.yaml     # Pre-commit hooks
├── scripts/map_repo.sh         # Repository mapper
├── examples/                   # Proof-of-concepts
│   ├── quantum-gate-example/   # Quantum gate with tests
│   └── hpc-vqe-benchmark/      # 117x speedup demonstration
└── docs/                       # Deep-dive documentation
    ├── philosophy.md           # Why good code matters
    ├── agent-coding-context.md # Full AI context (if needed)
    └── style-guide-reference.md # Exhaustive style guide
```

## The 5 Principles

1. **Correctness First** — Physical accuracy is non-negotiable
2. **Specification Before Code** — Tests define the contract
3. **Fail Fast with Clarity** — Validate inputs, descriptive errors
4. **Simplicity by Design** — Functions ≤50 lines, single responsibility
5. **Infrastructure Enables Speed** — CI, tests, linting from day one

## Templates

### Python Research Template

Pre-configured with:
- **ruff** for linting (strict rules)
- **mypy** in strict mode (type checking)
- **pytest** with 70% coverage minimum
- **Physical validation** helpers (unitarity, normalization, density matrices)
- **Reproducibility fixtures** (fixed seeds)

```bash
cp -r templates/python_research ~/my-project
```

### C++/HPC Template

Pre-configured with:
- **CMake** with zero-warning policy (`-Wall -Wextra -Werror` in CI)
- **GoogleTest** for testing
- **CudaMemory<T>** RAII wrapper (zero manual cudaFree)
- **Physical validation** functions (unitarity, normalization)
- **Optional CUDA** support

```bash
cp -r templates/cpp_hpc ~/my-project
```

## Agent Prompts

Instead of loading 500+ lines of context, use modular prompts:

| Prompt | Lines | Use Case |
|--------|-------|----------|
| `core-principles.md` | ~50 | Every session |
| `test-generation.md` | ~60 | Writing tests |
| `physics-validation.md` | ~60 | Quantum/physics code |
| `kernel-optimization.md` | ~60 | CUDA kernels |
| `code-review.md` | ~40 | Reviewing code |
| `error-handling.md` | ~50 | Fail-fast patterns |

**Combine as needed:**
```bash
cat agent_prompts/core-principles.md agent_prompts/test-generation.md > session.md
```

## Automated Enforcement

### GitHub Actions CI

Copy `.github/workflows/ci.yml` to your project. It:
- Runs tests with coverage check (70% minimum)
- Runs linting (ruff for Python, compiler warnings for C++)
- Runs type checking (mypy)
- Fails the build on any warning

### Pre-commit Hooks

Copy `.pre-commit-config.yaml` and install:

```bash
pip install pre-commit
pre-commit install
```

Hooks include:
- Secret detection (prevent credential leaks)
- Code formatting (ruff, clang-format)
- Type checking (mypy)
- Commit message linting (conventional commits)

## Examples

### Quantum Gate Example

Full implementation with tests demonstrating the principles:
- `examples/quantum-gate-example/gate.py` — Gate implementation
- `examples/quantum-gate-example/test_gate.py` — Comprehensive tests
- `examples/quantum-gate-example/prompting-log.md` — How it was built with AI

### HPC VQE Benchmark

Demonstrates the **117x speedup** achieved through systematic optimization:
- JAX JIT compilation (21.7x)
- GPU backend (2.8x additional)
- MPI parallelization (1.9x additional)

See `examples/hpc-vqe-benchmark/`

## Proof Points

These standards are applied across production projects:

| Project | Language | Tests | Key Practices |
|---------|----------|-------|---------------|
| **QubitPulseOpt** | Python | 800+ | CI/CD, 74% coverage, type hints |
| **cuda-quantum-simulator** | CUDA/C++ | 9 suites | RAII, zero manual cudaFree |
| **quantum-circuit-optimizer** | C++17 | 340+ | OpenQASM parser, DAG optimization |
| **QuantumVQE** | Python | - | 117x speedup, deterministic seeds |

## Documentation

For deep dives:
- `docs/philosophy.md` — Why good code matters (theory)
- `docs/agent-coding-context.md` — Full AI context (500+ lines)
- `docs/style-guide-reference.md` — Exhaustive style conventions

## License

MIT — Use and adapt freely.

## Author

Rylan Malarchick — [rylan1012@gmail.com](mailto:rylan1012@gmail.com)

---

**Latest:** v2.0 (Dec 2025) — Restructured as clonable infrastructure
