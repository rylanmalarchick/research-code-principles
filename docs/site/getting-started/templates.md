# Project Templates

AgentBible includes pre-configured project templates to get you started with best practices.

## Available Templates

### Python Scientific (`python-scientific`)

A complete Python project for scientific computing and research.

```bash
bible init my-project --template python-scientific
```

**Includes:**

| Component | Configuration |
|-----------|--------------|
| Package manager | `pyproject.toml` with hatch |
| Linting | ruff (strict rules, Google docstrings) |
| Type checking | mypy (strict mode) |
| Testing | pytest with 70% coverage minimum |
| Pre-commit | ruff, mypy, trailing whitespace |
| AI context | `.cursorrules` with physics rules |

**Project Structure:**

```
my-project/
├── pyproject.toml          # Package configuration
├── src/
│   ├── __init__.py
│   ├── core.py             # Core module
│   └── validation.py       # Physics validation helpers
├── tests/
│   ├── conftest.py         # Pytest fixtures
│   └── test_core.py        # Example tests (28 tests)
├── .cursorrules            # AI agent instructions
├── .pre-commit-config.yaml # Pre-commit hooks
├── .gitignore
└── README.md
```

### C++ HPC/CUDA (`cpp-hpc-cuda`)

A C++ project template for high-performance computing with optional CUDA support.

```bash
bible init my-project --template cpp-hpc-cuda
```

**Includes:**

| Component | Configuration |
|-----------|--------------|
| Build system | CMake 3.18+ |
| Testing | GoogleTest |
| Compiler | C++17, zero-warning policy |
| CUDA | Optional support |
| AI context | `.cursorrules` with HPC rules |

**Project Structure:**

```
my-project/
├── CMakeLists.txt          # Build configuration
├── include/
│   └── project_name/
│       └── core.hpp        # Header files
├── src/
│   ├── core.cpp            # Implementation
│   └── main.cpp            # Entry point
├── tests/
│   └── test_main.cpp       # GoogleTest tests
├── .cursorrules            # AI agent instructions
├── .clang-format           # Code formatting
└── README.md
```

## Template Options

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--author` | Author name | Git config or "Your Name" |
| `--email` | Author email | Git config or "you@example.com" |
| `--description` | Project description | "A scientific computing project" |

### Behavior Flags

| Flag | Description |
|------|-------------|
| `--no-git` | Skip git repository initialization |
| `--no-venv` | Skip Python virtual environment creation |
| `--force` | Overwrite existing directory |

## Examples

### Basic Python Project

```bash
bible init quantum-sim --template python-scientific
```

### Customized Python Project

```bash
bible init quantum-sim \
    --template python-scientific \
    --author "Jane Doe" \
    --email "jane@lab.edu" \
    --description "Quantum gate simulator"
```

### C++ Project Without Git

```bash
bible init hpc-solver \
    --template cpp-hpc-cuda \
    --no-git
```

### Force Overwrite Existing

```bash
bible init my-project --template python-scientific --force
```

## Post-Initialization

After running `bible init`, the template:

1. Creates the project directory
2. Copies and customizes all template files
3. Initializes a git repository (unless `--no-git`)
4. Creates a Python virtual environment (unless `--no-venv`)

### Python Projects

```bash
cd my-project
source .venv/bin/activate
pip install -e ".[dev]"
pytest  # All tests pass
```

### C++ Projects

```bash
cd my-project
mkdir build && cd build
cmake ..
make
ctest  # Run tests
```

## Customizing Templates

The templates are embedded in the AgentBible package. To customize:

1. Create your own template directory
2. Add a `pyproject.toml.template` or `CMakeLists.txt.template`
3. Use `{{project_name}}`, `{{author}}`, etc. for substitution

Template variables:

| Variable | Description |
|----------|-------------|
| `{{project_name}}` | Project name |
| `{{project_name_underscore}}` | Project name with underscores |
| `{{author}}` | Author name |
| `{{email}}` | Author email |
| `{{description}}` | Project description |
| `{{year}}` | Current year |
