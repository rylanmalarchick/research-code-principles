# Contributing

Thank you for considering contributing to AgentBible!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/rylanmalarchick/research-code-principles
cd research-code-principles

# Use bootstrap script
./bootstrap.sh

# Or manual setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,hdf5]"
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=agentbible --cov-report=html

# Run specific test file
pytest tests/test_validators_quantum.py -v
```

## Code Quality

We use strict code quality checks:

```bash
# Linting
ruff check agentbible/ tests/

# Formatting
ruff format agentbible/ tests/

# Type checking
mypy agentbible/
```

Pre-commit hooks run these automatically on commit.

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Your Changes

- Follow the [style guide](philosophy/style-guide.md)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
pytest tests/ -v
ruff check agentbible/
mypy agentbible/
```

### 4. Commit

```bash
git add .
git commit -m "feat: add new validator for XYZ"
```

Use [conventional commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code change that neither fixes nor adds
- `test:` Adding tests
- `chore:` Maintenance

### 5. Push and Create PR

```bash
git push origin feature/my-feature
```

Open a pull request on GitHub.

## Adding a Validator

1. Add implementation to `agentbible/validators/`
2. Export from `agentbible/validators/__init__.py`
3. Add to `agentbible/__init__.py` exports
4. Write tests in `tests/test_validators_*.py`
5. Document in `docs/site/guide/validators.md`

Example:

```python
# agentbible/validators/quantum.py
def validate_trace_preserving(rtol: float = 1e-10, atol: float = 1e-12):
    """Validate that a quantum channel is trace-preserving."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Validation logic here
            return result
        return wrapper
    return decorator
```

## Documentation

Documentation is built with MkDocs:

```bash
# Install docs dependencies
pip install mkdocs-material mkdocstrings[python]

# Serve locally
mkdocs serve

# Build
mkdocs build
```

Documentation files are in `docs/site/`.

## Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit: `git commit -m "chore: release v0.1.2"`
4. Tag: `git tag v0.1.2`
5. Push: `git push origin main --tags`
6. CI automatically publishes to PyPI

## Questions?

- Open an issue for bugs or feature requests
- Email: [rylan1012@gmail.com](mailto:rylan1012@gmail.com)

## Code of Conduct

Be respectful and constructive. We're all here to build better research software.
