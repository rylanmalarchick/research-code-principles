# Python Research Starter Template

Production-grade Python project scaffold for research software.

## Quick Start

```bash
# Copy this template to your project
cp -r templates/python_research ~/my-project
cd ~/my-project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy src/
```

## What's Included

| File | Purpose |
|------|---------|
| `pyproject.toml` | Pre-configured with ruff, mypy (strict), pytest, coverage |
| `src/validation.py` | Physical validation helpers (unitarity, normalization, etc.) |
| `tests/conftest.py` | Reproducibility fixtures (fixed seeds), common test data |
| `tests/test_validation.py` | Example tests demonstrating specification-first approach |

## Configuration Highlights

### Linting (ruff)
- Strict rules: E, F, W, I, N, UP, B, C4, SIM, ARG, PTH
- Auto-fixes import sorting (isort) and code upgrades (pyupgrade)

### Type Checking (mypy)
- Strict mode enabled: `disallow_untyped_defs`, `no_implicit_optional`
- Catches type errors before runtime

### Testing (pytest)
- **70% coverage minimum** enforced (research standard)
- Markers: `@pytest.mark.slow`, `@pytest.mark.deterministic`, `@pytest.mark.requires_gpu`
- Fixed random seeds via `conftest.py` for reproducibility

## Physical Validation

The `validation.py` module provides:

```python
from src.validation import check_unitarity, check_normalized, check_density_matrix

# Validate quantum operations
check_unitarity(gate_matrix)        # U†U = I
check_normalized(state_vector)       # ⟨ψ|ψ⟩ = 1
check_density_matrix(rho)            # Hermitian, trace=1, positive semidefinite
check_hermitian(hamiltonian)         # H = H†
check_probabilities(measurement)     # non-negative, sum=1
```

## Next Steps

1. Update `pyproject.toml` with your project name and dependencies
2. Add your source code to `src/`
3. Write tests in `tests/` (specification first!)
4. Run `pytest --cov` to check coverage
5. Set up pre-commit hooks (see `../.pre-commit-config.yaml`)

## License

MIT
