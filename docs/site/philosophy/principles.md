# The 5 Principles

AgentBible is built on five core principles for research software development.

## 1. Correctness First

**Physical accuracy is non-negotiable.**

In research code, a wrong answer is worse than no answer. Every function that produces scientific output must be validated.

```python
# Bad: No validation
def compute_density_matrix(psi):
    return np.outer(psi, psi.conj())

# Good: Validated output
@validate_density_matrix
def compute_density_matrix(psi):
    return np.outer(psi, psi.conj())
```

Key practices:

- Use physics validators on all quantum/physics functions
- Write tests that verify physical properties, not just outputs
- Fail loudly when constraints are violated

## 2. Specification Before Code

**Tests define the contract.**

Before writing implementation code, write the tests that define what "correct" means.

```python
# Write this first
def test_hadamard_is_unitary():
    H = create_hadamard()
    assert np.allclose(H @ H.conj().T, np.eye(2))

def test_hadamard_is_hermitian():
    H = create_hadamard()
    assert np.allclose(H, H.conj().T)

def test_hadamard_squared_is_identity():
    H = create_hadamard()
    assert np.allclose(H @ H, np.eye(2))

# Then implement
def create_hadamard():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
```

Benefits:

- Forces you to think about requirements first
- Creates living documentation
- Catches regressions immediately

## 3. Fail Fast with Clarity

**Validate inputs, provide descriptive errors.**

When something goes wrong, fail immediately with a clear message.

```python
# Bad: Silent corruption
def apply_gate(gate, state):
    return gate @ state  # Silently produces garbage if shapes mismatch

# Good: Immediate, clear failure
def apply_gate(gate, state):
    if gate.shape[1] != state.shape[0]:
        raise ValueError(
            f"Gate dimension {gate.shape[1]} doesn't match "
            f"state dimension {state.shape[0]}"
        )
    return gate @ state
```

Key practices:

- Validate inputs at function boundaries
- Include relevant values in error messages
- Use specific exception types

## 4. Simplicity by Design

**Functions ≤50 lines, single responsibility.**

Complex code hides bugs. Keep functions small and focused.

```python
# Bad: 200-line function doing everything
def run_vqe_simulation(params, hamiltonian, optimizer, ...):
    # ... 200 lines of intertwined logic ...

# Good: Decomposed into testable units
def prepare_ansatz(params):
    """Prepare variational ansatz circuit."""
    ...  # 20 lines

def compute_expectation(state, hamiltonian):
    """Compute <psi|H|psi>."""
    ...  # 15 lines

def optimize_parameters(objective, initial_params, optimizer):
    """Run optimization loop."""
    ...  # 30 lines

def run_vqe_simulation(params, hamiltonian, optimizer):
    """Orchestrate VQE algorithm."""
    state = prepare_ansatz(params)
    energy = compute_expectation(state, hamiltonian)
    return optimize_parameters(energy, params, optimizer)
```

Benefits:

- Each function is testable in isolation
- Easier to understand and review
- AI agents can reason about small functions

## 5. Infrastructure Enables Speed

**CI, tests, linting from day one.**

Setting up infrastructure early makes you faster, not slower.

```bash
# Day 1 of a new project
bible init my-project --template python-scientific
cd my-project
pytest  # 28 tests pass immediately
```

What you get immediately:

- Pre-commit hooks (linting, formatting)
- CI/CD pipeline (tests, type checking)
- Coverage reporting
- Dependency scanning

The 10 minutes spent on setup saves hours of debugging.

## Applying the Principles

### For New Projects

1. Use `bible init` to start with infrastructure
2. Write tests before implementation
3. Apply validators to all physics functions
4. Keep functions under 50 lines
5. Run CI on every commit

### For Existing Projects

1. Add validators to critical functions first
2. Write tests for the most important code paths
3. Set up pre-commit hooks
4. Gradually refactor large functions
5. Add CI when ready

### For AI-Assisted Development

The principles work especially well with AI coding assistants:

- **Correctness First**: AI can suggest validators
- **Specification Before Code**: AI can generate tests from specs
- **Fail Fast**: AI can add validation code
- **Simplicity**: AI works better with small functions
- **Infrastructure**: `.cursorrules` guides AI behavior

## Further Reading

- [Style Guide](style-guide.md) - Detailed coding conventions
- [Validators Guide](../guide/validators.md) - Using physics validators
- [Testing Guide](../guide/testing.md) - Writing physics-aware tests
