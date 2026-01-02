# AgentBible: Novelty Comparison Matrix

This document compares AgentBible to existing validation and testing tools, highlighting what makes it unique for research software development.

## Quick Comparison Table

| Feature | AgentBible | pytest | Hypothesis | Pydantic | Cerberus | Great Expectations |
|---------|------------|--------|------------|----------|----------|-------------------|
| **Physics-aware validation** | ✅ Built-in | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Academic citations in errors** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Automatic provenance tracking** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **NaN/Inf pre-checks** | ✅ Automatic | Manual | Manual | ❌ | ❌ | Partial |
| **Git diff capture** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **pip freeze in metadata** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Hardware snapshot** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Decorator-based validation** | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **HDF5 integration** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **AI agent friendly** | ✅ Designed for | ❌ | ❌ | ❌ | ❌ | ❌ |

## Detailed Comparisons

### AgentBible vs pytest

**pytest** is a testing framework. **AgentBible** is a validation and provenance framework.

| Aspect | pytest | AgentBible |
|--------|--------|------------|
| **When it runs** | At test time | At runtime (every function call) |
| **What it catches** | Test failures | Invalid physics outputs |
| **Error messages** | Assertion diffs | Physics context + references |
| **Reproducibility** | None built-in | Full environment capture |

**Use together**: pytest for tests, AgentBible for runtime validation.

```python
# pytest: Tests that your code is correct
def test_gate_is_unitary():
    gate = create_hadamard()
    assert is_unitary(gate)

# AgentBible: Validates every call, not just in tests
@validate_unitary
def create_hadamard():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
```

### AgentBible vs Hypothesis

**Hypothesis** generates test cases. **AgentBible** validates outputs.

| Aspect | Hypothesis | AgentBible |
|--------|------------|------------|
| **Purpose** | Property-based testing | Output validation |
| **When it runs** | Test time only | Every function call |
| **Focus** | Finding edge cases | Catching physics violations |
| **Domain knowledge** | General strategies | Physics-specific validators |

**Use together**: Hypothesis to find bugs, AgentBible to prevent them in production.

```python
# Hypothesis: Generate many test cases
@given(st.floats(0, 2*np.pi))
def test_rz_is_unitary(angle):
    gate = rz_gate(angle)
    assert is_unitary(gate)

# AgentBible: Validate every real call
@validate_unitary
def rz_gate(angle):
    return np.array([[np.exp(-1j*angle/2), 0], 
                     [0, np.exp(1j*angle/2)]])
```

### AgentBible vs Pydantic

**Pydantic** validates data structures. **AgentBible** validates physics.

| Aspect | Pydantic | AgentBible |
|--------|----------|------------|
| **Focus** | Data types & schemas | Physical constraints |
| **Domain** | Web APIs, configs | Scientific computing |
| **Validation type** | Type checking | Numerical properties |
| **Error messages** | Type mismatches | Physics violations + refs |

**Key difference**: Pydantic checks "is this a float?", AgentBible checks "is this matrix unitary?"

```python
# Pydantic: Type validation
class Config(BaseModel):
    learning_rate: float  # Checks it's a float

# AgentBible: Physics validation
@validate_unitary  # Checks U†U = I
def create_gate():
    return np.eye(2)
```

### AgentBible vs Cerberus

**Cerberus** validates dictionaries against schemas. **AgentBible** validates scientific outputs.

| Aspect | Cerberus | AgentBible |
|--------|----------|------------|
| **Input** | Dictionaries | NumPy arrays, scalars |
| **Constraints** | Schema rules | Physics laws |
| **Domain** | Document validation | Scientific computing |
| **Integration** | Manual checking | Decorator-based |

### AgentBible vs Great Expectations

**Great Expectations** validates data pipelines. **AgentBible** validates function outputs.

| Aspect | Great Expectations | AgentBible |
|--------|-------------------|------------|
| **Scope** | Data pipelines, ETL | Function outputs |
| **Integration** | Data warehouse tools | Python decorators |
| **Focus** | Data quality | Physics correctness |
| **Overhead** | Heavy (needs DB) | Lightweight (pure Python) |

## What Makes AgentBible Unique

### 1. Physics-First Design

Built-in validators for quantum mechanics, probability theory, and numerical computing:

```python
@validate_unitary      # Quantum gates
@validate_hermitian    # Observables  
@validate_density_matrix  # Quantum states
@validate_probabilities   # ML outputs
@validate_normalized      # Distributions
@validate_finite          # Numerical stability
```

### 2. Educational Error Messages

Errors include academic references and debugging guidance:

```
MATRIX IS NOT UNITARY

  Expected: U†U = I (conjugate transpose times matrix equals identity)
  Got: max|U†U - I| = 1.00e+00
  
  Reference: Nielsen & Chuang, 'Quantum Computation and Quantum 
             Information', Theorem 2.2

  Guidance: Your quantum gate is not reversible. Common causes:
    - Missing normalization factor (e.g., 1/√2 for Hadamard)
    - Incorrect matrix elements or signs
```

### 3. Automatic Provenance

Every saved result includes complete reproducibility info:

```python
save_with_metadata("results.h5", {"data": array})
# Automatically captures:
# - Git SHA, branch, diff (if dirty)
# - Full pip freeze
# - CPU model, core count, GPU info
# - Timestamps
# - Random seeds
```

### 4. NaN/Inf Pre-Checks

Validators check for non-finite values BEFORE physics checks:

```python
@validate_unitary
def create_gate():
    return np.array([[np.nan, 0], [0, 1]])

# Error: "Array contains NaN" (not "Matrix is not unitary")
# This is more helpful for debugging!
```

### 5. AI Agent Integration

Designed for AI-assisted development:

- Clear, parseable error messages
- Context files for AI agents
- Decorator-based API (easy for LLMs to use)
- Comprehensive docstrings

## When to Use What

| Scenario | Tool |
|----------|------|
| Running a test suite | pytest |
| Finding edge cases | Hypothesis |
| Validating API inputs | Pydantic |
| Validating config files | Cerberus |
| Data pipeline quality | Great Expectations |
| **Validating physics outputs** | **AgentBible** |
| **Ensuring reproducibility** | **AgentBible** |
| **AI-assisted research code** | **AgentBible** |

## Summary

AgentBible fills a gap that existing tools don't address:

1. **Domain-specific validation** for physics and scientific computing
2. **Educational errors** with academic references
3. **Automatic provenance** for reproducibility
4. **Numerical robustness** with NaN/Inf pre-checks
5. **AI-friendly design** for modern development workflows

It's designed to be used *alongside* pytest, Hypothesis, and other tools—not to replace them.
