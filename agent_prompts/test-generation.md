# Test Generation Prompt

Use this when writing tests for research code.

---

## Before Writing Tests

1. What is the **happy path**? (Normal, expected input)
2. What are the **boundaries**? (Empty, single, max, zero)
3. What are the **invalid inputs**? (Wrong type, out of range, NaN)
4. What **physical constraints** must hold? (Unitarity, normalization)
5. What **ground truth** can we compare against? (Known values, other implementations)

---

## Test Structure (pytest)

```python
class TestMyFunction:
    """Tests for my_function."""

    # Happy path
    def test_normal_case(self):
        result = my_function(valid_input)
        assert result == expected_value

    # Boundaries
    def test_empty_input(self):
        result = my_function([])
        assert result == expected_for_empty

    def test_single_element(self):
        result = my_function([x])
        assert result == expected_for_single

    def test_maximum_size(self):
        result = my_function(large_input)
        assert result == expected_for_large

    # Invalid inputs
    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            my_function("not a list")

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            my_function(invalid_value)
```

---

## Physical Validation Tests

```python
def test_unitarity(self):
    """Gate must satisfy U†U = I."""
    U = create_gate()
    identity = np.eye(U.shape[0])
    assert np.allclose(U.conj().T @ U, identity)

def test_normalization(self):
    """State must satisfy ⟨ψ|ψ⟩ = 1."""
    state = create_state()
    assert np.isclose(np.linalg.norm(state), 1.0)

def test_probabilities(self):
    """Probabilities must be non-negative and sum to 1."""
    probs = measure(state)
    assert np.all(probs >= 0)
    assert np.isclose(np.sum(probs), 1.0)
```

---

## Reproducibility

```python
@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(42)  # Document seed value
```
