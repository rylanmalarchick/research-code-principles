# ML Classifier with AgentBible

A practical example showing how to integrate AgentBible into a real machine learning workflow.

## What This Example Demonstrates

1. **Validated Predictions** - Automatically check that model outputs are valid probabilities
2. **Provenance Tracking** - Save models with complete reproducibility metadata
3. **Error Detection** - Catch NaN/Inf and invalid probabilities before they corrupt results

## Files

| File | Description |
|------|-------------|
| `main.py` | Train a classifier and save with provenance |
| `validate_model.py` | Demonstrate validation error handling |
| `test_reproducibility.py` | Load saved model and verify metadata |

## Quick Start

```bash
# Install dependencies
pip install agentbible[hdf5] scikit-learn

# Train and save a model
python main.py

# See validation in action
python validate_model.py

# Verify reproducibility metadata
python test_reproducibility.py
```

## How AgentBible Is Used

### 1. Validated Predictions

```python
from agentbible.validators import validate_finite, validate_probabilities

@validate_finite
@validate_probabilities
def predict_probabilities(model, X):
    """Predictions are automatically validated."""
    return model.predict_proba(X)
```

If the model produces NaN, Inf, or values outside [0, 1], a helpful error is raised immediately:

```
ARRAY CONTAINS NON-FINITE VALUES

  Expected: All finite values (no NaN or Inf)
  Got: 1 NaN
  Function: predict_probabilities

  Reference: Higham, 'Accuracy and Stability of Numerical Algorithms'

  Guidance: All values must be finite. Common causes:
    - Division by zero (check denominators)
    - Log of zero or negative number
    - Overflow in exponential (use log-space computation)
```

### 2. Provenance Tracking

```python
from agentbible.provenance import save_with_metadata

save_with_metadata(
    "model_weights.h5",
    data={"coefficients": model.coef_, "intercept": model.intercept_},
    description="Logistic regression classifier",
    extra={"accuracy": 0.95, "random_seed": 42},
)
```

The saved HDF5 file includes:

- **Git info**: SHA, branch, and full diff if repo is dirty
- **Environment**: Complete `pip freeze` output
- **Hardware**: CPU model, core count, GPU info, memory
- **Timestamps**: When the model was saved
- **Custom metadata**: Any extra info you provide

### 3. Loading with Metadata

```python
from agentbible.provenance import load_with_metadata

data, metadata = load_with_metadata("model_weights.h5")

print(metadata["git_sha"])        # ec53c561...
print(metadata["pip_freeze"])     # ['numpy==1.24.0', ...]
print(metadata["hardware"]["cpu_model"])  # Intel Core i9-13980HX
```

## Why This Matters

### Before AgentBible

```python
# Silent corruption - NaN propagates through entire pipeline
probs = model.predict_proba(X)  # Returns [[nan, nan, nan], ...]
loss = compute_loss(probs)       # Returns nan
optimizer.step()                 # Weights become nan
# 8 hours of training wasted
```

### After AgentBible

```python
@validate_finite
@validate_probabilities
def predict_probabilities(model, X):
    return model.predict_proba(X)

probs = predict_probabilities(model, X)
# NonFiniteError: Array contains non-finite values
#   Got: 1 NaN at index (0, 1)
#   Guidance: Check for log(0) or division by zero
```

The bug is caught immediately, with a helpful error message that includes academic references and debugging guidance.

## Output Example

```
$ python main.py
============================================================
ML Classifier with AgentBible Example
============================================================

2024-01-15 10:30:00 - INFO - Generating synthetic data...
2024-01-15 10:30:00 - INFO - Training logistic regression...
2024-01-15 10:30:01 - INFO - Train accuracy: 0.9500
2024-01-15 10:30:01 - INFO - Test accuracy: 0.9350

Testing validated predictions...
Sample probabilities (first 5 rows):
[[0.02 0.95 0.03]
 [0.89 0.08 0.03]
 [0.01 0.04 0.95]
 [0.85 0.10 0.05]
 [0.03 0.92 0.05]]

2024-01-15 10:30:01 - INFO - Model saved to model_weights.h5

Provenance metadata includes:
  - Git SHA: ec53c561abcd...
  - Git dirty: True
  - CPU: 13th Gen Intel Core i9-13980HX
  - GPU: [{'name': 'RTX 4070', 'memory': '8188 MiB'}]
  - pip packages: 42 installed

Done!
```

## Extending This Example

To adapt for your own ML workflow:

1. **Replace the model**: Use your own classifier/regressor
2. **Add custom validators**: Create validators for your specific constraints
3. **Include more metadata**: Add hyperparameters, dataset info, etc.
4. **Use in training loops**: Validate predictions at each epoch

## References

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- Kolmogorov, A. N. (1933). *Foundations of the Theory of Probability*.
