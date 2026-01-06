# Core Validators Example

This example demonstrates AgentBible's **core validators** - general-purpose validation decorators that work for any scientific/numerical code without domain-specific knowledge.

## What's Demonstrated

1. **Physics Simulation** - Validate energy (positive) and momentum (finite)
2. **Probability Validation** - Validate softmax outputs and normalized distributions
3. **Bounds Validation** - Validate unit vectors and distances
4. **Error Handling** - Catch and handle specific error types

## Core Validators Available

| Validator | Ensures |
|-----------|---------|
| `@validate_finite` | No NaN or Inf values |
| `@validate_positive` | All values > 0 |
| `@validate_non_negative` | All values >= 0 |
| `@validate_range(min, max)` | Values within bounds |
| `@validate_probability` | Single value in [0, 1] |
| `@validate_probabilities` | Array values in [0, 1] |
| `@validate_normalized()` | Array sums to 1 |

## Usage

```bash
# Install agentbible (core only - no optional dependencies)
pip install agentbible

# Run the example
python simulation.py
```

## Key Principle

These validators implement **"Fail Fast with Clarity"** - when a numerical error occurs, you get an immediate, helpful error message instead of silently propagating bad values through your code.

Compare:

```python
# Without validation - NaN silently propagates
def calculate_energy(mass, velocity):
    return 0.5 * mass * np.dot(velocity, velocity)

# With validation - NaN caught immediately with helpful message
@validate_finite
@validate_positive
def calculate_energy(mass, velocity):
    return 0.5 * mass * np.dot(velocity, velocity)
```

## Sample Output

```
=== Physics Simulation ===
Mass: 2.0 kg
Velocity: [1. 2. 3.] m/s
Kinetic Energy: 14.00 J
Momentum: [2. 4. 6.] kg·m/s

Trying to calculate energy with NaN velocity...
  Caught error: NonFiniteError
  Message: 
ARRAY CONTAINS NON-FINITE VALUES
  Expected: All finite values (no NaN or Inf)
  Got: 1 NaN
  Function: calculate_energy
  ...
```
