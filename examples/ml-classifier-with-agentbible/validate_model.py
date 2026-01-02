"""Validate a saved model and demonstrate error handling.

This script shows how AgentBible validators catch common ML problems:
1. NaN/Inf values in predictions
2. Probabilities outside [0, 1]
3. Distributions that don't sum to 1

These issues are common in real ML workflows due to:
- Numerical instability (log of zero, division by zero)
- Model bugs (missing softmax, wrong activation)
- Data issues (extreme outliers, missing values)

Example:
    $ python validate_model.py
    Demonstrating validation failures...
    
    Test 1: NaN in predictions
    ✗ NonFiniteError: Array contains non-finite values...
"""

from __future__ import annotations

import numpy as np
from agentbible.errors import NonFiniteError, ProbabilityBoundsError, NormalizationError
from agentbible.validators import (
    validate_finite,
    validate_probabilities,
    validate_normalized,
    ValidationError,
)


def demonstrate_nan_detection() -> None:
    """Show how NaN values are caught immediately."""
    print("=" * 60)
    print("Test 1: Detecting NaN in predictions")
    print("=" * 60)
    print()
    print("Scenario: Model produces NaN due to log(0) or 0/0")
    print()

    @validate_finite
    def bad_prediction() -> np.ndarray:
        """Simulates a model that produces NaN."""
        probs = np.array([0.3, np.nan, 0.4])
        return probs

    try:
        bad_prediction()
        print("ERROR: Should have raised!")
    except NonFiniteError as e:
        print(f"✓ Caught NonFiniteError:")
        # Show just the first few lines of the error
        error_lines = str(e).split("\n")[:6]
        for line in error_lines:
            print(f"  {line}")
        print()


def demonstrate_inf_detection() -> None:
    """Show how Inf values are caught."""
    print("=" * 60)
    print("Test 2: Detecting Inf in predictions")
    print("=" * 60)
    print()
    print("Scenario: Model produces Inf due to overflow or division")
    print()

    @validate_finite
    def overflow_prediction() -> np.ndarray:
        """Simulates exponential overflow."""
        # This is common in softmax without log-sum-exp trick
        x = np.exp(1000)  # Inf
        return np.array([x, 0.5, 0.5])

    try:
        overflow_prediction()
        print("ERROR: Should have raised!")
    except NonFiniteError as e:
        print(f"✓ Caught NonFiniteError:")
        error_lines = str(e).split("\n")[:6]
        for line in error_lines:
            print(f"  {line}")
        print()


def demonstrate_probability_bounds() -> None:
    """Show how out-of-bounds probabilities are caught."""
    print("=" * 60)
    print("Test 3: Detecting probabilities outside [0, 1]")
    print("=" * 60)
    print()
    print("Scenario: Bug in softmax implementation produces values > 1")
    print()

    @validate_probabilities
    def buggy_softmax() -> np.ndarray:
        """Simulates a broken softmax."""
        # Forgot to normalize!
        return np.array([0.3, 1.5, 0.2])

    try:
        buggy_softmax()
        print("ERROR: Should have raised!")
    except ProbabilityBoundsError as e:
        print(f"✓ Caught ProbabilityBoundsError:")
        error_lines = str(e).split("\n")[:6]
        for line in error_lines:
            print(f"  {line}")
        print()


def demonstrate_normalization() -> None:
    """Show how un-normalized distributions are caught."""
    print("=" * 60)
    print("Test 4: Detecting distributions that don't sum to 1")
    print("=" * 60)
    print()
    print("Scenario: Probabilities are valid individually but don't sum to 1")
    print()

    @validate_normalized
    def unnormalized_probs() -> np.ndarray:
        """Simulates forgetting to normalize."""
        # Each is in [0, 1] but sum != 1
        return np.array([0.3, 0.3, 0.3])  # Sum = 0.9

    try:
        unnormalized_probs()
        print("ERROR: Should have raised!")
    except NormalizationError as e:
        print(f"✓ Caught NormalizationError:")
        error_lines = str(e).split("\n")[:6]
        for line in error_lines:
            print(f"  {line}")
        print()


def demonstrate_valid_predictions() -> None:
    """Show that valid predictions pass all checks."""
    print("=" * 60)
    print("Test 5: Valid predictions pass all validators")
    print("=" * 60)
    print()

    @validate_finite
    @validate_probabilities
    @validate_normalized
    def good_prediction() -> np.ndarray:
        """A properly normalized probability distribution."""
        return np.array([0.2, 0.5, 0.3])

    result = good_prediction()
    print(f"✓ Valid prediction: {result}")
    print(f"  Sum: {result.sum()}")
    print(f"  All in [0,1]: {(result >= 0).all() and (result <= 1).all()}")
    print(f"  All finite: {np.isfinite(result).all()}")
    print()


def demonstrate_chained_validators() -> None:
    """Show how validators can be stacked."""
    print("=" * 60)
    print("Test 6: Stacking validators for comprehensive checks")
    print("=" * 60)
    print()
    print("Multiple validators check different properties in sequence.")
    print("The first failure stops execution immediately.")
    print()

    @validate_finite  # Check 1: No NaN/Inf
    @validate_probabilities  # Check 2: All in [0, 1]
    @validate_normalized  # Check 3: Sum to 1
    def comprehensive_check() -> np.ndarray:
        return np.array([0.25, 0.25, 0.5])

    result = comprehensive_check()
    print(f"✓ Passed all 3 validators: {result}")
    print()


def main() -> int:
    """Run all validation demonstrations."""
    print()
    print("AgentBible Validation Demonstrations")
    print("====================================")
    print()
    print("This script shows how AgentBible catches common ML problems")
    print("before they corrupt your results or waste compute time.")
    print()

    demonstrate_nan_detection()
    demonstrate_inf_detection()
    demonstrate_probability_bounds()
    demonstrate_normalization()
    demonstrate_valid_predictions()
    demonstrate_chained_validators()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("AgentBible validators provide:")
    print("  1. Early detection - Fail at the source, not downstream")
    print("  2. Clear messages - What went wrong and how to fix it")
    print("  3. Academic refs - Links to relevant literature")
    print("  4. Zero overhead - No validation at production time (optional)")
    print()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
