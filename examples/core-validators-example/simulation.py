"""Core Validators Example - General-Purpose Scientific Validation.

This example demonstrates AgentBible's core validators, which work for any
scientific/numerical code - no domain-specific knowledge required.

Core validators catch the most common bugs in numerical code:
- NaN/Inf values (validate_finite)
- Out-of-range values (validate_positive, validate_non_negative, validate_range)
- Invalid probabilities (validate_probability, validate_probabilities)
- Unnormalized distributions (validate_normalized)

These validators work as decorators that automatically check function outputs.
If a check fails, a clear error is raised with context about what went wrong.

Example:
    $ python simulation.py
    Running physics simulation with validated outputs...
    Step 1: energy=100.0, momentum=[1. 2. 3.]
    ...

Dependencies:
    pip install agentbible numpy
"""

from __future__ import annotations

import numpy as np

# Core validators - always available, no optional dependencies
from agentbible import (
    validate_finite,
    validate_positive,
    validate_non_negative,
    validate_probability,
    validate_probabilities,
    validate_normalized,
    validate_range,
)

# Errors for catching validation failures
from agentbible.errors import (
    ValidationError,
    NonFiniteError,
    BoundsError,
    ProbabilityBoundsError,
    NormalizationError,
)


# ============================================================================
# Example 1: Simple Physics Simulation
# ============================================================================

@validate_finite
@validate_positive
def calculate_energy(mass: float, velocity: np.ndarray) -> float:
    """Calculate kinetic energy: E = 0.5 * m * v^2
    
    The decorators ensure:
    - Output is finite (no NaN/Inf)
    - Output is positive (energy must be > 0)
    
    If mass or velocity contains bad values, the output will fail validation.
    """
    return 0.5 * mass * np.dot(velocity, velocity)


@validate_finite
def calculate_momentum(mass: float, velocity: np.ndarray) -> np.ndarray:
    """Calculate momentum: p = m * v
    
    Output is checked for NaN/Inf values.
    """
    return mass * velocity


# ============================================================================
# Example 2: Statistical Distribution
# ============================================================================

@validate_finite
@validate_probabilities
def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities via softmax.
    
    The decorators ensure:
    - Output is finite (no NaN/Inf from numerical overflow)
    - All values are in [0, 1]
    """
    # Numerically stable softmax
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


@validate_finite
@validate_normalized()
def normalize_distribution(weights: np.ndarray) -> np.ndarray:
    """Normalize weights to a proper probability distribution.
    
    The decorators ensure:
    - Output is finite
    - Output sums to 1.0 (within tolerance)
    """
    return weights / weights.sum()


@validate_probability
def calculate_confidence(score: float, threshold: float = 0.5) -> float:
    """Convert a score to a confidence value in [0, 1].
    
    The decorator ensures the output is a valid probability.
    """
    return 1.0 / (1.0 + np.exp(-(score - threshold)))


# ============================================================================
# Example 3: Constrained Optimization
# ============================================================================

@validate_finite
@validate_range(min_val=-1.0, max_val=1.0)
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length.
    
    Components of a unit vector are always in [-1, 1].
    The decorator validates this constraint.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError("Cannot normalize zero vector")
    return v / norm


@validate_non_negative
def calculate_distances(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distances from center.
    
    Distances must be non-negative by definition.
    """
    return np.sqrt(np.sum((points - center) ** 2, axis=1))


# ============================================================================
# Demonstration
# ============================================================================

def demo_physics_simulation() -> None:
    """Demonstrate physics validators."""
    print("\n=== Physics Simulation ===")
    
    mass = 2.0
    velocity = np.array([1.0, 2.0, 3.0])
    
    energy = calculate_energy(mass, velocity)
    momentum = calculate_momentum(mass, velocity)
    
    print(f"Mass: {mass} kg")
    print(f"Velocity: {velocity} m/s")
    print(f"Kinetic Energy: {energy:.2f} J")
    print(f"Momentum: {momentum} kg·m/s")
    
    # Show what happens with bad data
    print("\nTrying to calculate energy with NaN velocity...")
    try:
        bad_velocity = np.array([1.0, np.nan, 3.0])
        calculate_energy(mass, bad_velocity)
    except NonFiniteError as e:
        print(f"  Caught error: {type(e).__name__}")
        print(f"  Message: {e}")


def demo_probability_validation() -> None:
    """Demonstrate probability validators."""
    print("\n=== Probability Validation ===")
    
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    print(f"Logits: {logits}")
    print(f"Softmax: {probs}")
    print(f"Sum: {probs.sum():.6f}")
    
    weights = np.array([3.0, 1.0, 1.0])
    normalized = normalize_distribution(weights)
    print(f"\nWeights: {weights}")
    print(f"Normalized: {normalized}")
    print(f"Sum: {normalized.sum():.6f}")
    
    confidence = calculate_confidence(0.8)
    print(f"\nScore 0.8 -> Confidence: {confidence:.4f}")


def demo_bounds_validation() -> None:
    """Demonstrate bounds validators."""
    print("\n=== Bounds Validation ===")
    
    vector = np.array([3.0, 4.0, 0.0])
    unit_vector = normalize_vector(vector)
    print(f"Vector: {vector}")
    print(f"Unit vector: {unit_vector}")
    print(f"Norm: {np.linalg.norm(unit_vector):.6f}")
    
    points = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    center = np.array([0.0, 0.0])
    distances = calculate_distances(points, center)
    print(f"\nPoints: {points.tolist()}")
    print(f"Distances from origin: {distances}")


def demo_error_handling() -> None:
    """Show how to catch and handle validation errors."""
    print("\n=== Error Handling ===")
    
    # Catch specific error types
    try:
        # This would fail if we passed a value outside [0, 1]
        @validate_probability
        def bad_function() -> float:
            return 1.5  # Invalid probability!
        
        bad_function()
    except ProbabilityBoundsError as e:
        print(f"Caught ProbabilityBoundsError: {e}")
    
    # Catch any validation error
    try:
        @validate_positive
        def negative_result() -> float:
            return -5.0
        
        negative_result()
    except ValidationError as e:
        print(f"Caught ValidationError: {e}")


def main() -> None:
    """Run all demonstrations."""
    print("=" * 60)
    print("AgentBible Core Validators - Example")
    print("=" * 60)
    print("\nThese validators work for ANY scientific/numerical code.")
    print("No domain-specific knowledge required!")
    
    demo_physics_simulation()
    demo_probability_validation()
    demo_bounds_validation()
    demo_error_handling()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
