"""Pytest configuration and shared fixtures for ML research.

This file is automatically loaded by pytest and provides:
- Reproducibility: Fixed random seeds for deterministic tests
- Common fixtures: Sample datasets and model configurations
- Custom markers: @pytest.mark.slow, @pytest.mark.requires_torch, etc.
"""

import numpy as np
import pytest
from numpy.typing import NDArray


# ============================================================================
# Reproducibility: Set seeds before each test
# ============================================================================
@pytest.fixture(autouse=True)
def set_random_seeds() -> None:
    """Set random seeds for reproducibility in all tests.

    This fixture runs automatically before every test to ensure
    deterministic behavior. Document any seed changes.

    Seeds:
        numpy: 42
        python random: 42
    """
    import random

    np.random.seed(42)
    random.seed(42)


# ============================================================================
# Sample Datasets
# ============================================================================
@pytest.fixture
def binary_classification_data() -> tuple:
    """Sample binary classification dataset.

    Returns:
        Tuple of (X, y) where X is (100, 5) features and y is (100,) labels.
    """
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data() -> tuple:
    """Sample multiclass classification dataset.

    Returns:
        Tuple of (X, y) where X is (150, 4) features and y is (150,) labels.
    """
    np.random.seed(42)
    X = np.random.randn(150, 4)
    # Create 3 classes based on feature combinations
    y = np.zeros(150, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] > 1.0] = 2
    return X, y


@pytest.fixture
def regression_data() -> tuple:
    """Sample regression dataset.

    Returns:
        Tuple of (X, y) where X is (100, 3) features and y is (100,) targets.
    """
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0] + 0.5 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def imbalanced_data() -> tuple:
    """Imbalanced binary classification dataset.

    Returns:
        Tuple of (X, y) with 90/10 class split.
    """
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.zeros(100, dtype=int)
    y[:10] = 1  # Only 10% positive class
    return X, y


# ============================================================================
# Train/Test Splits
# ============================================================================
@pytest.fixture
def train_test_indices() -> tuple:
    """Sample train/test index split.

    Returns:
        Tuple of (train_idx, test_idx) arrays.
    """
    np.random.seed(42)
    indices = np.arange(100)
    np.random.shuffle(indices)
    train_idx = indices[:80]
    test_idx = indices[80:]
    return train_idx, test_idx


@pytest.fixture
def leaky_indices() -> tuple:
    """Train/test indices with data leakage (for testing detection).

    Returns:
        Tuple of (train_idx, test_idx) with overlap.
    """
    train_idx = np.arange(60)  # 0-59
    test_idx = np.arange(50, 80)  # 50-79, overlaps on 50-59
    return train_idx, test_idx


# ============================================================================
# Feature Arrays
# ============================================================================
@pytest.fixture
def standardized_features() -> NDArray[np.floating]:
    """Standardized features (mean=0, std=1)."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    return (X - X.mean(axis=0)) / X.std(axis=0)


@pytest.fixture
def minmax_features() -> NDArray[np.floating]:
    """Min-max scaled features (range [0, 1])."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)


@pytest.fixture
def unscaled_features() -> NDArray[np.floating]:
    """Unscaled features with varying scales."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X[:, 0] *= 1000  # Large scale
    X[:, 1] *= 0.001  # Small scale
    return X


@pytest.fixture
def features_with_nan() -> NDArray[np.floating]:
    """Features containing NaN values (for testing detection)."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X[0, 0] = np.nan
    X[10, 2] = np.nan
    return X


@pytest.fixture
def features_with_inf() -> NDArray[np.floating]:
    """Features containing Inf values (for testing detection)."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X[0, 0] = np.inf
    X[10, 2] = -np.inf
    return X


# ============================================================================
# Predictions
# ============================================================================
@pytest.fixture
def valid_probabilities() -> NDArray[np.floating]:
    """Valid probability predictions (2 classes)."""
    np.random.seed(42)
    p = np.random.rand(100)
    return np.column_stack([1 - p, p])


@pytest.fixture
def invalid_probabilities() -> NDArray[np.floating]:
    """Invalid probabilities that don't sum to 1."""
    np.random.seed(42)
    return np.random.rand(100, 2)  # Won't sum to 1


# ============================================================================
# Tolerance Constants
# ============================================================================
@pytest.fixture
def tolerance() -> float:
    """Default numerical tolerance for floating-point comparisons."""
    return 1e-10


@pytest.fixture
def loose_tolerance() -> float:
    """Looser tolerance for ML operations with numerical error."""
    return 1e-6
