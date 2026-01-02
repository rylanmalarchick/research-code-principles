"""Train a simple classifier with AgentBible provenance and validation.

This example demonstrates how to use AgentBible in a real ML workflow:
1. Train a logistic regression classifier on synthetic data
2. Validate predictions with probability validators
3. Save model weights with full provenance metadata

The five research code principles are applied:
1. Correctness First - validate outputs at every step
2. Simple Beats Clever - straightforward scikit-learn pipeline
3. Make It Inspectable - log metrics and intermediate values
4. Fail Fast and Loud - catch NaN/Inf immediately
5. Reproducibility Is Sacred - save complete environment info

Example:
    $ python main.py
    Training classifier on synthetic data...
    Accuracy: 0.95
    Model saved to model_weights.h5 with provenance metadata

Dependencies:
    pip install agentbible[hdf5] numpy scikit-learn
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# AgentBible imports
from agentbible.validators import (
    validate_finite,
    validate_normalized,
    validate_probabilities,
)
from agentbible.provenance import save_with_metadata, get_provenance_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reproducibility: Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@validate_finite
@validate_probabilities
def predict_probabilities(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Get prediction probabilities with automatic validation.

    This function is decorated to automatically:
    1. Check all outputs are finite (no NaN/Inf)
    2. Check all probabilities are in [0, 1]

    If either check fails, a ValidationError is raised with a helpful message.

    Args:
        model: Trained sklearn classifier
        X: Feature matrix to predict on

    Returns:
        Probability matrix of shape (n_samples, n_classes)
    """
    return model.predict_proba(X)


@validate_finite
@validate_normalized(axis=1)
def predict_probabilities_normalized(
    model: LogisticRegression, X: np.ndarray
) -> np.ndarray:
    """Get prediction probabilities and ensure they sum to 1 per sample.

    This is stricter than predict_probabilities - it also validates
    that each row sums to exactly 1.0 (within tolerance).

    Args:
        model: Trained sklearn classifier
        X: Feature matrix to predict on

    Returns:
        Probability matrix where each row sums to 1.0
    """
    return model.predict_proba(X)


def train_classifier(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 3,
    test_size: float = 0.2,
) -> tuple[LogisticRegression, dict[str, float], np.ndarray, np.ndarray]:
    """Train a logistic regression classifier on synthetic data.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        n_classes: Number of classes
        test_size: Fraction of data to use for testing

    Returns:
        Tuple of (trained_model, metrics_dict, X_test, y_test)
    """
    logger.info(
        "Generating synthetic data: %d samples, %d features, %d classes",
        n_samples,
        n_features,
        n_classes,
    )

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=RANDOM_SEED,
    )

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    logger.info("Training logistic regression...")

    # Train the model
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        multi_class="multinomial",
    )
    model.fit(X_train, y_train)

    # Calculate metrics
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "test_size": test_size,
    }

    logger.info("Train accuracy: %.4f", train_accuracy)
    logger.info("Test accuracy: %.4f", test_accuracy)

    return model, metrics, X_test, y_test


def save_model_with_provenance(
    model: LogisticRegression,
    metrics: dict[str, float],
    output_path: str | Path = "model_weights.h5",
) -> None:
    """Save model weights to HDF5 with full provenance metadata.

    The saved file includes:
    - Model coefficients and intercept
    - Training metrics
    - Git commit, branch, and diff (if dirty)
    - Full pip freeze for exact reproducibility
    - Hardware info (CPU, GPU, memory)
    - Timestamps

    Args:
        model: Trained sklearn classifier
        metrics: Dictionary of training metrics
        output_path: Path to save the HDF5 file
    """
    output_path = Path(output_path)

    # Prepare data dictionary
    data = {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "classes": model.classes_.astype(np.float64),
    }

    # Save with full provenance
    save_with_metadata(
        filepath=output_path,
        data=data,
        description="Logistic regression classifier trained on synthetic data",
        extra={
            "metrics": metrics,
            "model_type": "LogisticRegression",
            "sklearn_params": model.get_params(),
            "random_seed": RANDOM_SEED,
        },
    )

    logger.info("Model saved to %s with provenance metadata", output_path)


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("ML Classifier with AgentBible Example")
    print("=" * 60)
    print()

    # Train the classifier
    model, metrics, X_test, y_test = train_classifier()

    # Demonstrate validated predictions
    print()
    print("Testing validated predictions...")

    # Get probabilities with automatic validation
    probas = predict_probabilities(model, X_test[:5])
    print(f"Sample probabilities (first 5 rows):\n{probas}")

    # Also test the normalized version
    probas_norm = predict_probabilities_normalized(model, X_test[:5])
    print(f"Row sums (should all be ~1.0): {probas_norm.sum(axis=1)}")

    # Save the model with provenance
    print()
    save_model_with_provenance(model, metrics)

    # Show what's in the provenance metadata
    print()
    print("Provenance metadata includes:")
    meta = get_provenance_metadata()
    print(f"  - Git SHA: {meta.get('git_sha', 'N/A')[:12]}..." if meta.get('git_sha') else "  - Git SHA: N/A")
    print(f"  - Git dirty: {meta.get('git_dirty', 'N/A')}")
    if meta.get('hardware'):
        hw = meta['hardware']
        print(f"  - CPU: {hw.get('cpu_model', 'N/A')}")
        print(f"  - GPU: {hw.get('gpu_info', 'None')}")
    if meta.get('pip_freeze'):
        print(f"  - pip packages: {len(meta['pip_freeze'])} installed")

    print()
    print("Done! Check model_weights.h5 for the saved model and metadata.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
