"""Test that saved models can be loaded and reproduced.

This script demonstrates:
1. Loading a model from HDF5 with provenance metadata
2. Verifying the metadata captures everything needed for reproducibility
3. Recreating predictions from saved weights

Example:
    $ python main.py  # First, train and save a model
    $ python test_reproducibility.py
    Loading model from model_weights.h5...
    Verifying reproducibility metadata...
    ✓ All reproducibility checks passed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from agentbible.provenance import load_with_metadata


def load_and_verify_model(filepath: str | Path = "model_weights.h5") -> None:
    """Load a saved model and verify its provenance metadata."""
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"Error: {filepath} not found.")
        print("Run 'python main.py' first to train and save a model.")
        sys.exit(1)

    print(f"Loading model from {filepath}...")
    print()

    # Load the model and metadata
    data, metadata = load_with_metadata(filepath)

    # Display saved arrays
    print("=" * 60)
    print("Saved Model Data")
    print("=" * 60)
    print()
    for name, array in data.items():
        print(f"  {name}: shape={array.shape}, dtype={array.dtype}")
    print()

    # Display provenance metadata
    print("=" * 60)
    print("Provenance Metadata")
    print("=" * 60)
    print()

    # Basic info
    print("Basic Info:")
    print(f"  Description: {metadata.get('description', 'N/A')}")
    print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
    print()

    # Git info
    print("Git Info:")
    git_sha = metadata.get("git_sha", "N/A")
    print(f"  SHA: {git_sha[:12]}..." if git_sha and git_sha != "N/A" else f"  SHA: {git_sha}")
    print(f"  Branch: {metadata.get('git_branch', 'N/A')}")
    print(f"  Dirty: {metadata.get('git_dirty', 'N/A')}")
    if metadata.get("git_diff"):
        diff_lines = metadata["git_diff"].split("\n")
        print(f"  Diff: {len(diff_lines)} lines saved")
    print()

    # Hardware info
    hardware = metadata.get("hardware", {})
    if hardware:
        print("Hardware Info:")
        print(f"  CPU: {hardware.get('cpu_model', 'N/A')}")
        print(f"  Cores: {hardware.get('cpu_count_logical', 'N/A')} logical")
        print(f"  Memory: {hardware.get('memory_total_gb', 'N/A')} GB")
        gpu_info = hardware.get("gpu_info")
        if gpu_info:
            for i, gpu in enumerate(gpu_info):
                print(f"  GPU {i}: {gpu.get('name', 'N/A')} ({gpu.get('memory', 'N/A')})")
        else:
            print("  GPU: None detected")
        print()

    # Package versions
    packages = metadata.get("packages", {})
    print("Key Packages:")
    for pkg, version in packages.items():
        print(f"  {pkg}: {version}")
    print()

    # Pip freeze
    pip_freeze = metadata.get("pip_freeze", [])
    if pip_freeze:
        print(f"Full Environment: {len(pip_freeze)} packages")
        print("  (Full pip freeze saved in metadata)")
        print()

    # Extra metadata (training metrics)
    extra = metadata.get("extra", {})
    if extra:
        print("Training Info:")
        metrics = extra.get("metrics", {})
        if metrics:
            print(f"  Train Accuracy: {metrics.get('train_accuracy', 'N/A'):.4f}")
            print(f"  Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
            print(f"  Samples: {metrics.get('n_samples', 'N/A')}")
            print(f"  Features: {metrics.get('n_features', 'N/A')}")
            print(f"  Classes: {metrics.get('n_classes', 'N/A')}")
        print(f"  Random Seed: {extra.get('random_seed', 'N/A')}")
        print()


def verify_reproducibility_requirements(filepath: str | Path = "model_weights.h5") -> bool:
    """Check that all reproducibility requirements are met."""
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"Error: {filepath} not found.")
        return False

    data, metadata = load_with_metadata(filepath)

    print("=" * 60)
    print("Reproducibility Verification")
    print("=" * 60)
    print()

    checks = []

    # Check 1: Git commit saved
    git_sha = metadata.get("git_sha")
    if git_sha:
        checks.append(("Git SHA saved", True))
    else:
        checks.append(("Git SHA saved", False))

    # Check 2: If dirty, diff is saved
    if metadata.get("git_dirty"):
        if metadata.get("git_diff"):
            checks.append(("Git diff saved (repo was dirty)", True))
        else:
            checks.append(("Git diff saved (repo was dirty)", False))
    else:
        checks.append(("Repo was clean (no diff needed)", True))

    # Check 3: Pip freeze saved
    pip_freeze = metadata.get("pip_freeze", [])
    if len(pip_freeze) > 0:
        checks.append((f"Pip freeze saved ({len(pip_freeze)} packages)", True))
    else:
        checks.append(("Pip freeze saved", False))

    # Check 4: Hardware info saved
    hardware = metadata.get("hardware", {})
    if hardware.get("cpu_model"):
        checks.append(("Hardware info saved", True))
    else:
        checks.append(("Hardware info saved", False))

    # Check 5: Random seed saved
    extra = metadata.get("extra", {})
    if extra.get("random_seed") is not None:
        checks.append(("Random seed saved", True))
    else:
        checks.append(("Random seed saved", False))

    # Check 6: Model weights saved
    if "coefficients" in data and "intercept" in data:
        checks.append(("Model weights saved", True))
    else:
        checks.append(("Model weights saved", False))

    # Display results
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("Result: All reproducibility requirements met!")
    else:
        print("Result: Some requirements not met. Check the logs above.")

    return all_passed


def demonstrate_weight_restoration(filepath: str | Path = "model_weights.h5") -> None:
    """Show how to restore model weights for inference."""
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"Skipping weight restoration demo: {filepath} not found")
        return

    print()
    print("=" * 60)
    print("Weight Restoration Demo")
    print("=" * 60)
    print()

    data, metadata = load_with_metadata(filepath)

    # Extract weights
    coefficients = data["coefficients"]
    intercept = data["intercept"]
    classes = data["classes"]

    print("Restored model weights:")
    print(f"  Coefficients shape: {coefficients.shape}")
    print(f"  Intercept shape: {intercept.shape}")
    print(f"  Classes: {classes}")
    print()

    # Create a simple prediction function using raw weights
    def predict_proba_from_weights(X: np.ndarray) -> np.ndarray:
        """Manual softmax prediction using saved weights."""
        logits = X @ coefficients.T + intercept
        # Stable softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs

    # Demo with random input
    np.random.seed(42)
    X_sample = np.random.randn(3, coefficients.shape[1])

    probs = predict_proba_from_weights(X_sample)
    print("Predictions from restored weights (3 random samples):")
    print(probs)
    print(f"Row sums: {probs.sum(axis=1)}")
    print()


def main() -> int:
    """Run all reproducibility tests."""
    print()
    print("AgentBible Reproducibility Test")
    print("================================")
    print()

    load_and_verify_model()
    passed = verify_reproducibility_requirements()
    demonstrate_weight_restoration()

    print()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
