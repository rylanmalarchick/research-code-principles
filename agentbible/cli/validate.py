"""Implementation of the bible validate command.

Validates physics constraints in numpy/HDF5 data files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# Available validation checks
AVAILABLE_CHECKS = {
    "unitarity": "U @ U.H = I (unitary matrix)",
    "hermiticity": "A = A.H (Hermitian matrix)",
    "trace": "tr(A) = 1 (unit trace)",
    "positivity": "All eigenvalues >= 0 (positive semi-definite)",
    "normalization": "||v|| = 1 (normalized vector)",
    "all": "Run all applicable checks",
}


def _check_unitarity(matrix: np.ndarray, rtol: float, atol: float) -> tuple[bool, str]:
    """Check if matrix is unitary."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False, "Not a square matrix"

    product = matrix @ matrix.conj().T
    identity = np.eye(matrix.shape[0], dtype=complex)

    if np.allclose(product, identity, rtol=rtol, atol=atol):
        return True, "Matrix is unitary"
    else:
        max_diff: np.floating[Any] = np.max(np.abs(product - identity))
        return False, f"Not unitary (max deviation: {max_diff:.2e})"


def _check_hermiticity(
    matrix: np.ndarray, rtol: float, atol: float
) -> tuple[bool, str]:
    """Check if matrix is Hermitian."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False, "Not a square matrix"

    if np.allclose(matrix, matrix.conj().T, rtol=rtol, atol=atol):
        return True, "Matrix is Hermitian"
    else:
        max_diff: np.floating[Any] = np.max(np.abs(matrix - matrix.conj().T))
        return False, f"Not Hermitian (max asymmetry: {max_diff:.2e})"


def _check_trace(matrix: np.ndarray, rtol: float, atol: float) -> tuple[bool, str]:
    """Check if matrix has unit trace."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False, "Not a square matrix"

    trace = np.trace(matrix)
    if np.isclose(trace, 1.0, rtol=rtol, atol=atol):
        return True, f"Trace is 1 (actual: {trace:.6f})"
    else:
        return False, f"Trace is {trace:.6f}, expected 1.0"


def _check_positivity(matrix: np.ndarray, rtol: float, atol: float) -> tuple[bool, str]:
    """Check if matrix is positive semi-definite."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False, "Not a square matrix"

    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eigenvalue: np.floating[Any] = np.min(eigenvalues.real)

    if min_eigenvalue >= -atol:
        return True, f"Positive semi-definite (min eigenvalue: {min_eigenvalue:.2e})"
    else:
        return False, f"Negative eigenvalue: {min_eigenvalue:.2e}"


def _check_normalization(
    array: np.ndarray, rtol: float, atol: float
) -> tuple[bool, str]:
    """Check if array is normalized."""
    norm = np.linalg.norm(array)
    if np.isclose(norm, 1.0, rtol=rtol, atol=atol):
        return True, f"Normalized (norm: {norm:.6f})"
    else:
        return False, f"Not normalized (norm: {norm:.6f}, expected 1.0)"


CHECK_FUNCTIONS = {
    "unitarity": _check_unitarity,
    "hermiticity": _check_hermiticity,
    "trace": _check_trace,
    "positivity": _check_positivity,
    "normalization": _check_normalization,
}


def _load_file(filepath: Path) -> dict[str, np.ndarray]:
    """Load data from numpy or HDF5 file."""
    suffix = filepath.suffix.lower()

    if suffix == ".npy":
        # Single numpy array
        return {"array": np.load(filepath)}

    elif suffix == ".npz":
        # Numpy archive
        with np.load(filepath) as data:
            return dict(data)

    elif suffix in {".h5", ".hdf5", ".hdf"}:
        # HDF5 file
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "h5py is required for HDF5 files. "
                "Install with: pip install agentbible[hdf5]"
            ) from e

        data = {}
        with h5py.File(filepath, "r") as f:
            for key in f:
                data[key] = np.array(f[key])
        return data

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _determine_applicable_checks(array: np.ndarray, checks: list[str]) -> list[str]:
    """Determine which checks are applicable to an array."""
    if "all" in checks:
        applicable = []
        # Matrix checks
        if array.ndim == 2 and array.shape[0] == array.shape[1]:
            applicable.extend(["unitarity", "hermiticity", "trace", "positivity"])
        # Vector checks
        if array.ndim == 1 or (array.ndim == 2 and 1 in array.shape):
            applicable.append("normalization")
        return applicable if applicable else ["normalization"]  # Fallback
    else:
        return [c for c in checks if c in CHECK_FUNCTIONS]


def run_validate(
    filepath: str,
    checks: list[str],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> int:
    """Execute the validate command. Returns exit code."""
    path = Path(filepath)

    if not path.exists():
        console.print(f"[red]Error:[/] File not found: {filepath}")
        return 1

    # Load data
    try:
        data = _load_file(path)
    except Exception as e:
        console.print(f"[red]Error loading file:[/] {e}")
        return 1

    console.print(f"[bold blue]Validating '{path.name}'[/]")
    console.print(f"  Tolerance: rtol={rtol}, atol={atol}")
    console.print()

    all_passed = True

    for name, array in data.items():
        console.print(
            f"[bold]Dataset: {name}[/] (shape: {array.shape}, dtype: {array.dtype})"
        )

        # Determine applicable checks
        applicable = _determine_applicable_checks(array, checks)

        if not applicable:
            console.print("  [yellow]No applicable checks for this array[/]")
            continue

        # Run checks
        table = Table(show_header=True, header_style="bold")
        table.add_column("Check")
        table.add_column("Result")
        table.add_column("Details")

        for check_name in applicable:
            check_func = CHECK_FUNCTIONS[check_name]
            passed, message = check_func(array, rtol, atol)

            if passed:
                table.add_row(check_name, "[green]PASS[/]", message)
            else:
                table.add_row(check_name, "[red]FAIL[/]", message)
                all_passed = False

        console.print(table)
        console.print()

    # Summary
    if all_passed:
        console.print("[bold green]All checks passed![/]")
        return 0
    else:
        console.print("[bold red]Some checks failed.[/]")
        return 1


def get_available_checks() -> dict[str, str]:
    """Return dictionary of available checks and their descriptions."""
    return AVAILABLE_CHECKS.copy()
