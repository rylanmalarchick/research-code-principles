"""Implementation of the unified `bible validate` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from agentbible.errors import ValidationError
from agentbible.provenance import (
    CheckResult,
    load_provenance_record,
    validate_provenance_record,
)
from agentbible.validators import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    check_density_matrix,
    check_finite,
    check_hermitian,
    check_non_negative,
    check_normalized_l1,
    check_positive,
    check_positive_definite,
    check_positive_semidefinite,
    check_probabilities,
    check_symmetric,
    check_unitary,
)

console = Console()

AVAILABLE_CHECKS = {
    "finite_array": "All values finite",
    "positive_array": "All values strictly positive",
    "non_negative_array": "All values non-negative",
    "probability_array": "All values in [0, 1]",
    "normalized_l1": "Absolute L1 sum difference within tolerance",
    "symmetric": "Matrix symmetric within atol",
    "hermitian": "Matrix Hermitian within atol",
    "unitary": "Matrix unitary by Frobenius residual",
    "positive_definite": "Cholesky factorization succeeds",
    "positive_semidefinite": "All eigenvalues >= -atol",
    "density_matrix": "Hermitian + trace 1 + positive semidefinite",
}

CHECK_NORMS = {
    "finite_array": "n/a",
    "positive_array": "n/a",
    "non_negative_array": "n/a",
    "probability_array": "n/a",
    "normalized_l1": "l1",
    "symmetric": "max_elementwise",
    "hermitian": "max_elementwise",
    "unitary": "frobenius",
    "positive_definite": "n/a",
    "positive_semidefinite": "spectral",
    "density_matrix": "n/a",
}


def _load_python_file(filepath: Path) -> dict[str, np.ndarray]:
    suffix = filepath.suffix.lower()
    if suffix == ".npy":
        return {"array": np.load(filepath)}
    if suffix == ".npz":
        with np.load(filepath) as data:
            return {key: np.array(value) for key, value in data.items()}
    if suffix in {".h5", ".hdf5", ".hdf"}:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "h5py is required for HDF5 files. Install with: pip install agentbible[hdf5]"
            ) from exc
        with h5py.File(filepath, "r") as handle:
            return {key: np.array(handle[key]) for key in handle}
    raise ValueError(f"Unsupported file format: {suffix}")


def _python_check_result(
    array: np.ndarray,
    check_name: str,
    *,
    rtol: float,
    atol: float,
) -> CheckResult:
    try:
        if check_name == "finite_array":
            check_finite(array, name="array")
            return CheckResult(check_name, True, 0.0, 0.0, CHECK_NORMS[check_name])
        if check_name == "positive_array":
            check_positive(array, name="array")
            return CheckResult(check_name, True, 0.0, 0.0, CHECK_NORMS[check_name])
        if check_name == "non_negative_array":
            check_non_negative(array, name="array")
            return CheckResult(check_name, True, 0.0, 0.0, CHECK_NORMS[check_name])
        if check_name == "probability_array":
            check_probabilities(array, name="array")
            return CheckResult(check_name, True, 0.0, 0.0, CHECK_NORMS[check_name])
        if check_name == "normalized_l1":
            check_normalized_l1(array, name="array", rtol=0.0, atol=atol)
            return CheckResult(check_name, True, 0.0, atol, CHECK_NORMS[check_name])
        if check_name == "symmetric":
            check_symmetric(array, name="array", atol=atol)
            return CheckResult(check_name, True, 0.0, atol, CHECK_NORMS[check_name])
        if check_name == "hermitian":
            check_hermitian(array, name="array", atol=atol)
            return CheckResult(check_name, True, 0.0, atol, CHECK_NORMS[check_name])
        if check_name == "unitary":
            check_unitary(array, name="array", rtol=rtol, atol=atol)
            return CheckResult(check_name, True, rtol, atol, CHECK_NORMS[check_name])
        if check_name == "positive_definite":
            check_positive_definite(array, name="array")
            return CheckResult(check_name, True, 0.0, 0.0, CHECK_NORMS[check_name])
        if check_name == "positive_semidefinite":
            check_positive_semidefinite(array, name="array", atol=atol)
            return CheckResult(check_name, True, 0.0, atol, CHECK_NORMS[check_name])
        check_density_matrix(array, name="array", rtol=rtol, atol=atol)
        return CheckResult(check_name, True, rtol, atol, CHECK_NORMS[check_name])
    except ValidationError as exc:
        recorded_rtol = rtol if check_name in {"unitary", "density_matrix"} else 0.0
        recorded_atol = atol if check_name in {"normalized_l1", "symmetric", "hermitian", "unitary", "positive_semidefinite", "density_matrix"} else 0.0
        return CheckResult(
            check_name=check_name,
            passed=False,
            rtol=recorded_rtol,
            atol=recorded_atol,
            norm_used=CHECK_NORMS[check_name],
            error_message=str(exc),
        )


def _applicable_checks(array: np.ndarray) -> list[str]:
    checks = ["finite_array"]
    if np.isrealobj(array):
        checks.extend(["positive_array", "non_negative_array", "probability_array", "normalized_l1"])
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        if np.isrealobj(array):
            checks.append("symmetric")
        checks.extend(
            [
                "hermitian",
                "unitary",
                "positive_definite",
                "positive_semidefinite",
                "density_matrix",
            ]
        )
    return checks


def _print_results(dataset_name: str, results: list[CheckResult]) -> None:
    table = Table(title=f"{dataset_name}")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Details")
    for result in results:
        status = "[green]PASS[/]" if result.passed else "[red]FAIL[/]"
        detail = result.error_message or f"norm={result.norm_used} rtol={result.rtol} atol={result.atol}"
        table.add_row(result.check_name, status, detail)
    console.print(table)


def _run_python_validate(filepath: str, checks: list[str], rtol: float, atol: float) -> int:
    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]Error:[/] File not found: {filepath}")
        return 1
    try:
        data = _load_python_file(path)
    except Exception as exc:
        console.print(f"[red]Error loading file:[/] {exc}")
        return 1
    all_passed = True
    for dataset_name, array in data.items():
        selected = checks or _applicable_checks(array)
        results = [_python_check_result(array, check, rtol=rtol, atol=atol) for check in selected]
        _print_results(dataset_name, results)
        all_passed = all_passed and all(result.passed for result in results)
    return 0 if all_passed else 1


def _resolve_provenance_path(path: Path) -> Path:
    if path.suffix == ".json":
        return path
    sidecar = path.with_suffix(".provenance.json")
    return sidecar if sidecar.exists() else path


def _print_provenance_results(record: dict[str, Any], checks: list[str]) -> None:
    selected = checks or [str(item["check_name"]) for item in record["checks_passed"]]
    table = Table(title=str(record.get("language", "unknown")))
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Details")
    for item in record["checks_passed"]:
        if item["check_name"] not in selected:
            continue
        status = "[green]PASS[/]" if item["passed"] else "[red]FAIL[/]"
        detail = item.get("error_message") or f"norm={item.get('norm_used', 'n/a')} rtol={item['rtol']} atol={item['atol']}"
        table.add_row(str(item["check_name"]), status, detail)
    console.print(table)


def _run_provenance_validate(filepath: str, checks: list[str]) -> int:
    path = _resolve_provenance_path(Path(filepath))
    if not path.exists():
        console.print(f"[red]Error:[/] Provenance record not found: {filepath}")
        return 1
    try:
        record = load_provenance_record(path)
    except Exception as exc:
        console.print(f"[red]Error loading provenance:[/] {exc}")
        return 1
    errors = validate_provenance_record(record)
    if errors:
        for error in errors:
            console.print(f"[red]Schema error:[/] {error}")
        return 1
    _print_provenance_results(record, checks)
    selected = checks or [str(item["check_name"]) for item in record["checks_passed"]]
    filtered = [item for item in record["checks_passed"] if item["check_name"] in selected]
    return 0 if all(bool(item["passed"]) for item in filtered) else 1


def run_validate(
    filepath: str,
    checks: list[str] | None = None,
    *,
    lang: str = "python",
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> int:
    """Execute the validate command and return an exit code."""
    selected_checks = checks or []
    if lang == "python":
        return _run_python_validate(filepath, selected_checks, rtol, atol)
    return _run_provenance_validate(filepath, selected_checks)


def get_available_checks() -> dict[str, str]:
    """Return the supported check names."""
    return AVAILABLE_CHECKS.copy()
