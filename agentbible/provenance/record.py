"""Schema-aware provenance helpers shared across AgentBible surfaces."""

from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

SPEC_VERSION = "1.0"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "provenance_v1.json"
LANGUAGES = {"python", "cpp", "rust", "julia"}
NORMS = {"frobenius", "max_elementwise", "spectral", "l1", "l2", "linf", "n/a"}


@dataclass
class CheckResult:
    """A single check result for a provenance record."""

    check_name: str
    passed: bool
    rtol: float = 0.0
    atol: float = 0.0
    norm_used: str = "n/a"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the check result."""
        return asdict(self)


@dataclass
class ProvenanceRecord:
    """A schema-compliant provenance record."""

    language: str
    timestamp: str
    git_sha: str
    checks_passed: list[CheckResult] = field(default_factory=list)
    metadata: dict[str, Any] | None = None
    spec_version: str = SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record to a dictionary."""
        payload: dict[str, Any] = {
            "spec_version": self.spec_version,
            "language": self.language,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "checks_passed": [item.to_dict() for item in self.checks_passed],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def to_json(self) -> str:
        """Serialize the record to formatted JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def load_provenance_schema() -> dict[str, Any]:
    """Load the bundled provenance schema."""
    return cast(dict[str, Any], json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _run_command(args: list[str], timeout: int = 5) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_info() -> dict[str, Any]:
    dirty = _run_command(["git", "status", "--porcelain"])
    return {
        "git_sha": _run_command(["git", "rev-parse", "HEAD"]) or "",
        "git_branch": _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "",
        "git_dirty": bool(dirty) if dirty is not None else False,
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except (FileNotFoundError, OSError):
        pass
    return platform.processor() or "unknown"


def _memory_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
                return round(mem_kb / (1024 * 1024), 2)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def _gpu_info() -> str | None:
    info = _run_command(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    if info:
        return "; ".join(line.strip() for line in info.splitlines() if line.strip())
    return None


def _mpi_info() -> tuple[int | None, int | None]:
    try:
        mpi4py_rc = importlib.import_module("mpi4py.rc")
        mpi4py = importlib.import_module("mpi4py")
        mpi4py_rc.__dict__["initialize"] = False
        mpi = mpi4py.MPI
    except (AttributeError, ImportError):
        return None, None
    if not mpi.Is_initialized():
        return None, None
    comm = mpi.COMM_WORLD
    return int(comm.Get_rank()), int(comm.Get_size())


def _package_versions() -> dict[str, str]:
    versions = {"numpy": np.__version__}
    for name in ["h5py", "scipy", "pandas", "torch"]:
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _pip_freeze() -> str:
    frozen = _run_command([sys.executable, "-m", "pip", "freeze"], timeout=30)
    return frozen or ""


def _quantum_runtime_info() -> tuple[str | None, int | None]:
    for module_name in ("qiskit_ibm_runtime", "pennylane"):
        try:
            __import__(module_name)
        except ImportError:
            continue
        return module_name, None
    return None, None


def _schema_metadata(base: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "git_branch": str(base.get("git_branch", "")),
        "git_dirty": bool(base.get("git_dirty", False)),
        "hostname": str(base.get("hostname", "")),
        "platform": str(base.get("platform", "")),
        "cpu_model": str(base.get("cpu_model", "")),
        "memory_gb": base.get("memory_gb"),
        "gpu_info": base.get("gpu_info"),
        "slurm_job_id": base.get("slurm_job_id"),
        "slurm_nodelist": base.get("slurm_nodelist"),
        "mpi_rank": base.get("mpi_rank"),
        "mpi_size": base.get("mpi_size"),
        "random_seed_numpy": base.get("random_seed_numpy"),
        "random_seed_python": base.get("random_seed_python"),
        "packages": dict(base.get("packages", {})),
        "pip_freeze": str(base.get("pip_freeze", "")),
        "quantum_backend": base.get("quantum_backend"),
        "quantum_shots": base.get("quantum_shots"),
    }


def get_provenance_metadata(
    description: str = "",
    extra: dict[str, Any] | None = None,
    include_pip_freeze: bool = True,
    include_hardware: bool = True,
) -> dict[str, Any]:
    """Collect backward-compatible Python provenance metadata."""
    git = _git_info()
    mpi_rank, mpi_size = _mpi_info()
    quantum_backend, quantum_shots = _quantum_runtime_info()
    memory_gb = _memory_gb() if include_hardware else None
    gpu_info = _gpu_info() if include_hardware else None
    metadata = {
        "description": description,
        "timestamp": utc_timestamp(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_model": _cpu_model() if include_hardware else "",
        "memory_gb": memory_gb,
        "gpu_info": gpu_info,
        "random_seed_numpy": None,
        "random_seed_python": None,
        "numpy_seed": None,
        "python_seed": None,
        "packages": _package_versions(),
        "pip_freeze": _pip_freeze() if include_pip_freeze else "",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "mpi_rank": mpi_rank,
        "mpi_size": mpi_size,
        "quantum_backend": quantum_backend,
        "quantum_shots": quantum_shots,
        "hardware": {
            "cpu_model": _cpu_model() if include_hardware else "",
            "memory_total_gb": memory_gb,
            "gpu_info": gpu_info,
        },
        **git,
    }
    if extra:
        metadata["extra"] = extra
    return metadata


def _normalize_checks(checks: Iterable[CheckResult | Mapping[str, Any]]) -> list[CheckResult]:
    items: list[CheckResult] = []
    for check in checks:
        if isinstance(check, CheckResult):
            items.append(check)
            continue
        items.append(
            CheckResult(
                check_name=str(check["check_name"]),
                passed=bool(check["passed"]),
                rtol=float(check.get("rtol", 0.0)),
                atol=float(check.get("atol", 0.0)),
                norm_used=str(check.get("norm_used", "n/a")),
                error_message=check.get("error_message"),
            )
        )
    return items


def build_provenance_record(
    *,
    language: str = "python",
    checks_passed: Iterable[CheckResult | Mapping[str, Any]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ProvenanceRecord:
    """Build a schema-compliant provenance record."""
    base = dict(metadata) if metadata is not None else get_provenance_metadata()
    return ProvenanceRecord(
        language=language,
        timestamp=str(base.get("timestamp", utc_timestamp())),
        git_sha=str(base.get("git_sha", "")),
        checks_passed=_normalize_checks(checks_passed),
        metadata=_schema_metadata(base),
    )


def _is_iso8601_utc(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _validate_check_result(check: Any, index: int) -> list[str]:
    if not isinstance(check, dict):
        return [f"checks_passed[{index}] must be an object"]
    errors: list[str] = []
    required = ("check_name", "passed", "rtol", "atol")
    for key in required:
        if key not in check:
            errors.append(f"checks_passed[{index}] missing '{key}'")
    if "norm_used" in check and check["norm_used"] not in NORMS:
        errors.append(f"checks_passed[{index}].norm_used is invalid")
    if "error_message" in check and not isinstance(check["error_message"], (str, type(None))):
        errors.append(f"checks_passed[{index}].error_message must be string or null")
    return errors


def validate_provenance_record(record: Mapping[str, Any]) -> list[str]:
    """Validate a record against the bundled schema contract."""
    errors: list[str] = []
    required = ("spec_version", "language", "timestamp", "git_sha", "checks_passed")
    for key in required:
        if key not in record:
            errors.append(f"missing top-level field '{key}'")
    if record.get("spec_version") != SPEC_VERSION:
        errors.append("spec_version must equal '1.0'")
    if record.get("language") not in LANGUAGES:
        errors.append("language must be one of python, cpp, rust, julia")
    if not _is_iso8601_utc(record.get("timestamp")):
        errors.append("timestamp must be an ISO 8601 date-time string")
    if not isinstance(record.get("git_sha"), str):
        errors.append("git_sha must be a string")
    checks = record.get("checks_passed")
    if not isinstance(checks, list):
        errors.append("checks_passed must be an array")
    else:
        for index, check in enumerate(checks):
            errors.extend(_validate_check_result(check, index))
    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        errors.append("metadata must be an object when present")
    return errors


def load_provenance_record(path: str | Path) -> dict[str, Any]:
    """Load a provenance record from disk."""
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def emit_provenance_record(
    path: str | Path,
    checks_passed: Iterable[CheckResult | Mapping[str, Any]] = (),
    metadata: Mapping[str, Any] | None = None,
    *,
    language: str = "python",
) -> Path:
    """Write a schema-compliant JSON provenance record."""
    source = Path(path)
    target = source if source.suffix == ".json" else source.with_suffix(".provenance.json")
    record = build_provenance_record(
        language=language,
        checks_passed=checks_passed,
        metadata=metadata,
    )
    payload = record.to_dict()
    errors = validate_provenance_record(payload)
    if errors:
        raise ValueError("; ".join(errors))
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target
