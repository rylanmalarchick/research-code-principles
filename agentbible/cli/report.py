"""Generate reports from HDF5 provenance metadata.

This module provides the `bible report` command for extracting and formatting
provenance metadata from HDF5 files created with `save_with_metadata()`.

Usage:
    bible report results.h5
    bible report results.h5 --format markdown --output report.md
    bible report results.h5 --format json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def _format_timestamp(ts: str | None) -> str:
    """Format ISO timestamp to human-readable string."""
    if not ts:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, AttributeError):
        return ts


def _format_git_status(metadata: dict[str, Any]) -> str:
    """Format git information as a string."""
    sha = metadata.get("git_sha", "Unknown")
    branch = metadata.get("git_branch", "Unknown")
    dirty = metadata.get("git_dirty", False)

    if sha and sha != "Unknown":
        short_sha = sha[:8]
        status = f"{branch}@{short_sha}"
        if dirty:
            status += " (dirty)"
        return status
    return "Not in git repository"


def _format_hardware_text(hardware: dict[str, Any] | None) -> list[str]:
    """Format hardware info as text lines."""
    if not hardware:
        return ["  No hardware info available"]

    lines = []
    if hardware.get("cpu_model"):
        lines.append(f"  CPU: {hardware['cpu_model']}")
    if hardware.get("cpu_count_logical"):
        physical = hardware.get("cpu_count_physical", "?")
        lines.append(f"  Cores: {physical} physical, {hardware['cpu_count_logical']} logical")
    if hardware.get("memory_total_gb"):
        lines.append(f"  Memory: {hardware['memory_total_gb']} GB")
    if hardware.get("gpu_info"):
        for i, gpu in enumerate(hardware["gpu_info"]):
            name = gpu.get("name", "Unknown")
            mem = gpu.get("memory", "Unknown")
            lines.append(f"  GPU {i}: {name} ({mem})")
    if hardware.get("cuda_version"):
        lines.append(f"  CUDA: {hardware['cuda_version']}")

    return lines if lines else ["  No hardware info available"]


def _format_packages_text(packages: dict[str, str] | None) -> list[str]:
    """Format package versions as text lines."""
    if not packages:
        return ["  No package info available"]

    lines = []
    for name, version in packages.items():
        lines.append(f"  {name}: {version}")
    return lines


def generate_text_report(
    filepath: Path,
    metadata: dict[str, Any],
    datasets: dict[str, tuple[tuple[int, ...], str]],
) -> str:
    """Generate a plain text report.

    Args:
        filepath: Path to the HDF5 file
        metadata: Provenance metadata dictionary
        datasets: Dictionary mapping dataset names to (shape, dtype) tuples

    Returns:
        Formatted text report
    """
    lines = [
        "=" * 60,
        f"Provenance Report: {filepath.name}",
        "=" * 60,
        "",
        "OVERVIEW",
        "-" * 40,
        f"  File: {filepath}",
        f"  Description: {metadata.get('description', 'No description')}",
        f"  Created: {_format_timestamp(metadata.get('timestamp'))}",
        "",
        "GIT STATUS",
        "-" * 40,
        f"  Status: {_format_git_status(metadata)}",
    ]

    if metadata.get("git_sha"):
        lines.append(f"  SHA: {metadata['git_sha']}")
    if metadata.get("git_branch"):
        lines.append(f"  Branch: {metadata['git_branch']}")
    if metadata.get("git_dirty"):
        lines.append("  Warning: Repository had uncommitted changes!")
        if metadata.get("git_diff"):
            diff_lines = metadata["git_diff"].split("\n")[:10]
            lines.append("  Diff preview (first 10 lines):")
            for dl in diff_lines:
                lines.append(f"    {dl}")
            if metadata.get("git_diff_truncated"):
                lines.append("    ... (diff truncated)")

    lines.extend([
        "",
        "ENVIRONMENT",
        "-" * 40,
        f"  Hostname: {metadata.get('hostname', 'Unknown')}",
        f"  Platform: {metadata.get('platform', 'Unknown')}",
        f"  Python: {metadata.get('python_version', 'Unknown').split()[0]}",
        "",
        "RANDOM SEEDS",
        "-" * 40,
    ])

    numpy_seed = metadata.get("numpy_seed")
    python_seed = metadata.get("python_seed")
    torch_seed = metadata.get("torch_seed")

    if numpy_seed or python_seed or torch_seed:
        if numpy_seed:
            lines.append(f"  NumPy: {numpy_seed}")
        if python_seed:
            lines.append(f"  Python: {python_seed}")
        if torch_seed:
            lines.append(f"  PyTorch: {torch_seed}")
    else:
        lines.append("  No seeds captured (set seeds explicitly for reproducibility)")

    lines.extend([
        "",
        "HARDWARE",
        "-" * 40,
        *_format_hardware_text(metadata.get("hardware")),
        "",
        "KEY PACKAGES",
        "-" * 40,
        *_format_packages_text(metadata.get("packages")),
    ])

    # Datasets
    lines.extend([
        "",
        "DATASETS",
        "-" * 40,
    ])
    for name, (shape, dtype) in datasets.items():
        lines.append(f"  {name}: shape={shape}, dtype={dtype}")

    # Extra metadata if present
    if metadata.get("extra"):
        lines.extend([
            "",
            "EXTRA METADATA",
            "-" * 40,
        ])
        for key, value in metadata["extra"].items():
            lines.append(f"  {key}: {value}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def generate_markdown_report(
    filepath: Path,
    metadata: dict[str, Any],
    datasets: dict[str, tuple[tuple[int, ...], str]],
) -> str:
    """Generate a Markdown report.

    Args:
        filepath: Path to the HDF5 file
        metadata: Provenance metadata dictionary
        datasets: Dictionary mapping dataset names to (shape, dtype) tuples

    Returns:
        Formatted Markdown report
    """
    lines = [
        f"# Provenance Report: {filepath.name}",
        "",
        "## Overview",
        "",
        f"- **File**: `{filepath}`",
        f"- **Description**: {metadata.get('description', 'No description')}",
        f"- **Created**: {_format_timestamp(metadata.get('timestamp'))}",
        "",
        "## Git Status",
        "",
        f"- **Status**: {_format_git_status(metadata)}",
    ]

    if metadata.get("git_sha"):
        lines.append(f"- **SHA**: `{metadata['git_sha']}`")
    if metadata.get("git_branch"):
        lines.append(f"- **Branch**: `{metadata['git_branch']}`")
    if metadata.get("git_dirty"):
        lines.append("")
        lines.append("> **Warning**: Repository had uncommitted changes when this file was created!")
        if metadata.get("git_diff"):
            lines.extend([
                "",
                "<details>",
                "<summary>Diff preview</summary>",
                "",
                "```diff",
                metadata["git_diff"][:2000],  # Limit in markdown
                "```",
                "",
                "</details>",
            ])

    lines.extend([
        "",
        "## Environment",
        "",
        f"- **Hostname**: `{metadata.get('hostname', 'Unknown')}`",
        f"- **Platform**: `{metadata.get('platform', 'Unknown')}`",
        f"- **Python**: `{metadata.get('python_version', 'Unknown').split()[0]}`",
        "",
        "## Random Seeds",
        "",
    ])

    numpy_seed = metadata.get("numpy_seed")
    python_seed = metadata.get("python_seed")
    torch_seed = metadata.get("torch_seed")

    if numpy_seed or python_seed or torch_seed:
        if numpy_seed:
            lines.append(f"- **NumPy**: `{numpy_seed}`")
        if python_seed:
            lines.append(f"- **Python**: `{python_seed}`")
        if torch_seed:
            lines.append(f"- **PyTorch**: `{torch_seed}`")
    else:
        lines.append("*No seeds captured. Set seeds explicitly for reproducibility.*")

    # Hardware
    lines.extend([
        "",
        "## Hardware",
        "",
    ])
    hardware = metadata.get("hardware")
    if hardware:
        if hardware.get("cpu_model"):
            lines.append(f"- **CPU**: {hardware['cpu_model']}")
        if hardware.get("cpu_count_logical"):
            physical = hardware.get("cpu_count_physical", "?")
            lines.append(f"- **Cores**: {physical} physical, {hardware['cpu_count_logical']} logical")
        if hardware.get("memory_total_gb"):
            lines.append(f"- **Memory**: {hardware['memory_total_gb']} GB")
        if hardware.get("gpu_info"):
            for i, gpu in enumerate(hardware["gpu_info"]):
                name = gpu.get("name", "Unknown")
                mem = gpu.get("memory", "Unknown")
                lines.append(f"- **GPU {i}**: {name} ({mem})")
        if hardware.get("cuda_version"):
            lines.append(f"- **CUDA**: {hardware['cuda_version']}")
    else:
        lines.append("*No hardware info available.*")

    # Packages
    lines.extend([
        "",
        "## Key Packages",
        "",
        "| Package | Version |",
        "|---------|---------|",
    ])
    packages = metadata.get("packages", {})
    for name, version in packages.items():
        lines.append(f"| {name} | {version} |")

    # Datasets
    lines.extend([
        "",
        "## Datasets",
        "",
        "| Name | Shape | Dtype |",
        "|------|-------|-------|",
    ])
    for name, (shape, dtype) in datasets.items():
        lines.append(f"| {name} | {shape} | {dtype} |")

    # Extra metadata
    if metadata.get("extra"):
        lines.extend([
            "",
            "## Extra Metadata",
            "",
        ])
        for key, value in metadata["extra"].items():
            lines.append(f"- **{key}**: {value}")

    # pip freeze in collapsible section
    if metadata.get("pip_freeze"):
        lines.extend([
            "",
            "## Full Package List",
            "",
            "<details>",
            "<summary>pip freeze output</summary>",
            "",
            "```",
            "\n".join(metadata["pip_freeze"]),
            "```",
            "",
            "</details>",
        ])

    return "\n".join(lines)


def generate_json_report(
    filepath: Path,
    metadata: dict[str, Any],
    datasets: dict[str, tuple[tuple[int, ...], str]],
) -> str:
    """Generate a JSON report.

    Args:
        filepath: Path to the HDF5 file
        metadata: Provenance metadata dictionary
        datasets: Dictionary mapping dataset names to (shape, dtype) tuples

    Returns:
        Formatted JSON report
    """
    report = {
        "file": str(filepath),
        "provenance": metadata,
        "datasets": {
            name: {"shape": list(shape), "dtype": dtype}
            for name, (shape, dtype) in datasets.items()
        },
    }
    return json.dumps(report, indent=2)


def run_report(
    filepath: str,
    output_format: str = "text",
    output_file: str | None = None,
) -> int:
    """Run the report command.

    Args:
        filepath: Path to the HDF5 file
        output_format: Output format (text, markdown, json)
        output_file: Optional output file path

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    try:
        import h5py
    except ImportError:
        console.print(
            "[red]Error:[/] h5py is required for report generation. "
            "Install with: pip install agentbible[hdf5]"
        )
        return 1

    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]Error:[/] File not found: {filepath}")
        return 1

    if not path.suffix.lower() in (".h5", ".hdf5"):
        console.print(f"[yellow]Warning:[/] File may not be HDF5: {filepath}")

    # Load metadata and dataset info
    try:
        with h5py.File(path, "r") as f:
            # Get provenance metadata
            if "provenance" in f.attrs:
                metadata = json.loads(f.attrs["provenance"])
            else:
                console.print(
                    "[yellow]Warning:[/] No provenance metadata found. "
                    "File may not have been created with save_with_metadata()."
                )
                metadata = {}

            # Get dataset info
            datasets: dict[str, tuple[tuple[int, ...], str]] = {}
            for name in f:
                ds = f[name]
                if hasattr(ds, "shape"):
                    datasets[name] = (ds.shape, str(ds.dtype))

    except Exception as e:
        console.print(f"[red]Error reading HDF5 file:[/] {e}")
        return 1

    # Generate report
    if output_format == "markdown":
        report = generate_markdown_report(path, metadata, datasets)
    elif output_format == "json":
        report = generate_json_report(path, metadata, datasets)
    else:
        report = generate_text_report(path, metadata, datasets)

    # Output
    if output_file:
        output_path = Path(output_file)
        output_path.write_text(report)
        console.print(f"[green]Report saved to:[/] {output_path}")
    else:
        console.print(report)

    return 0
