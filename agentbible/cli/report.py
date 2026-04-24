"""Implementation of the `bible report` command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from agentbible.provenance import load_provenance_record, validate_provenance_record

console = Console()


def _git_status(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    dirty = bool(metadata.get("git_dirty", False))
    suffix = "dirty" if dirty else "clean"
    return f"{str(record.get('git_sha', 'unknown'))[:7]} ({suffix})"


def generate_markdown_report(record: dict[str, Any]) -> str:
    """Render a schema-compliant record as Markdown."""
    lines = [
        "AgentBible Provenance Report",
        "============================",
        f"Language: {record.get('language', 'unknown')}          Spec: {record.get('spec_version', 'unknown')}",
        f"Timestamp: {record.get('timestamp', 'unknown')}",
        f"Git SHA: {_git_status(record)}",
        "",
        "Checks:",
    ]
    for check in record.get("checks_passed", []):
        if check.get("passed"):
            lines.append(
                f"  ✓ {check['check_name']}  rtol={check['rtol']}  atol={check['atol']}  norm={check.get('norm_used', 'n/a')}"
            )
        else:
            lines.append(
                f"  ✗ {check['check_name']}  FAILED: {check.get('error_message', 'unknown failure')}"
            )
    return "\n".join(lines)


def run_report(
    filepath: str,
    output_format: str = "markdown",
    output: str | None = None,
) -> int:
    """Read a provenance record and render a report."""
    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]Error:[/] File not found: {filepath}")
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
    if output_format == "json":
        rendered = json.dumps(record, indent=2, sort_keys=True)
    else:
        rendered = generate_markdown_report(record)
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
    else:
        console.print(rendered)
    return 0
