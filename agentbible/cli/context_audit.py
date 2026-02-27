"""Context file auditor for evidence-based minimal context.

Analyzes AGENTS.md or .cursorrules files against recommendations from
arxiv:2602.11988 ("Evaluating AGENTS.md for Coding Agents").

Key findings applied here:
- Codebase overviews provide zero navigation benefit
- Workflow checklists trigger over-exploration behaviors
- Tool specifications ARE reliably followed (1.6x compliance rate)
- Files >30 lines increase cost with no measured benefit
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# Anti-pattern detection patterns
_TREE_CHARS = re.compile(r"[├└│]──")
_FILE_BULLET = re.compile(r"^\s*-\s+`[^`/]+/[^`]*`\s*:", re.MULTILINE)
_NUMBERED_STEP = re.compile(r"^\s*\d+\.\s+\S", re.MULTILINE)
_BEFORE_AFTER = re.compile(
    r"^(Before|After|Step\s+\d+):?\s", re.MULTILINE | re.IGNORECASE
)
_CODE_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# Safe-section detection (tool commands, physics validators)
_TOOL_CMD = re.compile(
    r"(pytest|ruff|mypy|black|flake8|coverage|pip install|uv|poetry|make|cmake|cargo|go test"
    r"|validate_unitary|validate_density_matrix|np\.random\.seed)",
    re.IGNORECASE,
)

WARN_LINES = 30
ERROR_LINES = 60
MAX_CODE_BLOCK_LINES = 5

# Score penalties
_PENALTY_ERROR = 25
_PENALTY_WARNING = 10


@dataclass
class ContextIssue:
    """A single issue found in a context file."""

    severity: str  # "error" or "warning"
    kind: str  # "codebase-overview", "workflow-checklist", "long-code-block", "length"
    message: str
    line_start: int | None = None
    line_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity,
            "kind": self.kind,
            "message": self.message,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass
class ContextAuditResult:
    """Result from auditing a context file."""

    file: str
    line_count: int
    token_estimate: int
    tightness_score: int
    issues: list[ContextIssue] = field(default_factory=list)
    safe_sections: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if tightness score >= 70."""
        return self.tightness_score >= 70

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "file": self.file,
            "line_count": self.line_count,
            "token_estimate": self.token_estimate,
            "tightness_score": self.tightness_score,
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "safe_sections": self.safe_sections,
            "reference": "arxiv:2602.11988",
        }


class ContextAuditor:
    """Audits AGENTS.md / .cursorrules files for evidence-based minimal context."""

    def audit_file(self, filepath: Path) -> ContextAuditResult:
        """Audit a single context file.

        Args:
            filepath: Path to AGENTS.md or .cursorrules file.

        Returns:
            ContextAuditResult with score and issues.
        """
        content = filepath.read_text(encoding="utf-8")
        lines = content.splitlines()
        line_count = len(lines)
        token_estimate = len(content) // 4

        issues: list[ContextIssue] = []

        issues.extend(self._check_codebase_overview(content, lines))
        issues.extend(self._check_workflow_checklist(content, lines))
        issues.extend(self._check_long_code_blocks(content, lines))
        issues.extend(self._check_length(line_count))

        safe_sections = self._find_safe_sections(lines)
        score = self._compute_score(issues)

        return ContextAuditResult(
            file=str(filepath),
            line_count=line_count,
            token_estimate=token_estimate,
            tightness_score=score,
            issues=issues,
            safe_sections=safe_sections,
        )

    def _check_codebase_overview(
        self, content: str, lines: list[str]
    ) -> list[ContextIssue]:
        """Detect directory tree overviews."""
        issues: list[ContextIssue] = []

        # Find lines with tree characters
        tree_lines = [
            i + 1 for i, line in enumerate(lines) if _TREE_CHARS.search(line)
        ]

        if tree_lines:
            start = tree_lines[0]
            end = tree_lines[-1]
            span = end - start + 1
            issues.append(
                ContextIssue(
                    severity="error",
                    kind="codebase-overview",
                    message=f"Lines {start}-{end}: Directory tree ({span} lines) — remove entirely",
                    line_start=start,
                    line_end=end,
                )
            )

        # Find bullet-list file path descriptions (- `path/to/file`:)
        bullet_matches = list(_FILE_BULLET.finditer(content))
        if bullet_matches and not tree_lines:
            # Only flag separately if no tree was already found
            line_nums = [content[: m.start()].count("\n") + 1 for m in bullet_matches]
            if len(line_nums) >= 3:
                start = line_nums[0]
                end = line_nums[-1]
                issues.append(
                    ContextIssue(
                        severity="error",
                        kind="codebase-overview",
                        message=(
                            f"Lines {start}-{end}: File path listing ({len(line_nums)} items)"
                            " — remove entirely"
                        ),
                        line_start=start,
                        line_end=end,
                    )
                )

        return issues

    def _check_workflow_checklist(
        self, _content: str, lines: list[str]
    ) -> list[ContextIssue]:
        """Detect workflow checklists and procedural instructions."""
        issues: list[ContextIssue] = []

        # Detect numbered list sequences (≥4 consecutive numbered items)
        numbered_runs = self._find_numbered_runs(lines, min_run=4)
        for start, end in numbered_runs:
            span = end - start + 1
            issues.append(
                ContextIssue(
                    severity="error",
                    kind="workflow-checklist",
                    message=(
                        f"Lines {start}-{end}: Workflow checklist ({span} lines)"
                        " — reduce to tool commands only"
                    ),
                    line_start=start,
                    line_end=end,
                )
            )

        # Detect "Before X:" / "After X:" / "Step N:" patterns
        before_after_lines = [
            i + 1
            for i, line in enumerate(lines)
            if _BEFORE_AFTER.match(line.strip())
        ]
        if before_after_lines and not numbered_runs:
            for ln in before_after_lines:
                issues.append(
                    ContextIssue(
                        severity="error",
                        kind="workflow-checklist",
                        message=(
                            f"Line {ln}: Procedural instruction"
                            " — replace with single tool command"
                        ),
                        line_start=ln,
                        line_end=ln,
                    )
                )

        return issues

    def _check_long_code_blocks(
        self, content: str, _lines: list[str]
    ) -> list[ContextIssue]:
        """Detect fenced code blocks exceeding MAX_CODE_BLOCK_LINES lines."""
        issues: list[ContextIssue] = []

        for match in _CODE_FENCE.finditer(content):
            block_content = match.group(1)
            block_lines = block_content.rstrip("\n").splitlines()
            num_lines = len(block_lines)
            if num_lines > MAX_CODE_BLOCK_LINES:
                # Find line number in file
                start_line = content[: match.start()].count("\n") + 1
                end_line = content[: match.end()].count("\n") + 1
                issues.append(
                    ContextIssue(
                        severity="warning",
                        kind="long-code-block",
                        message=(
                            f"Lines {start_line}-{end_line}: Code example ({num_lines} lines)"
                            " — flag blocks >5 lines"
                        ),
                        line_start=start_line,
                        line_end=end_line,
                    )
                )

        return issues

    def _check_length(self, line_count: int) -> list[ContextIssue]:
        """Check total file length."""
        issues: list[ContextIssue] = []
        if line_count > ERROR_LINES:
            issues.append(
                ContextIssue(
                    severity="error",
                    kind="length",
                    message=(
                        f"Total length {line_count} lines exceeds maximum {ERROR_LINES}"
                    ),
                )
            )
        elif line_count > WARN_LINES:
            issues.append(
                ContextIssue(
                    severity="warning",
                    kind="length",
                    message=(
                        f"Total length {line_count} lines exceeds recommended {WARN_LINES}"
                    ),
                )
            )
        return issues

    def _find_numbered_runs(
        self, lines: list[str], min_run: int = 4
    ) -> list[tuple[int, int]]:
        """Find runs of consecutive numbered list items."""
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        prev_num = 0

        for i, line in enumerate(lines):
            m = re.match(r"^\s*(\d+)\.\s+\S", line)
            if m:
                num = int(m.group(1))
                if run_start is None:
                    run_start = i + 1
                    prev_num = num
                elif num == prev_num + 1:
                    prev_num = num
                else:
                    # Non-consecutive — end current run
                    if run_start is not None and (i - (run_start - 1)) >= min_run:
                        runs.append((run_start, i))
                    run_start = i + 1
                    prev_num = num
            else:
                if run_start is not None:
                    run_end = i
                    if (run_end - run_start + 1) >= min_run:
                        runs.append((run_start, run_end))
                    run_start = None
                    prev_num = 0

        # Close any open run at EOF
        if run_start is not None:
            run_end = len(lines)
            if (run_end - run_start + 1) >= min_run:
                runs.append((run_start, run_end))

        return runs

    def _find_safe_sections(self, lines: list[str]) -> list[dict[str, Any]]:
        """Find sections containing only safe/tool-spec content."""
        safe: list[dict[str, Any]] = []
        run_start: int | None = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            is_safe = (
                not stripped
                or stripped.startswith("#")
                or bool(_TOOL_CMD.search(line))
                or (stripped.startswith("-")
                and bool(_TOOL_CMD.search(line)))
            )
            if is_safe and bool(_TOOL_CMD.search(line)):
                if run_start is None:
                    run_start = i + 1
            else:
                if run_start is not None:
                    safe.append(
                        {
                            "line_start": run_start,
                            "line_end": i,
                            "description": "Tool specifications",
                        }
                    )
                    run_start = None

        if run_start is not None:
            safe.append(
                {
                    "line_start": run_start,
                    "line_end": len(lines),
                    "description": "Tool specifications",
                }
            )

        return safe

    def _compute_score(self, issues: list[ContextIssue]) -> int:
        """Compute tightness score (0-100, higher = more minimal)."""
        score = 100
        for issue in issues:
            if issue.severity == "error":
                score -= _PENALTY_ERROR
            else:
                score -= _PENALTY_WARNING
        return max(0, score)


def _find_default_context_file() -> Path | None:
    """Find AGENTS.md or .cursorrules in current directory."""
    for name in ("AGENTS.md", ".cursorrules"):
        p = Path(name)
        if p.exists():
            return p
    return None


def run_audit_context(
    file: str | None,
    output_json: bool = False,
) -> int:
    """Run the context audit command.

    Args:
        file: Path to context file (AGENTS.md or .cursorrules).
              Defaults to searching current directory.
        output_json: Output results as JSON.

    Returns:
        Exit code: 0 if tightness score >= 70, 1 otherwise.
    """
    if file:
        target = Path(file)
        if not target.exists():
            console.print(f"[red]Error:[/] File not found: {file}")
            return 1
    else:
        target = _find_default_context_file()
        if target is None:
            console.print(
                "[red]Error:[/] No AGENTS.md or .cursorrules found in current directory"
            )
            console.print("Specify a file: bible audit context <file>")
            return 1

    auditor = ContextAuditor()
    result = auditor.audit_file(target)

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_context_report(result)

    return 0 if result.passed else 1


def _print_context_report(result: ContextAuditResult) -> None:
    """Print a human-readable context audit report."""
    console.print()
    console.print("[bold]AgentBible Context Audit[/]")
    console.print("=" * 24)
    console.print(
        f"File: {result.file} "
        f"({result.line_count} lines, ~{result.token_estimate:,} tokens)"
    )

    score_color = "green" if result.passed else "red"
    console.print(
        f"Tightness score: [{score_color}]{result.tightness_score}/100[/]"
    )

    if result.issues:
        console.print()
        console.print("Issues:")
        for issue in result.issues:
            severity_color = "red" if issue.severity == "error" else "yellow"
            console.print(
                f"  [[{severity_color}]{issue.severity}[/]]   {issue.message}"
            )

    if result.safe_sections:
        console.print()
        console.print("Sections to keep:")
        for section in result.safe_sections:
            console.print(
                f"  Lines {section['line_start']}-{section['line_end']}:"
                f"  {section['description']} [green]✓[/]"
            )

    console.print()
    console.print("[dim]Reference: arxiv:2602.11988[/]")
    if not result.passed:
        console.print("[dim]Fix: bible generate-agents-md > AGENTS.md[/]")
