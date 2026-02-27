"""Tests for bible audit context command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentbible.cli.context_audit import (
    ContextAuditor,
    ContextAuditResult,
    ContextIssue,
    run_audit_context,
)
from agentbible.cli.main import cli

# --- Sample content fixtures ---

MINIMAL_COMPLIANT = """\
# AGENTS.md

## Tools
- Test: `pytest -x`
- Lint: `ruff check .`
- Type: `mypy src/`

## Rules
- Max 50 lines per function
- No bare `except:`
"""

CODEBASE_OVERVIEW = """\
# AGENTS.md

## Project Structure
├── src/
│   ├── main.py
│   └── utils.py
├── tests/
│   └── test_main.py
└── pyproject.toml

## Tools
- Test: `pytest`
"""

WORKFLOW_CHECKLIST = """\
# AGENTS.md

## Tools
- Test: `pytest -x`

## Workflow
1. Pull latest changes
2. Run the test suite
3. Fix any failures
4. Run lint checks
5. Commit and push
"""

LONG_CODE_BLOCK = """\
# AGENTS.md

## Tools
- Test: `pytest -x`

## Example
```python
import os
import sys
import re
import json
import pathlib
import subprocess
result = subprocess.run(["pytest"])
```
"""

TOOL_SPEC_ONLY = """\
# AGENTS.md

## Tools
- Test: `pytest -x`
- Lint: `ruff check .`
- Type: `mypy src/`
"""

BEFORE_AFTER_CONTENT = """\
# AGENTS.md

## Workflow

Before committing: run all checks
After pushing: verify CI passes

## Tools
- Test: `pytest -x`
"""


class TestContextIssue:
    """Tests for ContextIssue dataclass."""

    def test_to_dict(self) -> None:
        """ContextIssue serializes to dict."""
        issue = ContextIssue(
            severity="error",
            kind="codebase-overview",
            message="Directory tree found",
            line_start=5,
            line_end=10,
        )
        d = issue.to_dict()
        assert d["severity"] == "error"
        assert d["kind"] == "codebase-overview"
        assert d["line_start"] == 5
        assert d["line_end"] == 10


class TestContextAuditResult:
    """Tests for ContextAuditResult dataclass."""

    def test_passed_when_score_high(self) -> None:
        """Result passes when tightness score >= 70."""
        result = ContextAuditResult(
            file="AGENTS.md",
            line_count=10,
            token_estimate=100,
            tightness_score=80,
        )
        assert result.passed

    def test_fails_when_score_low(self) -> None:
        """Result fails when tightness score < 70."""
        result = ContextAuditResult(
            file="AGENTS.md",
            line_count=100,
            token_estimate=2000,
            tightness_score=40,
        )
        assert not result.passed

    def test_to_dict_includes_reference(self) -> None:
        """to_dict includes arxiv reference."""
        result = ContextAuditResult(
            file="AGENTS.md",
            line_count=10,
            token_estimate=100,
            tightness_score=90,
        )
        d = result.to_dict()
        assert d["reference"] == "arxiv:2602.11988"
        assert "passed" in d
        assert "issues" in d


class TestContextAuditor:
    """Tests for ContextAuditor class."""

    def test_minimal_file_scores_high(self, tmp_path: Path) -> None:
        """Minimal compliant file gets score >= 70."""
        f = tmp_path / "AGENTS.md"
        f.write_text(MINIMAL_COMPLIANT)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        assert result.tightness_score >= 70
        assert result.passed

    def test_codebase_overview_flagged(self, tmp_path: Path) -> None:
        """Directory tree is flagged as error."""
        f = tmp_path / "AGENTS.md"
        f.write_text(CODEBASE_OVERVIEW)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        overview_issues = [i for i in result.issues if i.kind == "codebase-overview"]
        assert len(overview_issues) >= 1
        assert overview_issues[0].severity == "error"

    def test_workflow_checklist_flagged(self, tmp_path: Path) -> None:
        """Numbered workflow checklist is flagged as error."""
        f = tmp_path / "AGENTS.md"
        f.write_text(WORKFLOW_CHECKLIST)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        checklist_issues = [i for i in result.issues if i.kind == "workflow-checklist"]
        assert len(checklist_issues) >= 1
        assert checklist_issues[0].severity == "error"

    def test_long_code_block_flagged(self, tmp_path: Path) -> None:
        """Code block >5 lines is flagged as warning."""
        f = tmp_path / "AGENTS.md"
        f.write_text(LONG_CODE_BLOCK)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        code_issues = [i for i in result.issues if i.kind == "long-code-block"]
        assert len(code_issues) >= 1
        assert code_issues[0].severity == "warning"

    def test_short_code_block_not_flagged(self, tmp_path: Path) -> None:
        """Code block <= 5 lines is not flagged."""
        content = """\
# AGENTS.md

## Tools
- Test: `pytest -x`

## Example
```bash
pytest -x
ruff check .
```
"""
        f = tmp_path / "AGENTS.md"
        f.write_text(content)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        code_issues = [i for i in result.issues if i.kind == "long-code-block"]
        assert len(code_issues) == 0

    def test_file_over_60_lines_is_error(self, tmp_path: Path) -> None:
        """File exceeding 60 lines is flagged as error."""
        content = "# AGENTS.md\n" + "- line\n" * 65
        f = tmp_path / "AGENTS.md"
        f.write_text(content)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        length_issues = [i for i in result.issues if i.kind == "length"]
        assert len(length_issues) == 1
        assert length_issues[0].severity == "error"

    def test_file_31_to_60_lines_is_warning(self, tmp_path: Path) -> None:
        """File between 31-60 lines is flagged as warning."""
        content = "# AGENTS.md\n" + "- line\n" * 35
        f = tmp_path / "AGENTS.md"
        f.write_text(content)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        length_issues = [i for i in result.issues if i.kind == "length"]
        assert len(length_issues) == 1
        assert length_issues[0].severity == "warning"

    def test_tool_spec_lines_not_flagged(self, tmp_path: Path) -> None:
        """Tool specification lines are not flagged as issues."""
        f = tmp_path / "AGENTS.md"
        f.write_text(TOOL_SPEC_ONLY)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        # No overview, checklist, or code block issues
        bad_issues = [
            i
            for i in result.issues
            if i.kind in ("codebase-overview", "workflow-checklist", "long-code-block")
        ]
        assert len(bad_issues) == 0

    def test_cursorrules_file_works(self, tmp_path: Path) -> None:
        """Works with .cursorrules files same as AGENTS.md."""
        f = tmp_path / ".cursorrules"
        f.write_text(MINIMAL_COMPLIANT)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        assert result.tightness_score >= 70

    def test_score_decreases_with_errors(self, tmp_path: Path) -> None:
        """Each error reduces score by 25 points."""
        f = tmp_path / "AGENTS.md"
        f.write_text(CODEBASE_OVERVIEW)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        # One error (codebase overview) = 100 - 25 = 75 max, but may also have length warning
        assert result.tightness_score < 100

    def test_token_estimate_computed(self, tmp_path: Path) -> None:
        """Token estimate is roughly len(content) / 4."""
        content = "# AGENTS.md\n" + "x" * 400
        f = tmp_path / "AGENTS.md"
        f.write_text(content)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        # ~413 chars / 4 ≈ 103 tokens (rough estimate)
        assert result.token_estimate > 0

    def test_before_after_pattern_flagged(self, tmp_path: Path) -> None:
        """'Before X:' and 'After X:' patterns are flagged."""
        f = tmp_path / "AGENTS.md"
        f.write_text(BEFORE_AFTER_CONTENT)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        checklist_issues = [i for i in result.issues if i.kind == "workflow-checklist"]
        assert len(checklist_issues) >= 1

    def test_file_path_in_result(self, tmp_path: Path) -> None:
        """Result contains the file path."""
        f = tmp_path / "AGENTS.md"
        f.write_text(MINIMAL_COMPLIANT)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)

        assert "AGENTS.md" in result.file


class TestRunAuditContext:
    """Tests for run_audit_context function."""

    def test_missing_file_returns_error(self, capsys: pytest.CaptureFixture) -> None:  # noqa: ARG002
        """Missing file returns exit code 1."""
        exit_code = run_audit_context(file="/nonexistent/AGENTS.md")
        assert exit_code == 1

    def test_no_file_no_default_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No file and no default in cwd returns exit code 1."""
        monkeypatch.chdir(tmp_path)
        exit_code = run_audit_context(file=None)
        assert exit_code == 1

    def test_auto_detects_agents_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto-detects AGENTS.md in current directory."""
        (tmp_path / "AGENTS.md").write_text(MINIMAL_COMPLIANT)
        monkeypatch.chdir(tmp_path)

        exit_code = run_audit_context(file=None)
        assert exit_code == 0

    def test_auto_detects_cursorrules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto-detects .cursorrules when no AGENTS.md present."""
        (tmp_path / ".cursorrules").write_text(MINIMAL_COMPLIANT)
        monkeypatch.chdir(tmp_path)

        exit_code = run_audit_context(file=None)
        assert exit_code == 0

    def test_json_output_valid(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """JSON output is valid and has expected schema."""
        f = tmp_path / "AGENTS.md"
        f.write_text(MINIMAL_COMPLIANT)

        run_audit_context(file=str(f), output_json=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert "tightness_score" in data
        assert "passed" in data
        assert "issues" in data
        assert "reference" in data
        assert data["reference"] == "arxiv:2602.11988"

    def test_compliant_file_exit_0(self, tmp_path: Path) -> None:
        """Compliant file returns exit code 0."""
        f = tmp_path / "AGENTS.md"
        f.write_text(MINIMAL_COMPLIANT)

        exit_code = run_audit_context(file=str(f))
        assert exit_code == 0

    def test_bad_file_exit_1(self, tmp_path: Path) -> None:
        """File with multiple violations returns exit code 1 (score < 70)."""
        # Two errors: codebase overview (-25) + workflow checklist (-25) = score 50
        content = """\
# AGENTS.md

## Project Structure
├── src/
│   └── main.py
├── tests/
└── pyproject.toml

## Workflow
1. Pull latest changes
2. Run the test suite
3. Fix any failures
4. Run lint checks
5. Push and verify CI
"""
        f = tmp_path / "AGENTS.md"
        f.write_text(content)

        exit_code = run_audit_context(file=str(f))
        assert exit_code == 1


class TestCLIAuditContext:
    """Tests for bible audit context CLI command."""

    def test_audit_context_help(self) -> None:
        """audit context --help shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "context", "--help"])

        assert result.exit_code == 0
        assert "AGENTS.md" in result.output or "context" in result.output.lower()

    def test_audit_context_compliant_file(self, tmp_path: Path) -> None:
        """audit context passes compliant file."""
        f = tmp_path / "AGENTS.md"
        f.write_text(MINIMAL_COMPLIANT)

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "context", str(f)])

        assert result.exit_code == 0

    def test_audit_context_bad_file_exit_1(self, tmp_path: Path) -> None:
        """audit context fails on file with multiple violations (score < 70)."""
        # Two errors: codebase overview + workflow checklist = score 50
        content = """\
# AGENTS.md

## Structure
├── src/
│   └── main.py
└── tests/

## Workflow
1. Pull latest changes
2. Run the test suite
3. Fix any failures
4. Push and verify CI
"""
        f = tmp_path / "AGENTS.md"
        f.write_text(content)

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "context", str(f)])

        assert result.exit_code == 1

    def test_audit_context_json_output(self, tmp_path: Path) -> None:
        """audit context --json produces valid JSON."""
        f = tmp_path / "AGENTS.md"
        f.write_text(MINIMAL_COMPLIANT)

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "context", str(f), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "tightness_score" in data
        assert "passed" in data

    def test_audit_group_help(self) -> None:
        """audit --help shows group help with subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "--help"])

        assert result.exit_code == 0
        assert "code" in result.output
        assert "context" in result.output
