"""Tests for the bible audit command."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentbible.cli.audit import AuditResult, CodeAuditor, Violation, run_audit
from agentbible.cli.main import cli


class TestViolation:
    """Tests for Violation dataclass."""

    def test_to_dict(self) -> None:
        """Violation converts to dict."""
        v = Violation(
            file="test.py",
            line=10,
            rule="rule-of-50",
            name="my_func",
            message="Too long",
            severity="error",
        )
        d = v.to_dict()

        assert d["file"] == "test.py"
        assert d["line"] == 10
        assert d["rule"] == "rule-of-50"
        assert d["name"] == "my_func"
        assert d["message"] == "Too long"
        assert d["severity"] == "error"


class TestAuditResult:
    """Tests for AuditResult dataclass."""

    def test_empty_result_passes(self) -> None:
        """Empty result passes."""
        result = AuditResult()
        assert result.passed
        assert result.violation_count == 0

    def test_warning_only_passes(self) -> None:
        """Result with only warnings passes."""
        result = AuditResult()
        result.violations.append(
            Violation(
                file="test.py",
                line=1,
                rule="missing-docstring",
                name="func",
                message="Missing docstring",
                severity="warning",
            )
        )
        assert result.passed
        assert result.warning_count == 1
        assert result.error_count == 0

    def test_error_fails(self) -> None:
        """Result with error fails."""
        result = AuditResult()
        result.violations.append(
            Violation(
                file="test.py",
                line=1,
                rule="rule-of-50",
                name="func",
                message="Too long",
                severity="error",
            )
        )
        assert not result.passed
        assert result.error_count == 1

    def test_to_dict(self) -> None:
        """AuditResult converts to dict."""
        result = AuditResult(files_checked=5)
        d = result.to_dict()

        assert d["files_checked"] == 5
        assert d["passed"] is True
        assert d["summary"]["total_violations"] == 0


class TestCodeAuditor:
    """Tests for CodeAuditor class."""

    def test_audit_file_rule_of_50(self, tmp_path: Path) -> None:
        """Detects functions exceeding line limit."""
        # Create a file with a long function
        code = (
            '''
def long_function() -> None:
    """A function that is too long."""
'''
            + "    x = 1\n" * 55
        )  # 55 lines of body

        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        auditor = CodeAuditor(max_function_lines=50)
        violations = auditor.audit_file(test_file)

        rule_of_50 = [v for v in violations if v.rule == "rule-of-50"]
        assert len(rule_of_50) == 1
        assert "55 lines" in rule_of_50[0].message

    def test_audit_file_missing_docstring(self, tmp_path: Path) -> None:
        """Detects missing docstrings on public functions."""
        code = textwrap.dedent("""
            def public_function() -> int:
                return 42
        """)

        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        auditor = CodeAuditor()
        violations = auditor.audit_file(test_file)

        assert any(v.rule == "missing-docstring" for v in violations)

    def test_audit_file_missing_type_hints(self, tmp_path: Path) -> None:
        """Detects missing type hints."""
        code = textwrap.dedent('''
            def untyped_function(x, y):
                """A function without type hints."""
                return x + y
        ''')

        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        auditor = CodeAuditor()
        violations = auditor.audit_file(test_file)

        type_hint_violations = [v for v in violations if v.rule == "missing-type-hints"]
        assert len(type_hint_violations) == 1
        assert "return type" in type_hint_violations[0].message
        assert "param 'x'" in type_hint_violations[0].message

    def test_audit_file_private_function_skipped(self, tmp_path: Path) -> None:
        """Private functions skip docstring check."""
        code = textwrap.dedent("""
            def _private_function() -> int:
                return 42
        """)

        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        auditor = CodeAuditor()
        violations = auditor.audit_file(test_file)

        assert not any(v.rule == "missing-docstring" for v in violations)

    def test_audit_file_syntax_error(self, tmp_path: Path) -> None:
        """Handles syntax errors gracefully."""
        code = "def broken("  # Syntax error

        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        auditor = CodeAuditor()
        violations = auditor.audit_file(test_file)

        assert len(violations) == 1
        assert violations[0].rule == "parse-error"
        assert violations[0].severity == "error"

    def test_audit_file_class_missing_docstring(self, tmp_path: Path) -> None:
        """Detects missing docstrings on classes."""
        code = textwrap.dedent("""
            class MyClass:
                pass
        """)

        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        auditor = CodeAuditor()
        violations = auditor.audit_file(test_file)

        assert any(
            v.rule == "missing-docstring" and v.name == "MyClass" for v in violations
        )

    def test_strict_mode_makes_warnings_errors(self, tmp_path: Path) -> None:
        """Strict mode promotes warnings to errors."""
        code = textwrap.dedent("""
            def func() -> int:
                return 42
        """)

        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        auditor = CodeAuditor(strict=True)
        violations = auditor.audit_file(test_file)

        assert all(v.severity == "error" for v in violations)

    def test_audit_directory(self, tmp_path: Path) -> None:
        """Audits all Python files in directory."""
        # Create multiple files
        (tmp_path / "file1.py").write_text("def f1() -> int:\n    return 1")
        (tmp_path / "file2.py").write_text("def f2() -> int:\n    return 2")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.py").write_text("def f3() -> int:\n    return 3")

        auditor = CodeAuditor()
        result = auditor.audit_directory(tmp_path, exclude_patterns=[])

        assert result.files_checked == 3

    def test_audit_directory_excludes_patterns(self, tmp_path: Path) -> None:
        """Excludes files matching patterns."""
        (tmp_path / "main.py").write_text("def main() -> int:\n    return 1")
        (tmp_path / "test_main.py").write_text("def test_main() -> None:\n    pass")

        auditor = CodeAuditor()
        result = auditor.audit_directory(tmp_path, exclude_patterns=["**/test_*.py"])

        assert result.files_checked == 1


class TestRunAudit:
    """Tests for run_audit function."""

    def test_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """JSON output is valid JSON."""
        code = textwrap.dedent('''
            def good_func() -> int:
                """A good function."""
                return 42
        ''')
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        exit_code = run_audit(str(test_file), output_format="json")

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert exit_code == 0
        assert data["passed"] is True
        assert data["files_checked"] == 1

    def test_exit_code_on_error(self, tmp_path: Path) -> None:
        """Returns non-zero exit code on errors."""
        # Create file with rule-of-50 violation
        code = (
            '''
def long_function():
    """Too long."""
'''
            + "    x = 1\n" * 55
        )

        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        exit_code = run_audit(str(test_file), output_format="json")

        assert exit_code == 1

    def test_path_not_found(self, capsys: pytest.CaptureFixture) -> None:
        """Returns error for non-existent path."""
        exit_code = run_audit("/nonexistent/path", output_format="text")

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()


class TestCLIAudit:
    """Tests for audit CLI command."""

    def test_audit_help(self) -> None:
        """audit --help shows command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "--help"])

        assert result.exit_code == 0
        assert "Audit code" in result.output
        assert "--json" in result.output
        assert "--strict" in result.output
        assert "--max-lines" in result.output

    def test_audit_file_json(self, tmp_path: Path) -> None:
        """audit command produces valid JSON."""
        code = textwrap.dedent('''
            def good_func() -> int:
                """A good function."""
                return 42
        ''')
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", str(test_file), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["passed"] is True

    def test_audit_directory(self, tmp_path: Path) -> None:
        """audit command works on directories."""
        (tmp_path / "file.py").write_text(
            'def f() -> int:\n    """Doc."""\n    return 1'
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["audit", str(tmp_path), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["files_checked"] >= 1
