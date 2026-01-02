"""Code audit functionality for AgentBible principles compliance.

Checks Python code against AgentBible's 5 Principles:
1. Rule of 50: Functions should be <= 50 lines
2. Docstring presence: All public functions/classes need docstrings
3. Type hints: Function signatures should have type annotations

Output can be human-readable or JSON for CI integration.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class Violation:
    """A single code quality violation."""

    file: str
    line: int
    rule: str
    name: str
    message: str
    severity: str = "warning"  # "warning" or "error"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "file": self.file,
            "line": self.line,
            "rule": self.rule,
            "name": self.name,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class AuditResult:
    """Results from auditing a codebase."""

    files_checked: int = 0
    violations: list[Violation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def violation_count(self) -> int:
        """Total number of violations."""
        return len(self.violations)

    @property
    def error_count(self) -> int:
        """Count of error-severity violations."""
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of warning-severity violations."""
        return sum(1 for v in self.violations if v.severity == "warning")

    @property
    def passed(self) -> bool:
        """True if no error-severity violations."""
        return self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "files_checked": self.files_checked,
            "passed": self.passed,
            "summary": {
                "total_violations": self.violation_count,
                "errors": self.error_count,
                "warnings": self.warning_count,
            },
            "violations": [v.to_dict() for v in self.violations],
            "parse_errors": self.errors,
        }


class CodeAuditor:
    """Audits Python code for AgentBible principles compliance."""

    # Rule of 50: Maximum function lines (configurable)
    MAX_FUNCTION_LINES = 50

    def __init__(
        self,
        check_line_length: bool = True,
        check_docstrings: bool = True,
        check_type_hints: bool = True,
        max_function_lines: int = 50,
        strict: bool = False,
    ) -> None:
        """Initialize the auditor.

        Args:
            check_line_length: Check function line counts.
            check_docstrings: Check for docstring presence.
            check_type_hints: Check for type annotations.
            max_function_lines: Maximum allowed function lines.
            strict: If True, docstring/type hint issues are errors not warnings.
        """
        self.check_line_length = check_line_length
        self.check_docstrings = check_docstrings
        self.check_type_hints = check_type_hints
        self.max_function_lines = max_function_lines
        self.strict = strict

    def audit_file(self, filepath: Path) -> list[Violation]:
        """Audit a single Python file.

        Args:
            filepath: Path to the Python file.

        Returns:
            List of violations found.
        """
        violations: list[Violation] = []

        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(filepath))
        except SyntaxError as e:
            # Return a parse error as a violation
            return [
                Violation(
                    file=str(filepath),
                    line=e.lineno or 1,
                    rule="parse-error",
                    name="<module>",
                    message=f"Syntax error: {e.msg}",
                    severity="error",
                )
            ]
        except Exception as e:
            return [
                Violation(
                    file=str(filepath),
                    line=1,
                    rule="parse-error",
                    name="<module>",
                    message=f"Failed to parse: {e}",
                    severity="error",
                )
            ]

        # Walk the AST and check each function/method/class
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                violations.extend(self._check_function(node, filepath))
            elif isinstance(node, ast.ClassDef):
                violations.extend(self._check_class(node, filepath))

        return violations

    def _check_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        filepath: Path,
    ) -> list[Violation]:
        """Check a function definition for violations."""
        violations: list[Violation] = []
        is_private = node.name.startswith("_")

        # Rule of 50: Check line count
        if self.check_line_length:
            violation = self._check_line_count(node, filepath)
            if violation:
                violations.append(violation)

        # Docstring check (skip private methods)
        if self.check_docstrings and not is_private:
            violation = self._check_docstring(node, filepath, "function")
            if violation:
                violations.append(violation)

        # Type hints check (skip private functions in non-strict mode)
        if self.check_type_hints and (not is_private or self.strict):
            violation = self._check_type_hints_violation(node, filepath)
            if violation:
                violations.append(violation)

        return violations

    def _check_line_count(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        filepath: Path,
    ) -> Violation | None:
        """Check if function exceeds max line count."""
        body_lines = self._count_function_lines(node)
        if body_lines > self.max_function_lines:
            return Violation(
                file=str(filepath),
                line=node.lineno,
                rule="rule-of-50",
                name=node.name,
                message=f"Function has {body_lines} lines (max {self.max_function_lines})",
                severity="error",
            )
        return None

    def _check_docstring(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        filepath: Path,
        kind: str,
    ) -> Violation | None:
        """Check if node has a docstring."""
        if not ast.get_docstring(node):
            return Violation(
                file=str(filepath),
                line=node.lineno,
                rule="missing-docstring",
                name=node.name,
                message=f"Public {kind} missing docstring",
                severity="error" if self.strict else "warning",
            )
        return None

    def _check_type_hints_violation(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        filepath: Path,
    ) -> Violation | None:
        """Check for missing type hints and return violation if any."""
        missing_hints = self._get_missing_type_hints(node)
        if missing_hints:
            return Violation(
                file=str(filepath),
                line=node.lineno,
                rule="missing-type-hints",
                name=node.name,
                message=f"Missing type hints: {', '.join(missing_hints)}",
                severity="error" if self.strict else "warning",
            )
        return None

    def _check_class(self, node: ast.ClassDef, filepath: Path) -> list[Violation]:
        """Check a class definition for violations."""
        violations: list[Violation] = []
        is_private = node.name.startswith("_")

        # Docstring check for public classes
        if self.check_docstrings and not is_private:
            violation = self._check_docstring(node, filepath, "class")
            if violation:
                violations.append(violation)

        return violations

    def _count_function_lines(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> int:
        """Count the number of lines in a function body.

        Excludes the docstring from the count.
        """
        if not node.body:
            return 0

        # Get line range
        start_line = node.lineno
        end_line = node.end_lineno or node.lineno

        # Calculate total lines
        total_lines = end_line - start_line + 1

        # Subtract docstring lines if present
        first_stmt = node.body[0]
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        ):
            # It's a docstring
            docstring_lines = (
                (first_stmt.end_lineno or first_stmt.lineno) - first_stmt.lineno + 1
            )
            total_lines -= docstring_lines

        # Subtract the def line itself (we want body lines only)
        # But count decorator lines
        return max(0, total_lines - 1)

    def _get_missing_type_hints(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[str]:
        """Get list of missing type hints for a function.

        Returns list of missing hints (e.g., ["return type", "param 'x'"]).
        """
        missing: list[str] = []

        # Check return type (skip __init__ which conventionally has no return)
        if node.name != "__init__" and node.returns is None:
            missing.append("return type")

        # Check parameters (skip self/cls)
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation is None:
                missing.append(f"param '{arg.arg}'")

        # Check *args and **kwargs
        if node.args.vararg and node.args.vararg.annotation is None:
            missing.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg and node.args.kwarg.annotation is None:
            missing.append(f"**{node.args.kwarg.arg}")

        return missing

    def audit_directory(
        self,
        directory: Path,
        exclude_patterns: list[str] | None = None,
    ) -> AuditResult:
        """Audit all Python files in a directory.

        Args:
            directory: Directory to audit.
            exclude_patterns: Glob patterns to exclude.

        Returns:
            AuditResult with all findings.
        """
        result = AuditResult()
        exclude_patterns = exclude_patterns or ["**/test_*.py", "**/__pycache__/**"]

        # Find all Python files
        python_files = list(directory.rglob("*.py"))

        # Filter excluded files
        def should_exclude(path: Path) -> bool:
            return any(path.match(pattern) for pattern in exclude_patterns)

        python_files = [f for f in python_files if not should_exclude(f)]

        for filepath in python_files:
            try:
                violations = self.audit_file(filepath)
                result.violations.extend(violations)
                result.files_checked += 1
            except Exception as e:
                result.errors.append(f"{filepath}: {e}")

        return result


def run_audit(
    path: str,
    output_format: str = "text",
    check_line_length: bool = True,
    check_docstrings: bool = True,
    check_type_hints: bool = True,
    max_lines: int = 50,
    strict: bool = False,
    exclude: list[str] | None = None,
) -> int:
    """Run the audit command.

    Args:
        path: File or directory to audit.
        output_format: Output format ("text" or "json").
        check_line_length: Enable Rule of 50 check.
        check_docstrings: Enable docstring check.
        check_type_hints: Enable type hints check.
        max_lines: Maximum function lines.
        strict: Make all violations errors.
        exclude: Patterns to exclude.

    Returns:
        Exit code (0 = success, 1 = violations found).
    """
    target = Path(path)

    if not target.exists():
        console.print(f"[red]Error:[/] Path not found: {path}")
        return 1

    auditor = CodeAuditor(
        check_line_length=check_line_length,
        check_docstrings=check_docstrings,
        check_type_hints=check_type_hints,
        max_function_lines=max_lines,
        strict=strict,
    )

    if target.is_file():
        violations = auditor.audit_file(target)
        result = AuditResult(files_checked=1, violations=violations)
    else:
        result = auditor.audit_directory(target, exclude_patterns=exclude)

    # Output results
    if output_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_text_report(result)

    return 0 if result.passed else 1


def _print_text_report(result: AuditResult) -> None:
    """Print a human-readable audit report."""
    console.print()
    console.print("[bold]AgentBible Code Audit[/]")
    console.print(f"Files checked: {result.files_checked}")
    console.print()

    if not result.violations:
        console.print("[green]All checks passed![/]")
        return

    # Group violations by file
    by_file: dict[str, list[Violation]] = {}
    for v in result.violations:
        if v.file not in by_file:
            by_file[v.file] = []
        by_file[v.file].append(v)

    # Print violations table
    table = Table(title="Violations")
    table.add_column("File", style="cyan")
    table.add_column("Line", style="dim")
    table.add_column("Rule", style="yellow")
    table.add_column("Name", style="magenta")
    table.add_column("Message")
    table.add_column("Severity")

    for filepath, violations in by_file.items():
        for v in violations:
            severity_style = "red" if v.severity == "error" else "yellow"
            table.add_row(
                filepath,
                str(v.line),
                v.rule,
                v.name,
                v.message,
                f"[{severity_style}]{v.severity}[/]",
            )

    console.print(table)
    console.print()

    # Summary
    if result.error_count > 0:
        console.print(f"[red]Errors: {result.error_count}[/]")
    if result.warning_count > 0:
        console.print(f"[yellow]Warnings: {result.warning_count}[/]")

    if result.passed:
        console.print("[green]Audit passed (warnings only)[/]")
    else:
        console.print("[red]Audit failed[/]")
