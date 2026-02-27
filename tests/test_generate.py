"""Tests for bible generate-agents-md command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from agentbible.cli.context_audit import ContextAuditor
from agentbible.cli.generate import build_agents_md, run_generate_agents_md
from agentbible.cli.main import cli


class TestBuildAgentsMd:
    """Tests for build_agents_md()."""

    def test_default_output_under_20_lines(self) -> None:
        """Default output is <= 20 lines."""
        content = build_agents_md()
        lines = content.splitlines()
        assert len(lines) <= 20

    def test_includes_test_command(self) -> None:
        """Default test command is included."""
        content = build_agents_md()
        assert "pytest -x" in content

    def test_includes_lint_command(self) -> None:
        """Lint command is included."""
        content = build_agents_md()
        assert "ruff check" in content

    def test_includes_type_command(self) -> None:
        """Type check command is included."""
        content = build_agents_md()
        assert "mypy" in content

    def test_includes_coverage(self) -> None:
        """Coverage threshold is included."""
        content = build_agents_md(coverage=80)
        assert "80" in content

    def test_custom_coverage_threshold(self) -> None:
        """Custom coverage threshold is used."""
        content = build_agents_md(coverage=90)
        assert "90" in content
        assert "80" not in content

    def test_quantum_domain_includes_validator(self) -> None:
        """Quantum domain includes validate_unitary reference."""
        content = build_agents_md(domain="quantum")
        assert "validate_unitary" in content

    def test_ml_domain_includes_seeds(self) -> None:
        """ML domain includes reproducibility note."""
        content = build_agents_md(domain="ml")
        assert "seed" in content.lower()

    def test_atmospheric_domain_included(self) -> None:
        """Atmospheric domain includes physics constraint."""
        content = build_agents_md(domain="atmospheric")
        assert "validate_bounds" in content or "physical" in content.lower()

    def test_no_domain_no_physics_section(self) -> None:
        """Domain=none produces no Physics section."""
        content = build_agents_md(domain="none")
        assert "validate_unitary" not in content
        assert "## Physics" not in content

    def test_custom_test_command(self) -> None:
        """Custom test command is used."""
        content = build_agents_md(test_cmd="python -m pytest -v")
        assert "python -m pytest -v" in content

    def test_quantum_domain_still_under_20_lines(self) -> None:
        """Quantum domain output is still <= 20 lines."""
        content = build_agents_md(domain="quantum")
        lines = content.splitlines()
        assert len(lines) <= 20

    def test_output_ends_with_newline(self) -> None:
        """Output ends with a newline character."""
        content = build_agents_md()
        assert content.endswith("\n")


class TestGeneratedFilePassesAudit:
    """Generated AGENTS.md must pass its own audit context."""

    def test_default_passes_audit(self, tmp_path: Path) -> None:
        """Default generated file achieves tightness score >= 70."""
        content = build_agents_md()
        f = tmp_path / "AGENTS.md"
        f.write_text(content)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)
        assert result.passed, (
            f"Generated AGENTS.md failed audit with score {result.tightness_score}. "
            f"Issues: {[i.message for i in result.issues]}"
        )

    def test_quantum_domain_passes_audit(self, tmp_path: Path) -> None:
        """Quantum domain generated file passes audit."""
        content = build_agents_md(domain="quantum")
        f = tmp_path / "AGENTS.md"
        f.write_text(content)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)
        assert result.passed

    def test_ml_domain_passes_audit(self, tmp_path: Path) -> None:
        """ML domain generated file passes audit."""
        content = build_agents_md(domain="ml")
        f = tmp_path / "AGENTS.md"
        f.write_text(content)

        auditor = ContextAuditor()
        result = auditor.audit_file(f)
        assert result.passed


class TestRunGenerateAgentsMd:
    """Tests for run_generate_agents_md()."""

    def test_writes_agents_md_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writes AGENTS.md in current directory."""
        monkeypatch.chdir(tmp_path)
        exit_code = run_generate_agents_md()
        assert exit_code == 0
        assert (tmp_path / "AGENTS.md").exists()

    def test_stdout_mode_prints_not_writes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """--stdout prints content, does not write file."""
        monkeypatch.chdir(tmp_path)
        exit_code = run_generate_agents_md(stdout=True)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "pytest" in captured.out
        assert not (tmp_path / "AGENTS.md").exists()

    def test_domain_flag_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Domain flag is applied to written file."""
        monkeypatch.chdir(tmp_path)
        run_generate_agents_md(domain="quantum")
        content = (tmp_path / "AGENTS.md").read_text()
        assert "validate_unitary" in content


class TestCLIGenerateAgentsMd:
    """Tests for generate-agents-md CLI command."""

    def test_generate_help(self) -> None:
        """generate-agents-md --help shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate-agents-md", "--help"])

        assert result.exit_code == 0
        assert "--domain" in result.output
        assert "--coverage" in result.output
        assert "--stdout" in result.output

    def test_generate_default(self, tmp_path: Path) -> None:
        """generate-agents-md creates AGENTS.md file."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["generate-agents-md"])

            assert result.exit_code == 0
            assert Path("AGENTS.md").exists()

    def test_generate_quantum_domain(self, tmp_path: Path) -> None:
        """generate-agents-md --domain quantum includes validator."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["generate-agents-md", "--domain", "quantum"])

            assert result.exit_code == 0
            content = Path("AGENTS.md").read_text()
            assert "validate_unitary" in content

    def test_generate_stdout(self, tmp_path: Path) -> None:
        """generate-agents-md --stdout prints to stdout."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["generate-agents-md", "--stdout"])

            assert result.exit_code == 0
            assert "pytest" in result.output
            assert not Path("AGENTS.md").exists()

    def test_generate_custom_coverage(self, tmp_path: Path) -> None:
        """generate-agents-md --coverage 90 uses 90%."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["generate-agents-md", "--coverage", "90"])

            assert result.exit_code == 0
            content = Path("AGENTS.md").read_text()
            assert "90" in content

    def test_generated_file_passes_audit(self, tmp_path: Path) -> None:
        """Generated file passes bible audit context."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["generate-agents-md"])
            audit_result = runner.invoke(cli, ["audit", "context", "AGENTS.md"])

            assert audit_result.exit_code == 0
