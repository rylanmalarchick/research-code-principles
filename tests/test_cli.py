"""Tests for CLI commands."""

from __future__ import annotations

from click.testing import CliRunner

from agentbible import __version__
from agentbible.cli.main import cli


class TestCLI:
    """Tests for the bible CLI."""

    def test_version(self) -> None:
        """--version shows version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output

    def test_help(self) -> None:
        """--help shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "AgentBible" in result.output
        assert "init" in result.output
        assert "context" in result.output
        assert "validate" in result.output

    def test_init_help(self) -> None:
        """init --help shows command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        assert result.exit_code == 0
        assert "Initialize a new project" in result.output
        assert "--template" in result.output

    def test_context_help(self) -> None:
        """context --help shows command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["context", "--help"])

        assert result.exit_code == 0
        assert "Generate AI context" in result.output
        assert "--all" in result.output
        assert "--query" in result.output

    def test_validate_help(self) -> None:
        """validate --help shows command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])

        assert result.exit_code == 0
        assert "Validate physics constraints" in result.output
        assert "--check" in result.output

    def test_info(self) -> None:
        """info command shows installation info."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "AgentBible" in result.output
        assert __version__ in result.output
        assert "validators" in result.output

    # Note: Full init tests are in test_init.py
