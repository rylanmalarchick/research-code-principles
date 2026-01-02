"""Tests for the bible ci commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agentbible.cli.ci import (
    WorkflowRun,
    check_gh_available,
    get_workflow_runs,
    run_ci_status,
    run_ci_verify,
)
from agentbible.cli.main import cli


class TestWorkflowRun:
    """Tests for WorkflowRun dataclass."""

    def test_passed(self) -> None:
        """Completed successful run is passed."""
        run = WorkflowRun(
            id=1,
            name="CI",
            status="completed",
            conclusion="success",
            branch="main",
            event="push",
            created_at="2024-01-01T00:00:00Z",
            url="https://example.com",
        )
        assert run.passed
        assert not run.failed
        assert not run.in_progress

    def test_failed(self) -> None:
        """Completed failed run is failed."""
        run = WorkflowRun(
            id=1,
            name="CI",
            status="completed",
            conclusion="failure",
            branch="main",
            event="push",
            created_at="2024-01-01T00:00:00Z",
            url="https://example.com",
        )
        assert not run.passed
        assert run.failed
        assert not run.in_progress

    def test_in_progress(self) -> None:
        """In-progress run is in_progress."""
        run = WorkflowRun(
            id=1,
            name="CI",
            status="in_progress",
            conclusion=None,
            branch="main",
            event="push",
            created_at="2024-01-01T00:00:00Z",
            url="https://example.com",
        )
        assert not run.passed
        assert not run.failed
        assert run.in_progress


class TestCheckGhAvailable:
    """Tests for check_gh_available."""

    @patch("agentbible.cli.ci._run_gh")
    def test_authenticated(self, mock_run: MagicMock) -> None:
        """Returns True when authenticated."""
        mock_run.return_value = MagicMock(returncode=0)

        available, message = check_gh_available()

        assert available
        assert "authenticated" in message.lower()

    @patch("agentbible.cli.ci._run_gh")
    def test_not_authenticated(self, mock_run: MagicMock) -> None:
        """Returns False when not authenticated."""
        mock_run.return_value = MagicMock(returncode=1)

        available, message = check_gh_available()

        assert not available
        assert "login" in message.lower()

    @patch("agentbible.cli.ci._run_gh")
    def test_gh_not_installed(self, mock_run: MagicMock) -> None:
        """Returns False when gh not installed."""
        mock_run.side_effect = RuntimeError("GitHub CLI (gh) not found")

        available, message = check_gh_available()

        assert not available
        assert "not found" in message.lower()


class TestGetWorkflowRuns:
    """Tests for get_workflow_runs."""

    @patch("agentbible.cli.ci._run_gh")
    def test_parses_runs(self, mock_run: MagicMock) -> None:
        """Parses workflow runs from JSON."""
        mock_run.return_value = MagicMock(
            stdout=json.dumps(
                [
                    {
                        "databaseId": 123,
                        "name": "CI",
                        "status": "completed",
                        "conclusion": "success",
                        "headBranch": "main",
                        "event": "push",
                        "createdAt": "2024-01-01T00:00:00Z",
                        "url": "https://example.com/run/123",
                    },
                    {
                        "databaseId": 124,
                        "name": "Docs",
                        "status": "completed",
                        "conclusion": "failure",
                        "headBranch": "main",
                        "event": "push",
                        "createdAt": "2024-01-01T00:01:00Z",
                        "url": "https://example.com/run/124",
                    },
                ]
            )
        )

        runs = get_workflow_runs(limit=10)

        assert len(runs) == 2
        assert runs[0].id == 123
        assert runs[0].name == "CI"
        assert runs[0].passed
        assert runs[1].id == 124
        assert runs[1].failed

    @patch("agentbible.cli.ci._run_gh")
    def test_empty_runs(self, mock_run: MagicMock) -> None:
        """Handles empty run list."""
        mock_run.return_value = MagicMock(stdout="[]")

        runs = get_workflow_runs()

        assert runs == []


class TestRunCiStatus:
    """Tests for run_ci_status."""

    @patch("agentbible.cli.ci.check_gh_available")
    def test_gh_not_available(self, mock_check: MagicMock) -> None:
        """Returns error when gh not available."""
        mock_check.return_value = (False, "gh not found")

        exit_code = run_ci_status()

        assert exit_code == 1

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_all_passing(self, mock_check: MagicMock, mock_runs: MagicMock) -> None:
        """Returns 0 when all runs pass."""
        mock_check.return_value = (True, "ok")
        mock_runs.return_value = [
            WorkflowRun(
                id=1,
                name="CI",
                status="completed",
                conclusion="success",
                branch="main",
                event="push",
                created_at="2024-01-01T00:00:00Z",
                url="https://example.com",
            )
        ]

        exit_code = run_ci_status()

        assert exit_code == 0

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_has_failures(self, mock_check: MagicMock, mock_runs: MagicMock) -> None:
        """Returns 1 when runs have failures."""
        mock_check.return_value = (True, "ok")
        mock_runs.return_value = [
            WorkflowRun(
                id=1,
                name="CI",
                status="completed",
                conclusion="failure",
                branch="main",
                event="push",
                created_at="2024-01-01T00:00:00Z",
                url="https://example.com",
            )
        ]

        exit_code = run_ci_status()

        assert exit_code == 1

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_json_output(
        self, mock_check: MagicMock, mock_runs: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """JSON output is valid JSON."""
        mock_check.return_value = (True, "ok")
        mock_runs.return_value = [
            WorkflowRun(
                id=1,
                name="CI",
                status="completed",
                conclusion="success",
                branch="main",
                event="push",
                created_at="2024-01-01T00:00:00Z",
                url="https://example.com",
            )
        ]

        exit_code = run_ci_status(output_json=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert exit_code == 0
        assert data["all_passing"]
        assert not data["has_failures"]


class TestRunCiVerify:
    """Tests for run_ci_verify."""

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_verify_passes(self, mock_check: MagicMock, mock_runs: MagicMock) -> None:
        """Returns 0 when verification passes."""
        mock_check.return_value = (True, "ok")
        mock_runs.return_value = [
            WorkflowRun(
                id=1,
                name="CI",
                status="completed",
                conclusion="success",
                branch="main",
                event="push",
                created_at="2024-01-01T00:00:00Z",
                url="https://example.com",
            )
        ]

        exit_code = run_ci_verify()

        assert exit_code == 0

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_verify_fails(self, mock_check: MagicMock, mock_runs: MagicMock) -> None:
        """Returns 1 when verification fails."""
        mock_check.return_value = (True, "ok")
        mock_runs.return_value = [
            WorkflowRun(
                id=1,
                name="CI",
                status="completed",
                conclusion="failure",
                branch="main",
                event="push",
                created_at="2024-01-01T00:00:00Z",
                url="https://example.com",
            )
        ]

        exit_code = run_ci_verify()

        assert exit_code == 1


class TestCLICi:
    """Tests for ci CLI commands."""

    def test_ci_help(self) -> None:
        """ci --help shows command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ci", "--help"])

        assert result.exit_code == 0
        assert "CI/CD" in result.output
        assert "status" in result.output
        assert "verify" in result.output
        assert "release" in result.output

    def test_ci_status_help(self) -> None:
        """ci status --help shows command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ci", "status", "--help"])

        assert result.exit_code == 0
        assert "workflow" in result.output.lower()
        assert "--json" in result.output

    def test_ci_verify_help(self) -> None:
        """ci verify --help shows command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ci", "verify", "--help"])

        assert result.exit_code == 0
        assert "--wait" in result.output

    def test_ci_release_help(self) -> None:
        """ci release --help shows command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ci", "release", "--help"])

        assert result.exit_code == 0
        assert "VERSION" in result.output
        assert "--draft" in result.output
        assert "--no-bump" in result.output

    @patch("agentbible.cli.ci.check_gh_available")
    def test_ci_status_gh_not_available(self, mock_check: MagicMock) -> None:
        """ci status shows error when gh not available."""
        mock_check.return_value = (False, "gh not found")

        runner = CliRunner()
        result = runner.invoke(cli, ["ci", "status"])

        assert result.exit_code == 1
        assert "error" in result.output.lower()
