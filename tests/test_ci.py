"""Tests for the bible ci commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agentbible.cli.ci import (
    WorkflowRun,
    _bump_version_files,
    _create_release_impl,
    _run_gh,
    check_gh_available,
    create_release,
    get_failed_run_logs,
    get_repo_info,
    get_workflow_runs,
    run_ci_release,
    run_ci_status,
    run_ci_verify,
    wait_for_runs,
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


class TestGetFailedRunLogs:
    """Tests for get_failed_run_logs."""

    @patch("agentbible.cli.ci._run_gh")
    def test_returns_stdout(self, mock_run: MagicMock) -> None:
        """Returns stdout from gh run view."""
        mock_run.return_value = MagicMock(stdout="Error log content", stderr="")

        logs = get_failed_run_logs(123)

        assert logs == "Error log content"
        mock_run.assert_called_once()

    @patch("agentbible.cli.ci._run_gh")
    def test_returns_stderr_if_no_stdout(self, mock_run: MagicMock) -> None:
        """Returns stderr if stdout is empty."""
        mock_run.return_value = MagicMock(stdout="", stderr="Error from stderr")

        logs = get_failed_run_logs(123)

        assert logs == "Error from stderr"


class TestGetRepoInfo:
    """Tests for get_repo_info."""

    @patch("agentbible.cli.ci._run_gh")
    def test_parses_repo_info(self, mock_run: MagicMock) -> None:
        """Parses repo info from JSON."""
        mock_run.return_value = MagicMock(
            stdout=json.dumps(
                {
                    "name": "my-repo",
                    "url": "https://github.com/owner/my-repo",
                    "owner": {"login": "owner"},
                    "defaultBranchRef": {"name": "main"},
                }
            )
        )

        info = get_repo_info()

        assert info["name"] == "my-repo"
        assert info["url"] == "https://github.com/owner/my-repo"


class TestBumpVersionFiles:
    """Tests for _bump_version_files."""

    def test_bumps_pyproject_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Updates version in pyproject.toml."""
        monkeypatch.chdir(tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('version = "0.1.0"\nname = "test"')

        result = _bump_version_files("0.2.0")

        assert result is True
        content = pyproject.read_text()
        assert 'version = "0.2.0"' in content

    def test_bumps_init_py(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Updates __version__ in __init__.py."""
        monkeypatch.chdir(tmp_path)
        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        init_file = pkg_dir / "__init__.py"
        init_file.write_text('__version__ = "0.1.0"\n__author__ = "Test"')

        result = _bump_version_files("0.2.0")

        assert result is True
        content = init_file.read_text()
        assert '__version__ = "0.2.0"' in content


class TestRunCiRelease:
    """Tests for run_ci_release."""

    @patch("agentbible.cli.ci.check_gh_available")
    def test_gh_not_available(self, mock_check: MagicMock) -> None:
        """Returns error when gh not available."""
        mock_check.return_value = (False, "gh not found")

        exit_code = run_ci_release("0.3.0")

        assert exit_code == 1

    @patch("subprocess.run")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_dirty_working_dir(
        self, mock_check: MagicMock, mock_subprocess: MagicMock
    ) -> None:
        """Returns error when working directory is not clean."""
        mock_check.return_value = (True, "ok")
        mock_subprocess.return_value = MagicMock(
            stdout="M modified_file.py", returncode=0
        )

        exit_code = run_ci_release("0.3.0")

        assert exit_code == 1


class TestRunCiStatusInProgress:
    """Tests for in-progress run handling."""

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_in_progress_returns_zero(
        self, mock_check: MagicMock, mock_runs: MagicMock
    ) -> None:
        """Returns 0 when runs are in progress (not a failure)."""
        mock_check.return_value = (True, "ok")
        mock_runs.return_value = [
            WorkflowRun(
                id=1,
                name="CI",
                status="in_progress",
                conclusion=None,
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
    def test_empty_runs(self, mock_check: MagicMock, mock_runs: MagicMock) -> None:
        """Returns 0 when no runs exist."""
        mock_check.return_value = (True, "ok")
        mock_runs.return_value = []

        exit_code = run_ci_status()

        assert exit_code == 0


class TestRunGh:
    """Tests for _run_gh helper."""

    @patch("subprocess.run")
    def test_gh_not_found(self, mock_run: MagicMock) -> None:
        """Raises RuntimeError when gh not installed."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(RuntimeError, match="not found"):
            _run_gh(["auth", "status"])

    @patch("subprocess.run")
    def test_not_authenticated_error(self, mock_run: MagicMock) -> None:
        """Raises RuntimeError when not authenticated."""
        mock_run.return_value = MagicMock(
            returncode=1, stderr="gh auth login required", stdout=""
        )

        with pytest.raises(RuntimeError, match="not authenticated"):
            _run_gh(["run", "list"])

    @patch("subprocess.run")
    def test_not_git_repo_error(self, mock_run: MagicMock) -> None:
        """Raises RuntimeError when not in git repo."""
        mock_run.return_value = MagicMock(
            returncode=1, stderr="not a git repository", stdout=""
        )

        with pytest.raises(RuntimeError, match="git repository"):
            _run_gh(["repo", "view"])

    @patch("subprocess.run")
    def test_general_gh_error(self, mock_run: MagicMock) -> None:
        """Raises RuntimeError on general gh error."""
        mock_run.return_value = MagicMock(
            returncode=1, stderr="some other error", stdout=""
        )

        with pytest.raises(RuntimeError, match="gh command failed"):
            _run_gh(["run", "list"])

    @patch("subprocess.run")
    def test_no_check_returns_result(self, mock_run: MagicMock) -> None:
        """Returns result without raising when check=False."""
        mock_run.return_value = MagicMock(returncode=1, stderr="error", stdout="")

        result = _run_gh(["run", "list"], check=False)

        assert result.returncode == 1

    @patch("subprocess.run")
    def test_success_returns_result(self, mock_run: MagicMock) -> None:
        """Returns result on success."""
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")

        result = _run_gh(["run", "list"])

        assert result.returncode == 0
        assert result.stdout == "output"


class TestGetWorkflowRunsBranch:
    """Tests for get_workflow_runs with branch filter."""

    @patch("agentbible.cli.ci._run_gh")
    def test_branch_filter(self, mock_run: MagicMock) -> None:
        """Passes branch filter to gh command."""
        mock_run.return_value = MagicMock(stdout="[]")

        get_workflow_runs(limit=5, branch="feature-branch")

        call_args = mock_run.call_args[0][0]
        assert "--branch" in call_args
        assert "feature-branch" in call_args


class TestCreateRelease:
    """Tests for create_release function."""

    @patch("agentbible.cli.ci._run_gh")
    def test_creates_release_with_defaults(self, mock_run: MagicMock) -> None:
        """Creates release with default title and auto notes."""
        mock_run.return_value = MagicMock(
            stdout="https://github.com/owner/repo/releases/v1.0.0\n"
        )

        url = create_release("v1.0.0")

        assert "github.com" in url
        call_args = mock_run.call_args[0][0]
        assert "release" in call_args
        assert "create" in call_args
        assert "v1.0.0" in call_args
        assert "--generate-notes" in call_args

    @patch("agentbible.cli.ci._run_gh")
    def test_creates_release_with_title(self, mock_run: MagicMock) -> None:
        """Creates release with custom title."""
        mock_run.return_value = MagicMock(stdout="https://github.com/releases/v1.0.0\n")

        create_release("v1.0.0", title="Custom Title")

        call_args = mock_run.call_args[0][0]
        assert "--title" in call_args
        idx = call_args.index("--title")
        assert call_args[idx + 1] == "Custom Title"

    @patch("agentbible.cli.ci._run_gh")
    def test_creates_release_with_notes(self, mock_run: MagicMock) -> None:
        """Creates release with custom notes."""
        mock_run.return_value = MagicMock(stdout="https://github.com/releases/v1.0.0\n")

        create_release("v1.0.0", notes="Release notes here")

        call_args = mock_run.call_args[0][0]
        assert "--notes" in call_args
        assert "--generate-notes" not in call_args

    @patch("agentbible.cli.ci._run_gh")
    def test_creates_draft_release(self, mock_run: MagicMock) -> None:
        """Creates draft release."""
        mock_run.return_value = MagicMock(stdout="https://github.com/releases/v1.0.0\n")

        create_release("v1.0.0", draft=True)

        call_args = mock_run.call_args[0][0]
        assert "--draft" in call_args

    @patch("agentbible.cli.ci._run_gh")
    def test_creates_prerelease(self, mock_run: MagicMock) -> None:
        """Creates prerelease."""
        mock_run.return_value = MagicMock(stdout="https://github.com/releases/v1.0.0\n")

        create_release("v1.0.0", prerelease=True)

        call_args = mock_run.call_args[0][0]
        assert "--prerelease" in call_args


class TestWaitForRuns:
    """Tests for wait_for_runs function."""

    @patch("time.sleep")
    @patch("agentbible.cli.ci.get_workflow_runs")
    def test_returns_immediately_when_all_complete(
        self, mock_runs: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Returns True immediately when no runs in progress."""
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

        passed, runs = wait_for_runs(timeout_seconds=60, poll_interval=1)

        assert passed
        assert len(runs) == 1
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    @patch("agentbible.cli.ci.get_workflow_runs")
    def test_returns_empty_when_no_runs(
        self, mock_runs: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """Returns True when no runs exist."""
        mock_runs.return_value = []

        passed, result_runs = wait_for_runs()

        assert passed
        assert result_runs == []

    @patch("time.sleep")
    @patch("agentbible.cli.ci.get_workflow_runs")
    def test_waits_for_in_progress_runs(
        self, mock_runs: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Polls until in-progress runs complete."""
        in_progress_run = WorkflowRun(
            id=1,
            name="CI",
            status="in_progress",
            conclusion=None,
            branch="main",
            event="push",
            created_at="2024-01-01T00:00:00Z",
            url="https://example.com",
        )
        completed_run = WorkflowRun(
            id=1,
            name="CI",
            status="completed",
            conclusion="success",
            branch="main",
            event="push",
            created_at="2024-01-01T00:00:00Z",
            url="https://example.com",
        )
        mock_runs.side_effect = [[in_progress_run], [completed_run]]

        passed, _result_runs = wait_for_runs(timeout_seconds=60, poll_interval=1)

        assert passed
        assert mock_sleep.call_count == 1

    @patch("time.sleep")
    @patch("agentbible.cli.ci.get_workflow_runs")
    def test_returns_false_on_failure(
        self, mock_runs: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """Returns False when runs fail."""
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

        passed, runs = wait_for_runs()

        assert not passed
        assert len(runs) == 1

    @patch("time.time")
    @patch("time.sleep")
    @patch("agentbible.cli.ci.get_workflow_runs")
    def test_timeout(
        self, mock_runs: MagicMock, _mock_sleep: MagicMock, mock_time: MagicMock
    ) -> None:
        """Returns False on timeout."""
        in_progress_run = WorkflowRun(
            id=1,
            name="CI",
            status="in_progress",
            conclusion=None,
            branch="main",
            event="push",
            created_at="2024-01-01T00:00:00Z",
            url="https://example.com",
        )
        mock_runs.return_value = [in_progress_run]
        # Simulate timeout: first call is 0, second is past timeout
        mock_time.side_effect = [0, 0, 1000]

        passed, _result_runs = wait_for_runs(timeout_seconds=60, poll_interval=1)

        assert not passed

    @patch("time.sleep")
    @patch("agentbible.cli.ci.get_workflow_runs")
    def test_branch_filter_passed(
        self, mock_runs: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """Branch filter is passed to get_workflow_runs."""
        mock_runs.return_value = []

        wait_for_runs(branch="feature")

        mock_runs.assert_called_with(limit=5, branch="feature")


class TestRunCiStatusError:
    """Tests for run_ci_status error handling."""

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_runtime_error(self, mock_check: MagicMock, mock_runs: MagicMock) -> None:
        """Returns 1 when get_workflow_runs raises."""
        mock_check.return_value = (True, "ok")
        mock_runs.side_effect = RuntimeError("API error")

        exit_code = run_ci_status()

        assert exit_code == 1


class TestRunCiVerifyWait:
    """Tests for run_ci_verify with wait option."""

    @patch("agentbible.cli.ci.wait_for_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_wait_passes(self, mock_check: MagicMock, mock_wait: MagicMock) -> None:
        """Returns 0 when wait succeeds."""
        mock_check.return_value = (True, "ok")
        mock_wait.return_value = (True, [])

        exit_code = run_ci_verify(wait=True)

        assert exit_code == 0
        mock_wait.assert_called_once()

    @patch("agentbible.cli.ci.wait_for_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_wait_fails(self, mock_check: MagicMock, mock_wait: MagicMock) -> None:
        """Returns 1 when wait fails."""
        mock_check.return_value = (True, "ok")
        failed_run = WorkflowRun(
            id=1,
            name="CI",
            status="completed",
            conclusion="failure",
            branch="main",
            event="push",
            created_at="2024-01-01T00:00:00Z",
            url="https://example.com",
        )
        mock_wait.return_value = (False, [failed_run])

        exit_code = run_ci_verify(wait=True)

        assert exit_code == 1

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_in_progress_warning(
        self, mock_check: MagicMock, mock_runs: MagicMock
    ) -> None:
        """Shows warning when runs in progress without wait."""
        mock_check.return_value = (True, "ok")
        mock_runs.return_value = [
            WorkflowRun(
                id=1,
                name="CI",
                status="in_progress",
                conclusion=None,
                branch="main",
                event="push",
                created_at="2024-01-01T00:00:00Z",
                url="https://example.com",
            )
        ]

        # Should still return 0 (in progress is not failure)
        exit_code = run_ci_verify()

        assert exit_code == 0

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_runtime_error(self, mock_check: MagicMock, mock_runs: MagicMock) -> None:
        """Returns 1 when get_workflow_runs raises."""
        mock_check.return_value = (True, "ok")
        mock_runs.side_effect = RuntimeError("API error")

        exit_code = run_ci_verify()

        assert exit_code == 1


class TestRunCiReleaseFlow:
    """Tests for run_ci_release full flow."""

    @patch("subprocess.run")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_version_normalization_with_v(
        self, mock_check: MagicMock, mock_subprocess: MagicMock
    ) -> None:
        """Version starting with v is handled correctly."""
        mock_check.return_value = (True, "ok")
        # Clean working dir
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)

        # Will fail at tag step, but we're testing normalization
        run_ci_release(
            "v1.0.0", bump_files=False, push=False, verify=False, create_release=False
        )

        # Check that tag command used v1.0.0
        tag_calls = [c for c in mock_subprocess.call_args_list if "tag" in str(c)]
        assert len(tag_calls) >= 1

    @patch("subprocess.run")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_bump_files_commits(
        self,
        mock_check: MagicMock,
        mock_subprocess: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bumps version files and commits."""
        monkeypatch.chdir(tmp_path)
        mock_check.return_value = (True, "ok")
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)

        # Create pyproject.toml
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('version = "0.1.0"\nname = "test"')

        exit_code = run_ci_release(
            "0.2.0", bump_files=True, push=False, verify=False, create_release=False
        )

        assert exit_code == 0
        assert 'version = "0.2.0"' in pyproject.read_text()

    @patch("subprocess.run")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_tag_creation_failure(
        self, mock_check: MagicMock, mock_subprocess: MagicMock
    ) -> None:
        """Returns 1 when tag creation fails."""
        mock_check.return_value = (True, "ok")

        def subprocess_side_effect(*args, **kwargs):
            cmd = args[0]
            if "status" in cmd:
                return MagicMock(stdout="", returncode=0)
            if "tag" in cmd:
                return MagicMock(stdout="", stderr="tag exists", returncode=1)
            return MagicMock(stdout="", returncode=0)

        mock_subprocess.side_effect = subprocess_side_effect

        exit_code = run_ci_release("1.0.0", bump_files=False, push=True)

        assert exit_code == 1

    @patch("agentbible.cli.ci.wait_for_runs")
    @patch("subprocess.run")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_push_and_wait(
        self, mock_check: MagicMock, mock_subprocess: MagicMock, mock_wait: MagicMock
    ) -> None:
        """Pushes and waits for CI."""
        mock_check.return_value = (True, "ok")
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)
        mock_wait.return_value = (True, [])

        exit_code = run_ci_release(
            "1.0.0", bump_files=False, push=True, verify=True, create_release=False
        )

        assert exit_code == 0
        mock_wait.assert_called_once()

    @patch("agentbible.cli.ci.wait_for_runs")
    @patch("subprocess.run")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_ci_failure_aborts(
        self, mock_check: MagicMock, mock_subprocess: MagicMock, mock_wait: MagicMock
    ) -> None:
        """Aborts when CI fails."""
        mock_check.return_value = (True, "ok")
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)
        failed_run = WorkflowRun(
            id=1,
            name="CI",
            status="completed",
            conclusion="failure",
            branch="main",
            event="push",
            created_at="2024-01-01T00:00:00Z",
            url="https://example.com",
        )
        mock_wait.return_value = (False, [failed_run])

        exit_code = run_ci_release("1.0.0", bump_files=False, push=True, verify=True)

        assert exit_code == 1

    @patch("agentbible.cli.ci._create_release_impl")
    @patch("agentbible.cli.ci.wait_for_runs")
    @patch("subprocess.run")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_creates_github_release(
        self,
        mock_check: MagicMock,
        mock_subprocess: MagicMock,
        mock_wait: MagicMock,
        mock_release: MagicMock,
    ) -> None:
        """Creates GitHub release after CI passes."""
        mock_check.return_value = (True, "ok")
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)
        mock_wait.return_value = (True, [])
        mock_release.return_value = "https://github.com/releases/v1.0.0"

        exit_code = run_ci_release(
            "1.0.0", bump_files=False, push=True, verify=True, create_release=True
        )

        assert exit_code == 0
        mock_release.assert_called_once_with("v1.0.0", draft=False)

    @patch("agentbible.cli.ci._create_release_impl")
    @patch("subprocess.run")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_release_creation_error(
        self,
        mock_check: MagicMock,
        mock_subprocess: MagicMock,
        mock_release: MagicMock,
    ) -> None:
        """Returns 1 when release creation fails."""
        mock_check.return_value = (True, "ok")
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)
        mock_release.side_effect = RuntimeError("Release failed")

        exit_code = run_ci_release(
            "1.0.0", bump_files=False, push=False, verify=False, create_release=True
        )

        assert exit_code == 1

    @patch("subprocess.run")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_push_tag_failure(
        self, mock_check: MagicMock, mock_subprocess: MagicMock
    ) -> None:
        """Returns 1 when tag push fails."""
        mock_check.return_value = (True, "ok")

        call_count = [0]

        def subprocess_side_effect(*args, **kwargs):
            call_count[0] += 1
            cmd = args[0]
            if "status" in cmd:
                return MagicMock(stdout="", returncode=0)
            if "tag" in cmd and "-a" in cmd:
                return MagicMock(stdout="", returncode=0)
            if "push" in cmd and "origin" in cmd:
                if "v1.0.0" in cmd:  # Tag push
                    return MagicMock(stdout="", stderr="failed to push", returncode=1)
                return MagicMock(stdout="", returncode=0)  # Branch push
            return MagicMock(stdout="", returncode=0)

        mock_subprocess.side_effect = subprocess_side_effect

        exit_code = run_ci_release(
            "1.0.0", bump_files=False, push=True, verify=False, create_release=False
        )

        assert exit_code == 1


class TestCreateReleaseImpl:
    """Tests for _create_release_impl function."""

    @patch("agentbible.cli.ci._run_gh")
    def test_creates_release(self, mock_run: MagicMock) -> None:
        """Creates release with gh CLI."""
        mock_run.return_value = MagicMock(
            stdout="https://github.com/owner/repo/releases/v1.0.0\n"
        )

        url = _create_release_impl("v1.0.0")

        assert url == "https://github.com/owner/repo/releases/v1.0.0"
        call_args = mock_run.call_args[0][0]
        assert "release" in call_args
        assert "create" in call_args
        assert "v1.0.0" in call_args
        assert "--generate-notes" in call_args

    @patch("agentbible.cli.ci._run_gh")
    def test_creates_draft_release(self, mock_run: MagicMock) -> None:
        """Creates draft release."""
        mock_run.return_value = MagicMock(stdout="https://github.com/releases/v1.0.0\n")

        _create_release_impl("v1.0.0", draft=True)

        call_args = mock_run.call_args[0][0]
        assert "--draft" in call_args


class TestRunCiStatusOtherConclusions:
    """Tests for run_ci_status with various conclusions."""

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_cancelled_run(self, mock_check: MagicMock, mock_runs: MagicMock) -> None:
        """Handles cancelled runs."""
        mock_check.return_value = (True, "ok")
        mock_runs.return_value = [
            WorkflowRun(
                id=1,
                name="CI",
                status="completed",
                conclusion="cancelled",
                branch="main",
                event="push",
                created_at="2024-01-01T00:00:00Z",
                url="https://example.com",
            )
        ]

        exit_code = run_ci_status()

        # Cancelled is not a failure, not a pass
        assert exit_code == 0

    @patch("agentbible.cli.ci.get_workflow_runs")
    @patch("agentbible.cli.ci.check_gh_available")
    def test_invalid_timestamp(
        self, mock_check: MagicMock, mock_runs: MagicMock
    ) -> None:
        """Handles invalid timestamp gracefully."""
        mock_check.return_value = (True, "ok")
        mock_runs.return_value = [
            WorkflowRun(
                id=1,
                name="CI",
                status="completed",
                conclusion="success",
                branch="main",
                event="push",
                created_at="invalid-timestamp",
                url="https://example.com",
            )
        ]

        exit_code = run_ci_status()

        assert exit_code == 0
