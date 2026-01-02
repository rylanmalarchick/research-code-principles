"""CI/CD utilities for AgentBible.

Provides commands for AI agents to interact with GitHub Actions,
verify CI status, and automate releases.

Key principle: Agents should ALWAYS verify CI status after pushing code.
This module makes that easy and explicit.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class WorkflowRun:
    """A GitHub Actions workflow run."""

    id: int
    name: str
    status: str
    conclusion: str | None
    branch: str
    event: str
    created_at: str
    url: str

    @property
    def passed(self) -> bool:
        """True if the run completed successfully."""
        return self.status == "completed" and self.conclusion == "success"

    @property
    def failed(self) -> bool:
        """True if the run failed."""
        return self.status == "completed" and self.conclusion == "failure"

    @property
    def in_progress(self) -> bool:
        """True if the run is still in progress."""
        return self.status in ("in_progress", "queued", "pending")


def _run_gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a gh CLI command.

    Args:
        args: Arguments to pass to gh.
        check: If True, raise on non-zero exit code.

    Returns:
        CompletedProcess with stdout/stderr.

    Raises:
        RuntimeError: If gh is not installed or not authenticated.
    """
    try:
        result = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            if "gh auth login" in result.stderr:
                raise RuntimeError("GitHub CLI not authenticated. Run: gh auth login")
            if "not a git repository" in result.stderr:
                raise RuntimeError(
                    "Not in a git repository. Run from a git repo directory."
                )
            raise RuntimeError(f"gh command failed: {result.stderr}")
        return result
    except FileNotFoundError:
        raise RuntimeError(
            "GitHub CLI (gh) not found. Install from: https://cli.github.com/"
        ) from None


def check_gh_available() -> tuple[bool, str]:
    """Check if gh CLI is available and authenticated.

    Returns:
        Tuple of (available, message).
    """
    try:
        result = _run_gh(["auth", "status"], check=False)
        if result.returncode == 0:
            return True, "GitHub CLI authenticated"
        return False, "GitHub CLI not authenticated. Run: gh auth login"
    except RuntimeError as e:
        return False, str(e)


def get_workflow_runs(limit: int = 10, branch: str | None = None) -> list[WorkflowRun]:
    """Get recent workflow runs.

    Args:
        limit: Maximum number of runs to fetch.
        branch: Filter by branch name (optional).

    Returns:
        List of WorkflowRun objects.
    """
    args = [
        "run",
        "list",
        "--limit",
        str(limit),
        "--json",
        "databaseId,name,status,conclusion,headBranch,event,createdAt,url",
    ]
    if branch:
        args.extend(["--branch", branch])

    result = _run_gh(args)
    runs_data = json.loads(result.stdout)

    return [
        WorkflowRun(
            id=run["databaseId"],
            name=run["name"],
            status=run["status"],
            conclusion=run.get("conclusion"),
            branch=run["headBranch"],
            event=run["event"],
            created_at=run["createdAt"],
            url=run["url"],
        )
        for run in runs_data
    ]


def get_failed_run_logs(run_id: int) -> str:
    """Get logs from a failed workflow run.

    Args:
        run_id: The workflow run ID.

    Returns:
        Log output as string.
    """
    result = _run_gh(["run", "view", str(run_id), "--log-failed"], check=False)
    return result.stdout or result.stderr


def get_repo_info() -> dict[str, Any]:
    """Get current repository information.

    Returns:
        Dict with repo name, URL, default branch, etc.
    """
    result = _run_gh(
        [
            "repo",
            "view",
            "--json",
            "name,url,defaultBranchRef,owner",
        ]
    )
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def create_release(
    version: str,
    title: str | None = None,
    notes: str | None = None,
    draft: bool = False,
    prerelease: bool = False,
) -> str:
    """Create a GitHub release.

    Args:
        version: Version tag (e.g., "v0.3.0").
        title: Release title (defaults to version).
        notes: Release notes.
        draft: Create as draft.
        prerelease: Mark as prerelease.

    Returns:
        URL of created release.
    """
    args = ["release", "create", version]

    if title:
        args.extend(["--title", title])
    else:
        args.extend(["--title", version])

    if notes:
        args.extend(["--notes", notes])
    else:
        args.append("--generate-notes")

    if draft:
        args.append("--draft")
    if prerelease:
        args.append("--prerelease")

    result = _run_gh(args)
    return result.stdout.strip()


def wait_for_runs(
    timeout_seconds: int = 300,
    poll_interval: int = 10,
    branch: str | None = None,
) -> tuple[bool, list[WorkflowRun]]:
    """Wait for in-progress workflow runs to complete.

    Args:
        timeout_seconds: Maximum time to wait.
        poll_interval: Seconds between status checks.
        branch: Filter by branch name.

    Returns:
        Tuple of (all_passed, list of runs).
    """
    import time

    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        runs = get_workflow_runs(limit=5, branch=branch)

        if not runs:
            return True, []

        in_progress = [r for r in runs if r.in_progress]

        if not in_progress:
            # All runs completed
            failed = [r for r in runs if r.failed]
            return len(failed) == 0, runs

        time.sleep(poll_interval)

    # Timeout
    return False, get_workflow_runs(limit=5, branch=branch)


def run_ci_status(
    limit: int = 10,
    branch: str | None = None,
    output_json: bool = False,
) -> int:
    """Show CI/CD status.

    Args:
        limit: Number of runs to show.
        branch: Filter by branch.
        output_json: Output as JSON.

    Returns:
        Exit code (0 = all passing, 1 = failures present).
    """
    # Check gh availability
    available, message = check_gh_available()
    if not available:
        console.print(f"[red]Error:[/] {message}")
        return 1

    try:
        runs = get_workflow_runs(limit=limit, branch=branch)
    except RuntimeError as e:
        console.print(f"[red]Error:[/] {e}")
        return 1

    if output_json:
        data = {
            "runs": [
                {
                    "id": r.id,
                    "name": r.name,
                    "status": r.status,
                    "conclusion": r.conclusion,
                    "branch": r.branch,
                    "passed": r.passed,
                    "url": r.url,
                }
                for r in runs
            ],
            "all_passing": all(r.passed for r in runs if not r.in_progress),
            "has_failures": any(r.failed for r in runs),
            "in_progress": any(r.in_progress for r in runs),
        }
        print(json.dumps(data, indent=2))
        return 0 if not data["has_failures"] else 1

    # Text output
    console.print()
    console.print("[bold]CI/CD Status[/]")
    console.print()

    if not runs:
        console.print("[yellow]No workflow runs found[/]")
        return 0

    table = Table()
    table.add_column("Status", style="bold")
    table.add_column("Workflow")
    table.add_column("Branch")
    table.add_column("Event")
    table.add_column("Time")

    for run in runs:
        if run.passed:
            status = "[green]passed[/]"
        elif run.failed:
            status = "[red]FAILED[/]"
        elif run.in_progress:
            status = "[yellow]running[/]"
        else:
            status = f"[dim]{run.conclusion or run.status}[/]"

        # Parse and format time
        try:
            dt = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, AttributeError):
            time_str = run.created_at

        table.add_row(status, run.name, run.branch, run.event, time_str)

    console.print(table)
    console.print()

    # Summary
    failed = [r for r in runs if r.failed]
    in_progress = [r for r in runs if r.in_progress]

    if failed:
        console.print(f"[red]Failed runs: {len(failed)}[/]")
        console.print()
        console.print("[bold]To view failure logs:[/]")
        for run in failed[:3]:
            console.print(f"  gh run view {run.id} --log-failed")
        return 1

    if in_progress:
        console.print(f"[yellow]In progress: {len(in_progress)}[/]")
        console.print("Run [bold]bible ci wait[/] to wait for completion")
        return 0

    console.print("[green]All checks passing[/]")
    return 0


def run_ci_verify(branch: str | None = None, wait: bool = False) -> int:
    """Verify CI is passing.

    Args:
        branch: Branch to verify.
        wait: Wait for in-progress runs to complete.

    Returns:
        Exit code (0 = passing, 1 = failing).
    """
    available, message = check_gh_available()
    if not available:
        console.print(f"[red]Error:[/] {message}")
        return 1

    console.print("[bold]Verifying CI status...[/]")

    try:
        if wait:
            console.print("Waiting for runs to complete...")
            all_passed, runs = wait_for_runs(branch=branch)
        else:
            runs = get_workflow_runs(limit=10, branch=branch)
            all_passed = all(r.passed for r in runs if not r.in_progress)
            if any(r.in_progress for r in runs):
                console.print("[yellow]Warning:[/] Some runs still in progress")
                console.print("Use [bold]--wait[/] to wait for completion")
    except RuntimeError as e:
        console.print(f"[red]Error:[/] {e}")
        return 1

    if all_passed:
        console.print("[green]CI verification passed[/]")
        return 0
    else:
        console.print("[red]CI verification failed[/]")
        failed = [r for r in runs if r.failed]
        for run in failed[:3]:
            console.print(f"  - {run.name}: {run.url}")
        return 1


def run_ci_release(
    version: str,
    bump_files: bool = True,
    push: bool = True,
    create_release: bool = True,
    verify: bool = True,
    draft: bool = False,
) -> int:
    """Run full release flow.

    Args:
        version: Version to release (e.g., "0.3.0" or "v0.3.0").
        bump_files: Update version in pyproject.toml and __init__.py.
        push: Push commits and tags.
        create_release: Create GitHub release.
        verify: Wait for CI and verify it passes.
        draft: Create release as draft.

    Returns:
        Exit code.
    """
    # Normalize version
    if not version.startswith("v"):
        tag_version = f"v{version}"
        version_number = version
    else:
        tag_version = version
        version_number = version[1:]

    console.print(f"[bold]Releasing {tag_version}[/]")
    console.print()

    # Check gh availability
    available, message = check_gh_available()
    if not available:
        console.print(f"[red]Error:[/] {message}")
        return 1

    # Check for clean working directory
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        console.print("[red]Error:[/] Working directory not clean")
        console.print("Commit or stash changes before releasing")
        return 1

    # Step 1: Bump version in files
    if bump_files:
        console.print("[bold]Step 1:[/] Bumping version...")
        if not _bump_version_files(version_number):
            return 1
        console.print(f"  Updated version to {version_number}")

        # Commit version bump
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"chore: bump version to {version_number}"],
            capture_output=True,
        )
        console.print("  Committed version bump")

    # Step 2: Create and push tag
    console.print(f"[bold]Step 2:[/] Creating tag {tag_version}...")
    result = subprocess.run(
        ["git", "tag", "-a", tag_version, "-m", f"Release {tag_version}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Error creating tag:[/] {result.stderr}")
        return 1
    console.print(f"  Created tag {tag_version}")

    # Step 3: Push
    if push:
        console.print("[bold]Step 3:[/] Pushing to remote...")
        result = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            console.print(f"[yellow]Warning pushing branch:[/] {result.stderr}")

        result = subprocess.run(
            ["git", "push", "origin", tag_version],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            console.print(f"[red]Error pushing tag:[/] {result.stderr}")
            return 1
        console.print("  Pushed to remote")

    # Step 4: Wait for CI
    if verify:
        console.print("[bold]Step 4:[/] Waiting for CI...")
        all_passed, runs = wait_for_runs(timeout_seconds=600, branch=tag_version)

        if not all_passed:
            console.print("[red]CI failed![/]")
            failed = [r for r in runs if r.failed]
            for run in failed:
                console.print(f"  - {run.name}: gh run view {run.id} --log-failed")
            console.print()
            console.print("[yellow]Release aborted. Fix CI and try again.[/]")
            console.print(
                f"To delete the tag: git tag -d {tag_version} && git push origin :{tag_version}"
            )
            return 1
        console.print("  [green]CI passed[/]")

    # Step 5: Create release
    if create_release:
        console.print("[bold]Step 5:[/] Creating GitHub release...")
        try:
            url = _create_release_impl(tag_version, draft=draft)
            console.print(f"  Release created: {url}")
        except RuntimeError as e:
            console.print(f"[red]Error creating release:[/] {e}")
            return 1

    console.print()
    console.print(f"[green bold]Release {tag_version} complete![/]")
    return 0


def _bump_version_files(version: str) -> bool:
    """Update version in project files.

    Args:
        version: New version string (without 'v' prefix).

    Returns:
        True if successful.
    """
    # Find and update pyproject.toml
    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        content = pyproject.read_text()
        # Match version = "x.y.z" pattern
        new_content = re.sub(
            r'version\s*=\s*"[^"]*"',
            f'version = "{version}"',
            content,
            count=1,
        )
        if new_content != content:
            pyproject.write_text(new_content)

    # Find and update __init__.py with __version__
    for init_path in Path().rglob("__init__.py"):
        if "site-packages" in str(init_path) or ".venv" in str(init_path):
            continue
        content = init_path.read_text()
        if "__version__" in content:
            new_content = re.sub(
                r'__version__\s*=\s*"[^"]*"',
                f'__version__ = "{version}"',
                content,
            )
            if new_content != content:
                init_path.write_text(new_content)
                break

    return True


def _create_release_impl(tag: str, draft: bool = False) -> str:
    """Create GitHub release (internal implementation).

    Args:
        tag: Version tag.
        draft: Create as draft.

    Returns:
        Release URL.
    """
    args = [
        "release",
        "create",
        tag,
        "--title",
        tag,
        "--generate-notes",
    ]
    if draft:
        args.append("--draft")

    result = _run_gh(args)
    return result.stdout.strip()
