"""Implementation of the bible retrofit command.

Adds AgentBible structure to existing projects without recreating them.
Detects project type and merges configuration intelligently.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.prompt import Confirm

if TYPE_CHECKING:
    pass

console = Console()


def detect_project_type(path: Path) -> str:
    """Detect if project is Python, C++, or mixed.

    Args:
        path: Project root directory.

    Returns:
        One of "python", "cpp", "mixed", or "unknown".
    """
    has_python = (
        (path / "pyproject.toml").exists()
        or (path / "setup.py").exists()
        or (path / "requirements.txt").exists()
        or list(path.glob("**/*.py"))
    )
    has_cpp = (
        (path / "CMakeLists.txt").exists()
        or list(path.glob("**/*.cpp"))
        or list(path.glob("**/*.hpp"))
    )

    if has_python and has_cpp:
        return "mixed"
    elif has_python:
        return "python"
    elif has_cpp:
        return "cpp"
    else:
        return "unknown"


def add_cursorrules(path: Path, force: bool = False) -> bool:
    """Add .cursorrules file to project.

    Args:
        path: Project root directory.
        force: If True, overwrite existing file.

    Returns:
        True if file was created/updated, False otherwise.
    """
    cursorrules_path = path / ".cursorrules"

    if cursorrules_path.exists() and not force:
        console.print(f"[yellow]Skipping:[/] .cursorrules already exists")
        return False

    # Get the template from agentbible
    from agentbible.templates import AVAILABLE_TEMPLATES

    template_path = AVAILABLE_TEMPLATES.get("python-scientific")
    if template_path is None:
        console.print("[red]Error:[/] Could not find template")
        return False

    source_cursorrules = template_path / ".cursorrules"
    if not source_cursorrules.exists():
        console.print("[red]Error:[/] Template .cursorrules not found")
        return False

    shutil.copy2(source_cursorrules, cursorrules_path)
    console.print(f"[green]Added:[/] .cursorrules")
    return True


def add_precommit_config(path: Path, force: bool = False) -> bool:
    """Add or update .pre-commit-config.yaml with AgentBible hooks.

    Args:
        path: Project root directory.
        force: If True, overwrite existing file.

    Returns:
        True if file was created/updated, False otherwise.
    """
    precommit_path = path / ".pre-commit-config.yaml"

    agentbible_hook = """
  # AgentBible - Code quality checks
  - repo: https://github.com/rylanmalarchick/research-code-principles
    rev: v0.5.0  # Update to latest version
    hooks:
      - id: agentbible-audit
        args: [--strict, --max-lines, "50"]
"""

    if precommit_path.exists():
        content = precommit_path.read_text()
        if "agentbible-audit" in content:
            console.print("[yellow]Skipping:[/] .pre-commit-config.yaml already has AgentBible hook")
            return False

        # Append hook to existing file
        with open(precommit_path, "a") as f:
            f.write(agentbible_hook)
        console.print("[green]Updated:[/] .pre-commit-config.yaml (added AgentBible hook)")
        return True
    else:
        # Create new file with basic config
        basic_config = """# Pre-commit configuration
# Install: pip install pre-commit && pre-commit install
# Run manually: pre-commit run --all-files

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-added-large-files
"""
        precommit_path.write_text(basic_config + agentbible_hook)
        console.print("[green]Added:[/] .pre-commit-config.yaml")
        return True


def add_validation_helpers(path: Path, project_type: str) -> bool:
    """Add validation.py helper module.

    Args:
        path: Project root directory.
        project_type: Type of project ("python", "cpp", "mixed").

    Returns:
        True if file was created, False otherwise.
    """
    if project_type not in ("python", "mixed"):
        return False

    # Find or create src directory
    src_paths = [path / "src", path / "lib", path]
    src_path = None
    for sp in src_paths:
        if sp.exists() and sp.is_dir():
            src_path = sp
            break

    if src_path is None:
        src_path = path / "src"
        src_path.mkdir(exist_ok=True)

    validation_path = src_path / "validation.py"
    if validation_path.exists():
        console.print(f"[yellow]Skipping:[/] {validation_path.relative_to(path)} already exists")
        return False

    content = '''"""Validation helpers using AgentBible.

This module re-exports commonly used validators for convenience.
Import from here instead of agentbible directly for project consistency.
"""

from __future__ import annotations

# Core validators
from agentbible import (
    validate_finite,
    validate_positive,
    validate_non_negative,
    validate_range,
    validate_probability,
    validate_probabilities,
    validate_normalized,
)

# Direct check functions for data pipelines
from agentbible import (
    check_finite,
    check_positive,
    check_non_negative,
    check_range,
    check_probability,
    check_probabilities,
    check_normalized,
)

__all__ = [
    # Decorators
    "validate_finite",
    "validate_positive",
    "validate_non_negative",
    "validate_range",
    "validate_probability",
    "validate_probabilities",
    "validate_normalized",
    # Check functions
    "check_finite",
    "check_positive",
    "check_non_negative",
    "check_range",
    "check_probability",
    "check_probabilities",
    "check_normalized",
]
'''

    validation_path.write_text(content)
    console.print(f"[green]Added:[/] {validation_path.relative_to(path)}")
    return True


def add_test_conftest(path: Path) -> bool:
    """Add tests/conftest.py with AgentBible fixtures.

    Args:
        path: Project root directory.

    Returns:
        True if file was created, False otherwise.
    """
    tests_path = path / "tests"
    tests_path.mkdir(exist_ok=True)

    conftest_path = tests_path / "conftest.py"
    if conftest_path.exists():
        # Check if it already has agentbible fixtures
        content = conftest_path.read_text()
        if "agentbible" in content.lower() or "deterministic_seed" in content:
            console.print("[yellow]Skipping:[/] tests/conftest.py already has fixtures")
            return False

        # Append fixtures
        with open(conftest_path, "a") as f:
            f.write("\n\n# AgentBible fixtures\n")
            f.write("from agentbible.testing import deterministic_seed, tolerance, quantum_tolerance\n")
            f.write("\n# Re-export fixtures for pytest discovery\n")
            f.write("__all__ = ['deterministic_seed', 'tolerance', 'quantum_tolerance']\n")
        console.print("[green]Updated:[/] tests/conftest.py (added AgentBible fixtures)")
        return True

    # Create new conftest
    content = '''"""Pytest configuration and fixtures.

AgentBible fixtures for reproducible, physics-aware testing.
"""

import pytest

# Import AgentBible fixtures
from agentbible.testing import deterministic_seed, tolerance, quantum_tolerance

# Re-export for pytest to discover
__all__ = ['deterministic_seed', 'tolerance', 'quantum_tolerance']


# Add any project-specific fixtures below
'''
    conftest_path.write_text(content)
    console.print("[green]Added:[/] tests/conftest.py")
    return True


def add_agent_docs(path: Path) -> bool:
    """Create agent_docs directory with templates.

    Args:
        path: Project root directory.

    Returns:
        True if directory was created, False otherwise.
    """
    agent_docs = path / "agent_docs"

    if agent_docs.exists():
        console.print("[yellow]Skipping:[/] agent_docs/ already exists")
        return False

    agent_docs.mkdir()

    # Create PROJECT_CONTEXT.md template
    project_context = agent_docs / "PROJECT_CONTEXT.md"
    project_context.write_text('''# Project Context

## Overview

**Name:** TODO
**Purpose:** TODO
**Status:** TODO

## Key Components

TODO: Describe main modules and their responsibilities

## Dependencies

TODO: List key dependencies

## Development Notes

TODO: Important context for AI agents working on this project
''')

    console.print("[green]Added:[/] agent_docs/ directory with templates")
    return True


def run_retrofit(
    path: str | None = None,
    cursorrules: bool = False,
    precommit: bool = False,
    validators: bool = False,
    conftest: bool = False,
    agent_docs: bool = False,
    all_components: bool = False,
    interactive: bool = True,
    force: bool = False,
) -> int:
    """Execute the retrofit command.

    Args:
        path: Project root path (defaults to current directory).
        cursorrules: Add .cursorrules file.
        precommit: Add pre-commit configuration.
        validators: Add validation.py helper.
        conftest: Add tests/conftest.py with fixtures.
        agent_docs: Add agent_docs/ directory.
        all_components: Add all components.
        interactive: If True, prompt for each component.
        force: If True, overwrite existing files.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    project_path = Path(path) if path else Path.cwd()

    if not project_path.exists():
        console.print(f"[red]Error:[/] Path does not exist: {project_path}")
        return 1

    # Detect project type
    project_type = detect_project_type(project_path)
    console.print(f"[bold blue]Detected project type:[/] {project_type}")

    if project_type == "unknown":
        console.print("[yellow]Warning:[/] Could not detect project type")
        if interactive and not Confirm.ask("Continue anyway?"):
            return 0

    # Determine what to add
    add_all = all_components or (
        not cursorrules
        and not precommit
        and not validators
        and not conftest
        and not agent_docs
    )

    components_to_add = []

    if add_all and interactive:
        console.print()
        console.print("[bold]Select components to add:[/]")
        if Confirm.ask("  Add .cursorrules (AI agent coding rules)?", default=True):
            components_to_add.append("cursorrules")
        if Confirm.ask("  Add .pre-commit-config.yaml (code quality hooks)?", default=True):
            components_to_add.append("precommit")
        if Confirm.ask("  Add src/validation.py (validation helpers)?", default=True):
            components_to_add.append("validators")
        if Confirm.ask("  Add tests/conftest.py (pytest fixtures)?", default=True):
            components_to_add.append("conftest")
        if Confirm.ask("  Add agent_docs/ (AI context documents)?", default=True):
            components_to_add.append("agent_docs")
    else:
        if add_all or cursorrules:
            components_to_add.append("cursorrules")
        if add_all or precommit:
            components_to_add.append("precommit")
        if add_all or validators:
            components_to_add.append("validators")
        if add_all or conftest:
            components_to_add.append("conftest")
        if add_all or agent_docs:
            components_to_add.append("agent_docs")

    if not components_to_add:
        console.print("[yellow]No components selected. Nothing to do.[/]")
        return 0

    console.print()
    console.print("[bold]Adding AgentBible components...[/]")

    added_count = 0

    if "cursorrules" in components_to_add:
        if add_cursorrules(project_path, force):
            added_count += 1

    if "precommit" in components_to_add:
        if add_precommit_config(project_path, force):
            added_count += 1

    if "validators" in components_to_add:
        if add_validation_helpers(project_path, project_type):
            added_count += 1

    if "conftest" in components_to_add:
        if add_test_conftest(project_path):
            added_count += 1

    if "agent_docs" in components_to_add:
        if add_agent_docs(project_path):
            added_count += 1

    console.print()
    if added_count > 0:
        console.print(f"[bold green]Done![/] Added {added_count} component(s)")
        console.print()
        console.print("[bold]Next steps:[/]")
        console.print("  1. Review the added files and customize as needed")
        if "precommit" in components_to_add:
            console.print("  2. Run: pip install pre-commit && pre-commit install")
        console.print("  3. Commit your changes")
    else:
        console.print("[yellow]No new components added (already exist).[/]")

    return 0
