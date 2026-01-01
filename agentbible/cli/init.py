"""Implementation of the bible init command.

Creates new projects from templates with variable substitution.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from rich.console import Console

from agentbible.templates import AVAILABLE_TEMPLATES

console = Console()

# Template variable pattern: {{VARIABLE_NAME}}
TEMPLATE_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")

# Files to process as templates (apply variable substitution)
TEMPLATE_EXTENSIONS = {".template"}

# Files to skip when copying
SKIP_PATTERNS = {"__pycache__", ".pyc", ".git", ".DS_Store"}


def validate_project_name(name: str) -> tuple[bool, str]:
    """Validate project name and return (is_valid, error_message)."""
    if not name:
        return False, "Project name cannot be empty"

    if " " in name:
        return False, "Project name cannot contain spaces"

    if name.startswith("-") or name.startswith("."):
        return False, "Project name cannot start with - or ."

    # Check for valid characters (alphanumeric, dash, underscore)
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", name):
        return (
            False,
            "Project name must start with a letter and contain only letters, numbers, dashes, and underscores",
        )

    return True, ""


def to_python_identifier(name: str) -> str:
    """Convert project name to valid Python identifier."""
    # Replace dashes with underscores
    identifier = name.replace("-", "_")
    # Convert to lowercase
    identifier = identifier.lower()
    # Remove any remaining invalid characters
    identifier = re.sub(r"[^a-z0-9_]", "", identifier)
    # Ensure it starts with a letter
    if identifier and identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier


def get_git_config(key: str) -> str | None:
    """Get value from git config."""
    try:
        result = subprocess.run(
            ["git", "config", "--get", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_template_variables(
    project_name: str,
    author: str | None,
    email: str | None,
    description: str | None,
) -> dict[str, str]:
    """Build template variables dictionary."""
    # Get defaults from git config
    default_author = get_git_config("user.name") or "Your Name"
    default_email = get_git_config("user.email") or "your.email@example.com"

    return {
        "PROJECT_NAME": project_name,
        "PROJECT_NAME_UNDERSCORE": to_python_identifier(project_name),
        "PROJECT_DESCRIPTION": description or f"A research project: {project_name}",
        "AUTHOR_NAME": author or default_author,
        "AUTHOR_EMAIL": email or default_email,
        "YEAR": str(datetime.now().year),
        "DATE": datetime.now().strftime("%Y-%m-%d"),
    }


def substitute_variables(content: str, variables: dict[str, str]) -> str:
    """Replace template variables in content."""

    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return variables.get(var_name, match.group(0))

    return TEMPLATE_VAR_PATTERN.sub(replace, content)


def should_skip(path: Path) -> bool:
    """Check if path should be skipped during copy."""
    return any(pattern in str(path) for pattern in SKIP_PATTERNS)


def copy_template(
    template_dir: Path,
    dest_dir: Path,
    variables: dict[str, str],
) -> list[Path]:
    """Copy template directory with variable substitution.

    Returns list of created files.
    """
    created_files: list[Path] = []

    for src_path in template_dir.rglob("*"):
        if should_skip(src_path):
            continue

        # Calculate relative path and destination
        rel_path = src_path.relative_to(template_dir)
        dest_path = dest_dir / rel_path

        if src_path.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
        elif src_path.is_file():
            # Ensure parent directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if this is a template file
            if src_path.suffix in TEMPLATE_EXTENSIONS:
                # Read, substitute, and write
                content = src_path.read_text(encoding="utf-8")
                content = substitute_variables(content, variables)

                # Remove .template extension
                final_path = dest_path.with_suffix("")
                final_path.write_text(content, encoding="utf-8")
                created_files.append(final_path)
            else:
                # Copy as-is
                shutil.copy2(src_path, dest_path)
                created_files.append(dest_path)

    return created_files


def init_git(project_dir: Path) -> bool:
    """Initialize git repository in project directory."""
    try:
        result = subprocess.run(
            ["git", "init"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def create_venv(project_dir: Path) -> bool:
    """Create virtual environment in project directory."""
    try:
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "venv", ".venv"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_init(
    name: str,
    template: str,
    author: str | None,
    email: str | None,
    description: str | None,
    no_git: bool,
    no_venv: bool,
    force: bool,
) -> int:
    """Execute the init command. Returns exit code."""
    # Validate project name
    is_valid, error = validate_project_name(name)
    if not is_valid:
        console.print(f"[red]Error:[/] {error}")
        return 1

    # Check template exists
    template_dir = AVAILABLE_TEMPLATES.get(template)
    if template_dir is None:
        console.print(f"[red]Error:[/] Unknown template '{template}'")
        console.print(f"Available templates: {', '.join(AVAILABLE_TEMPLATES.keys())}")
        return 1

    if not template_dir.exists():
        console.print(f"[red]Error:[/] Template directory not found: {template_dir}")
        return 1

    # Check destination
    dest_dir = Path.cwd() / name
    if dest_dir.exists():
        if not force:
            console.print(f"[red]Error:[/] Directory '{name}' already exists")
            console.print("Use --force to overwrite")
            return 1
        else:
            console.print(f"[yellow]Removing existing directory:[/] {name}")
            shutil.rmtree(dest_dir)

    # Build template variables
    variables = get_template_variables(name, author, email, description)

    # Create project
    console.print(
        f"[bold blue]Creating project '{name}' from template '{template}'...[/]"
    )

    try:
        dest_dir.mkdir(parents=True)
        created_files = copy_template(template_dir, dest_dir, variables)
        console.print(f"[green]✓[/] Created {len(created_files)} files")
    except Exception as e:
        console.print(f"[red]Error copying template:[/] {e}")
        return 1

    # Initialize git
    if not no_git:
        console.print("[dim]Initializing git repository...[/]")
        if init_git(dest_dir):
            console.print("[green]✓[/] Git repository initialized")
        else:
            console.print("[yellow]Warning:[/] Failed to initialize git repository")

    # Create virtual environment
    if not no_venv and template.startswith("python"):
        console.print("[dim]Creating virtual environment...[/]")
        if create_venv(dest_dir):
            console.print("[green]✓[/] Virtual environment created at .venv/")
        else:
            console.print("[yellow]Warning:[/] Failed to create virtual environment")

    # Success message
    console.print()
    console.print("[bold green]Project created successfully![/]")
    console.print()
    console.print("[bold]Next steps:[/]")
    console.print(f"  cd {name}")
    if not no_venv and template.startswith("python"):
        console.print("  source .venv/bin/activate")
        console.print('  pip install -e ".[dev]"')
        console.print("  pytest")
    elif template.startswith("cpp"):
        console.print("  mkdir build && cd build")
        console.print("  cmake ..")
        console.print("  make")

    return 0
