"""AgentBible CLI - Command-line interface for research code infrastructure.

Usage:
    bible --help
    bible init my-project --template python-scientific
    bible context --all ./agent_docs
    bible validate state.npy --check unitarity
"""

from __future__ import annotations

import sys

import click
from rich.console import Console

from agentbible import __version__
from agentbible.cli.init import run_init
from agentbible.cli.validate import run_validate

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="agentbible")
def cli() -> None:
    """AgentBible: Production-grade research code infrastructure.

    Tools for AI-assisted scientific software development with
    physics validation, project scaffolding, and context management.
    """
    pass


@cli.command()
@click.argument("name")
@click.option(
    "--template",
    "-t",
    type=click.Choice(["python-scientific", "cpp-hpc-cuda"]),
    default="python-scientific",
    help="Project template to use.",
)
@click.option(
    "--author",
    "-a",
    type=str,
    help="Author name (defaults to git config user.name).",
)
@click.option(
    "--email",
    "-e",
    type=str,
    help="Author email (defaults to git config user.email).",
)
@click.option(
    "--description",
    "-d",
    type=str,
    help="Project description.",
)
@click.option(
    "--no-git",
    is_flag=True,
    help="Skip git initialization.",
)
@click.option(
    "--no-venv",
    is_flag=True,
    help="Skip virtual environment creation.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing directory.",
)
def init(
    name: str,
    template: str,
    author: str | None,
    email: str | None,
    description: str | None,
    no_git: bool,
    no_venv: bool,
    force: bool,
) -> None:
    """Initialize a new project from template.

    Creates a new project directory with the specified template,
    including pre-configured testing, linting, and CI/CD.

    Example:
        bible init my-quantum-sim --template python-scientific
    """
    sys.exit(run_init(name, template, author, email, description, no_git, no_venv, force))


@cli.command()
@click.argument("path", required=False, type=click.Path(exists=True))
@click.option(
    "--all",
    "-a",
    "load_all",
    is_flag=True,
    help="Load all documents from path.",
)
@click.option(
    "--query",
    "-q",
    type=str,
    help="Semantic query to filter context.",
)
@click.option(
    "--stdout",
    is_flag=True,
    help="Output to stdout instead of clipboard.",
)
def context(
    path: str | None,
    load_all: bool,
    query: str | None,
    stdout: bool,
) -> None:
    """Generate AI context from documentation.

    Retrieves relevant documentation for AI coding sessions.
    Wraps the opencode-context functionality.

    Examples:
        bible context --all ./agent_docs
        bible context --query "error handling"
        bible context ./docs/philosophy.md
    """
    console.print("[bold blue]Loading context...[/]")

    if path:
        console.print(f"  Path: {path}")
    if load_all:
        console.print("  Mode: Load all documents")
    if query:
        console.print(f"  Query: {query}")

    console.print("[yellow]Note: Full implementation wraps opencode-context[/]")
    # TODO: Implementation wraps opencode-context
    # - Call oc-context with appropriate flags
    # - Format output for AI consumption
    # - Copy to clipboard or stdout


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--check",
    "-c",
    multiple=True,
    type=click.Choice([
        "unitarity",
        "hermiticity",
        "trace",
        "positivity",
        "normalization",
        "all",
    ]),
    help="Validation check to perform.",
)
@click.option(
    "--rtol",
    type=float,
    default=1e-5,
    help="Relative tolerance for comparisons.",
)
@click.option(
    "--atol",
    type=float,
    default=1e-8,
    help="Absolute tolerance for comparisons.",
)
def validate(
    file: str,
    check: tuple[str, ...],
    rtol: float,
    atol: float,
) -> None:
    """Validate physics constraints in data files.

    Checks numpy arrays or HDF5 datasets against physics constraints
    like unitarity, hermiticity, and trace preservation.

    Examples:
        bible validate state.npy --check unitarity
        bible validate results.h5 --check all
        bible validate matrix.npy -c unitarity -c hermiticity
    """
    checks = list(check) if check else ["all"]
    sys.exit(run_validate(file, checks, rtol, atol))


@cli.command()
def info() -> None:
    """Show information about AgentBible installation."""
    console.print(f"[bold]AgentBible[/] v{__version__}")
    console.print()
    console.print("[bold]Installed components:[/]")
    console.print("  - validators: Physics validation decorators")
    console.print("  - provenance: HDF5 data tracking with metadata")
    console.print("  - testing: Physics-aware pytest fixtures")
    console.print("  - cli: Command-line interface")
    console.print()
    console.print("[bold]Available commands:[/]")
    console.print("  bible init      - Create new project from template")
    console.print("  bible context   - Generate AI context from docs")
    console.print("  bible validate  - Validate physics constraints")
    console.print("  bible info      - Show this information")


if __name__ == "__main__":
    cli()
