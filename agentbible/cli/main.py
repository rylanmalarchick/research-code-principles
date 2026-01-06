"""AgentBible CLI - Command-line interface for research code infrastructure.

Usage:
    bible --help
    bible init my-project --template python-scientific
    bible context --all ./agent_docs
    bible validate state.npy --check unitarity
    bible audit ./src --json
    bible ci status
    bible ci verify --wait
    bible ci release 0.3.0
"""

from __future__ import annotations

import sys

import click
from rich.console import Console

from agentbible import __version__
from agentbible.cli.audit import run_audit
from agentbible.cli.ci import run_ci_release, run_ci_status, run_ci_verify
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
    sys.exit(
        run_init(name, template, author, email, description, no_git, no_venv, force)
    )


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
@click.option(
    "--embed",
    is_flag=True,
    help="Embed documents into vector database for semantic search.",
)
def context(
    path: str | None,
    load_all: bool,
    query: str | None,
    stdout: bool,
    embed: bool,
) -> None:
    """Generate AI context from documentation.

    Retrieves relevant documentation for AI coding sessions.
    Supports both simple directory loading and semantic search.

    Examples:
        bible context --all ./agent_docs
        bible context --query "error handling"
        bible context ./agent_docs --embed
    """
    from agentbible.context import ContextManager

    try:
        ctx = ContextManager()
    except Exception as e:
        console.print(f"[red]Error initializing context manager:[/] {e}")
        return

    # Embed mode: create vector embeddings
    if embed and path:
        console.print(f"[bold blue]Embedding documents from {path}...[/]")
        try:
            num_chunks = ctx.embed_directory(path)
            console.print(f"[green]Created {num_chunks} chunks in vector database[/]")
        except ImportError as e:
            console.print(f"[red]Missing dependencies:[/] {e}")
            console.print("Install with: pip install agentbible[context]")
        except Exception as e:
            console.print(f"[red]Error embedding documents:[/] {e}")
        return

    # Load all mode: simple directory loading
    if load_all and path:
        console.print(f"[bold blue]Loading all documents from {path}...[/]")
        try:
            result = ctx.load_directory(path)
            if stdout:
                console.print(result)
            else:
                # Try to copy to clipboard
                try:
                    import subprocess

                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=result.encode(),
                        check=True,
                    )
                    console.print("[green]Context copied to clipboard![/]")
                except Exception:
                    # Fallback to stdout
                    console.print(result)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/] {e}")
        return

    # Query mode: semantic search
    if query:
        console.print(f"[bold blue]Searching for:[/] {query}")
        try:
            result = ctx.build_context(query=query)
            if stdout:
                console.print(result)
            else:
                console.print(result)
        except ImportError as e:
            console.print(f"[red]Missing dependencies for semantic search:[/] {e}")
            console.print("Install with: pip install agentbible[context]")
        except FileNotFoundError:
            console.print(
                "[yellow]No embeddings found.[/] Run 'bible context --embed <path>' first."
            )
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
        return

    # Single file mode
    if path:
        from pathlib import Path

        p = Path(path)
        if p.is_file():
            content = p.read_text()
            console.print(f"[bold blue]Loaded {p.name}[/]")
            if stdout:
                console.print(content)
            else:
                console.print(content)
        else:
            console.print(
                "[yellow]Hint: Use --all to load all docs from a directory[/]"
            )
        return

    # No arguments: show help
    console.print("[yellow]Usage examples:[/]")
    console.print(
        "  bible context --all ./agent_docs    # Load all docs from directory"
    )
    console.print(
        "  bible context --query 'error'       # Semantic search (requires --embed first)"
    )
    console.print("  bible context ./docs/README.md      # Load single file")


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--check",
    "-c",
    multiple=True,
    type=click.Choice(
        [
            "unitarity",
            "hermiticity",
            "trace",
            "positivity",
            "normalization",
            "all",
        ]
    ),
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
    console.print("  bible audit     - Check code against AgentBible principles")
    console.print("  bible ci        - CI/CD status and release automation")
    console.print("  bible info      - Show this information")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON for CI integration.",
)
@click.option(
    "--no-line-length",
    is_flag=True,
    help="Skip Rule of 50 (function line length) check.",
)
@click.option(
    "--no-docstrings",
    is_flag=True,
    help="Skip docstring presence check.",
)
@click.option(
    "--no-type-hints",
    is_flag=True,
    help="Skip type hints check.",
)
@click.option(
    "--max-lines",
    type=int,
    default=50,
    help="Maximum allowed function lines (default: 50).",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Treat all violations as errors (fail on warnings).",
)
@click.option(
    "--exclude",
    "-e",
    multiple=True,
    help="Glob patterns to exclude (can be used multiple times).",
)
def audit(
    path: str,
    output_json: bool,
    no_line_length: bool,
    no_docstrings: bool,
    no_type_hints: bool,
    max_lines: int,
    strict: bool,
    exclude: tuple[str, ...],
) -> None:
    """Audit code against AgentBible principles.

    Checks Python code for compliance with:
    - Rule of 50: Functions should be <= 50 lines
    - Docstrings: Public functions/classes need documentation
    - Type hints: Function signatures should have annotations

    Use --json for CI integration (machine-readable output).

    Examples:
        bible audit ./src
        bible audit ./src --json
        bible audit ./src --strict --max-lines 30
        bible audit ./src --exclude "**/test_*.py"
    """
    output_format = "json" if output_json else "text"
    exclude_list = list(exclude) if exclude else None

    sys.exit(
        run_audit(
            path=path,
            output_format=output_format,
            check_line_length=not no_line_length,
            check_docstrings=not no_docstrings,
            check_type_hints=not no_type_hints,
            max_lines=max_lines,
            strict=strict,
            exclude=exclude_list,
        )
    )


@cli.group()
def ci() -> None:
    """CI/CD status and release automation.

    Commands for checking GitHub Actions status, verifying CI passes,
    and automating the release process.

    IMPORTANT: AI agents should ALWAYS run 'bible ci status' after
    pushing code to verify CI is passing.

    Examples:
        bible ci status              # Show recent workflow runs
        bible ci verify --wait       # Wait for CI and verify it passes
        bible ci release 0.3.0       # Full release flow
    """
    pass


@ci.command("status")
@click.option(
    "--limit",
    "-l",
    type=int,
    default=10,
    help="Number of runs to show.",
)
@click.option(
    "--branch",
    "-b",
    type=str,
    help="Filter by branch name.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON.",
)
def ci_status(limit: int, branch: str | None, output_json: bool) -> None:
    """Show CI/CD workflow status.

    Displays recent GitHub Actions workflow runs with their status.
    Use this to verify that CI is passing after pushing code.

    Examples:
        bible ci status
        bible ci status --branch main
        bible ci status --json
    """
    sys.exit(run_ci_status(limit=limit, branch=branch, output_json=output_json))


@ci.command("verify")
@click.option(
    "--branch",
    "-b",
    type=str,
    help="Branch to verify.",
)
@click.option(
    "--wait",
    "-w",
    is_flag=True,
    help="Wait for in-progress runs to complete.",
)
def ci_verify(branch: str | None, wait: bool) -> None:
    """Verify CI is passing.

    Checks that all recent workflow runs have passed.
    Use --wait to wait for in-progress runs to complete.

    Exit codes:
        0 = All checks passing
        1 = One or more checks failed

    Examples:
        bible ci verify
        bible ci verify --wait
        bible ci verify --branch feature-x --wait
    """
    sys.exit(run_ci_verify(branch=branch, wait=wait))


@ci.command("release")
@click.argument("version")
@click.option(
    "--no-bump",
    is_flag=True,
    help="Skip version bump in files.",
)
@click.option(
    "--no-push",
    is_flag=True,
    help="Skip pushing to remote.",
)
@click.option(
    "--no-verify",
    is_flag=True,
    help="Skip waiting for CI verification.",
)
@click.option(
    "--draft",
    is_flag=True,
    help="Create release as draft.",
)
def ci_release(
    version: str,
    no_bump: bool,
    no_push: bool,
    no_verify: bool,
    draft: bool,
) -> None:
    """Run full release flow.

    Automates the complete release process:
    1. Bump version in pyproject.toml and __init__.py
    2. Commit version changes
    3. Create and push git tag
    4. Wait for CI to pass
    5. Create GitHub release

    VERSION should be like "0.3.0" or "v0.3.0".

    Examples:
        bible ci release 0.3.0
        bible ci release v0.3.0 --draft
        bible ci release 0.3.0 --no-verify
    """
    sys.exit(
        run_ci_release(
            version=version,
            bump_files=not no_bump,
            push=not no_push,
            verify=not no_verify,
            draft=draft,
        )
    )


if __name__ == "__main__":
    cli()
