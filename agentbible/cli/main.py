"""AgentBible CLI - command-line interface for correctness and provenance.

Usage:
    bible --help
    bible init my-project --template python-scientific
    bible context --query "unitary validation"
    bible validate results.npy --check unitary
    bible audit code ./src --json
    bible audit context AGENTS.md
    bible generate-agents-md --domain quantum
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
from agentbible.cli.context_audit import run_audit_context
from agentbible.cli.generate import run_generate_agents_md
from agentbible.cli.init import run_init
from agentbible.cli.registry import run_registry
from agentbible.cli.report import run_report
from agentbible.cli.retrofit import run_retrofit
from agentbible.cli.scaffold import run_scaffold
from agentbible.cli.validate import run_validate

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="agentbible")
def cli() -> None:
    """AgentBible: language-agnostic correctness tooling.

    Tools for numerical validation, provenance reporting,
    source retrieval, and project scaffolding.
    """
    pass


@cli.command()
@click.argument("name")
@click.option(
    "--template",
    "-t",
    type=click.Choice(
        [
            "python-scientific",
            "python-quantum",
            "python-ml",
            "python-simulation",
            "cpp-hpc-cuda",
        ]
    ),
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
@click.argument("filepath", type=click.Path())
@click.option(
    "--class",
    "class_name",
    type=str,
    help="Generate a class with this name.",
)
@click.option(
    "--dataclass",
    "dataclass_name",
    type=str,
    help="Generate a dataclass with this name.",
)
@click.option(
    "--methods",
    type=str,
    help="Comma-separated method names (use with --class).",
)
@click.option(
    "--fields",
    type=str,
    help="Comma-separated field:type pairs (use with --dataclass).",
)
@click.option(
    "--functions",
    type=str,
    help="Comma-separated function names to generate.",
)
@click.option(
    "--no-test",
    is_flag=True,
    help="Skip test file generation.",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Add validation to dataclass fields.",
)
def scaffold(
    filepath: str,
    class_name: str | None,
    dataclass_name: str | None,
    methods: str | None,
    fields: str | None,
    functions: str | None,
    no_test: bool,
    validate: bool,
) -> None:
    """Generate module stubs with docstrings and type hints.

    Creates a source file with proper documentation structure
    and a matching test file. Follows AgentBible principles.

    Examples:
        bible scaffold src/optimizer.py --class Optimizer
        bible scaffold src/metrics.py --dataclass Metrics --fields "x:int,y:float"
        bible scaffold src/utils.py --functions "load,save,validate"
        bible scaffold src/bridge.py --class Bridge --methods "run,process"
    """
    sys.exit(
        run_scaffold(
            filepath=filepath,
            class_name=class_name,
            dataclass_name=dataclass_name,
            methods=methods,
            fields=fields,
            functions=functions,
            no_test=no_test,
            validate=validate,
        )
    )


@cli.command()
@click.argument("path", required=False, type=click.Path(exists=True))
@click.option(
    "--cursorrules",
    is_flag=True,
    help="Add .cursorrules file.",
)
@click.option(
    "--precommit",
    is_flag=True,
    help="Add .pre-commit-config.yaml with AgentBible hooks.",
)
@click.option(
    "--validators",
    is_flag=True,
    help="Add validation.py helper module.",
)
@click.option(
    "--conftest",
    is_flag=True,
    help="Add tests/conftest.py with fixtures.",
)
@click.option(
    "--agent-docs",
    "agent_docs",
    is_flag=True,
    help="Add agent_docs/ directory.",
)
@click.option(
    "--all",
    "all_components",
    is_flag=True,
    help="Add all components without prompting.",
)
@click.option(
    "--no-interactive",
    "no_interactive",
    is_flag=True,
    help="Skip interactive prompts.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing files.",
)
def retrofit(
    path: str | None,
    cursorrules: bool,
    precommit: bool,
    validators: bool,
    conftest: bool,
    agent_docs: bool,
    all_components: bool,
    no_interactive: bool,
    force: bool,
) -> None:
    """Add AgentBible structure to existing projects.

    Detects project type and adds AgentBible components:
    - .cursorrules (AI agent coding rules)
    - .pre-commit-config.yaml (code quality hooks)
    - validation.py (AgentBible validator re-exports)
    - tests/conftest.py (pytest fixtures)
    - agent_docs/ (AI context documents)

    Run without options for interactive mode.

    Examples:
        bible retrofit                    # Interactive mode
        bible retrofit --all              # Add everything
        bible retrofit --cursorrules      # Just add .cursorrules
        bible retrofit /path/to/project   # Retrofit specific project
    """
    sys.exit(
        run_retrofit(
            path=path,
            cursorrules=cursorrules,
            precommit=precommit,
            validators=validators,
            conftest=conftest,
            agent_docs=agent_docs,
            all_components=all_components,
            interactive=not no_interactive,
            force=force,
        )
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
    "--lang",
    type=click.Choice(["python", "cpp", "rust", "julia", "all"]),
    default="all",
    show_default=True,
    help="Restrict query results to one language tree.",
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
    lang: str,
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

    # Load all mode: simple directory loading (deprecated)
    if load_all:
        import sys as _sys

        _sys.stderr.write(
            "DeprecationWarning: `--all` is deprecated and will be removed in v0.7.0.\n"
            "  Broad context loading increases agent inference cost by 20-23% with no\n"
            "  measurable benefit when documentation already exists (arxiv:2602.11988).\n"
            '  Use `bible context --query "your topic"` for task-specific retrieval instead.\n'
        )

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
            result = ctx.build_context(query=query, lang=lang)
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
            "finite_array",
            "positive_array",
            "non_negative_array",
            "probability_array",
            "normalized_l1",
            "symmetric",
            "hermitian",
            "unitary",
            "positive_definite",
            "positive_semidefinite",
            "density_matrix",
        ]
    ),
    help="Validation check to perform.",
)
@click.option(
    "--lang",
    type=click.Choice(["python", "cpp", "rust", "julia"]),
    default="python",
    show_default=True,
    help="Language implementation to validate or inspect.",
)
@click.option(
    "--rtol",
    type=float,
    default=1e-10,
    help="Relative tolerance for comparisons.",
)
@click.option(
    "--atol",
    type=float,
    default=1e-12,
    help="Absolute tolerance for comparisons.",
)
def validate(
    file: str,
    check: tuple[str, ...],
    lang: str,
    rtol: float,
    atol: float,
) -> None:
    """Validate Python data or inspect a provenance JSON record.

    Examples:
        bible validate results.npy --check unitary
        bible validate --lang rust results.provenance.json
        bible validate results.h5 --check density_matrix
    """
    sys.exit(run_validate(file, list(check), lang=lang, rtol=rtol, atol=atol))


@cli.command("check-coverage")
@click.option(
    "--threshold",
    type=float,
    default=80.0,
    help="Coverage threshold percentage (default: 80).",
)
@click.option(
    "--fail-under",
    type=float,
    help="Fail if coverage is below this percentage.",
)
@click.option(
    "--html",
    is_flag=True,
    help="Generate HTML coverage report.",
)
@click.option(
    "--source",
    type=str,
    default=".",
    help="Source directory to measure coverage for.",
)
def check_coverage(
    threshold: float,
    fail_under: float | None,
    html: bool,
    source: str,
) -> None:
    """Quick coverage check using pytest-cov.

    Runs pytest with coverage and checks against threshold.
    Simpler than remembering pytest-cov flags.

    Examples:
        bible check-coverage
        bible check-coverage --threshold 90
        bible check-coverage --fail-under 70 --html
    """
    import subprocess

    args = ["pytest", f"--cov={source}", "--cov-report=term-missing"]

    fail_threshold = fail_under if fail_under is not None else threshold
    args.append(f"--cov-fail-under={fail_threshold}")

    if html:
        args.append("--cov-report=html")

    console.print(f"[bold blue]Running:[/] {' '.join(args)}")
    result = subprocess.run(args)
    sys.exit(result.returncode)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "markdown", "json"]),
    default="text",
    help="Output format.",
)
@click.option(
    "--output",
    "-o",
    type=str,
    help="Output file path (default: stdout).",
)
def report(
    file: str,
    output_format: str,
    output: str | None,
) -> None:
    """Generate a report from a schema-compliant provenance JSON record.

    Examples:
        bible report results.provenance.json
        bible report results.provenance.json --format markdown --output report.md
        bible report results.provenance.json --format json
    """
    sys.exit(run_report(file, output_format, output))


@cli.command()
def info() -> None:
    """Show information about AgentBible installation."""
    import os

    from agentbible.validators.base import ENV_VALIDATION_LEVEL, get_validation_level

    console.print(f"[bold]AgentBible[/] v{__version__}")
    console.print()

    # Show current validation level
    level = get_validation_level()
    env_value = os.environ.get(ENV_VALIDATION_LEVEL)
    console.print("[bold]Environment:[/]")
    if env_value:
        console.print(f"  {ENV_VALIDATION_LEVEL} = {level.value}")
    else:
        console.print(f"  {ENV_VALIDATION_LEVEL} = {level.value} [dim](default)[/]")
    console.print()
    console.print("  [dim]Validation levels:[/]")
    console.print("    debug - Full validation (default)")
    console.print("    lite  - Only NaN/Inf checks (fast)")
    console.print("    off   - Skip validation (benchmarking only)")
    console.print()

    console.print("[bold]Installed components:[/]")
    console.print("  - validators: scalar, array, and matrix checks")
    console.print("  - provenance: HDF5 helpers and JSON records")
    console.print("  - context: cross-language source retrieval")
    console.print("  - testing: reproducibility fixtures and runtime checks")
    console.print("  - cli: validation, reporting, audit, and scaffolding")
    console.print()
    console.print("[bold]Available commands:[/]")
    console.print("  bible init                - Create new project from template")
    console.print("  bible scaffold            - Generate module stubs with tests")
    console.print("  bible retrofit            - Add AgentBible to existing project")
    console.print("  bible context             - Generate AI context from docs")
    console.print("  bible validate            - Validate Python data or inspect provenance JSON")
    console.print(
        "  bible audit code          - Check code against AgentBible principles"
    )
    console.print(
        "  bible audit context       - Score AGENTS.md / .cursorrules for minimal context"
    )
    console.print(
        "  bible generate-agents-md  - Generate minimal AGENTS.md (arxiv:2602.11988)"
    )
    console.print("  bible report              - Generate a report from provenance JSON")
    console.print("  bible ci                  - CI/CD status and release automation")
    console.print("  bible registry            - Manage agent_registry.yaml")
    console.print("  bible info                - Show this information")


@cli.group()
def audit() -> None:
    """Audit code and context files against AgentBible principles.

    Subcommands:
        code     Check Python source code (Rule of 50, docstrings, types)
        context  Check AGENTS.md / .cursorrules for minimal context compliance

    Examples:
        bible audit code ./src --json
        bible audit context AGENTS.md
        bible audit context  (auto-detects AGENTS.md or .cursorrules)
    """
    pass


@audit.command("code")
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
def audit_code(
    path: str,
    output_json: bool,
    no_line_length: bool,
    no_docstrings: bool,
    no_type_hints: bool,
    max_lines: int,
    strict: bool,
    exclude: tuple[str, ...],
) -> None:
    """Audit Python code against AgentBible principles.

    Checks Python code for compliance with:
    - Rule of 50: Functions should be <= 50 lines
    - Docstrings: Public functions/classes need documentation
    - Type hints: Function signatures should have annotations

    Use --json for CI integration (machine-readable output).

    Examples:
        bible audit code ./src
        bible audit code ./src --json
        bible audit code ./src --strict --max-lines 30
        bible audit code ./src --exclude "**/test_*.py"
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


@audit.command("context")
@click.argument("file", required=False, type=click.Path())
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON for CI integration.",
)
def audit_context(file: str | None, output_json: bool) -> None:
    """Audit AGENTS.md or .cursorrules for minimal context compliance.

    Scores the file against arxiv:2602.11988 recommendations.
    Reports codebase overviews, workflow checklists, and long code blocks.

    Exit code 0 if tightness score >= 70, 1 otherwise.

    Examples:
        bible audit context
        bible audit context AGENTS.md
        bible audit context .cursorrules --json
    """
    sys.exit(run_audit_context(file=file, output_json=output_json))


@cli.command("generate-agents-md")
@click.option(
    "--domain",
    type=click.Choice(["quantum", "ml", "atmospheric", "none"]),
    default="none",
    help="Physics domain for domain-specific constraints.",
)
@click.option(
    "--test-cmd",
    "test_cmd",
    type=str,
    default="pytest -x",
    help="Test command (default: pytest -x).",
)
@click.option(
    "--coverage",
    type=int,
    default=80,
    help="Coverage threshold percentage (default: 80).",
)
@click.option(
    "--stdout",
    is_flag=True,
    help="Print to stdout instead of writing AGENTS.md.",
)
def generate_agents_md(
    domain: str,
    test_cmd: str,
    coverage: int,
    stdout: bool,
) -> None:
    """Generate a minimal, evidence-based AGENTS.md.

    Creates an AGENTS.md following arxiv:2602.11988 recommendations:
    tool specifications only, no codebase overviews or checklists.

    Examples:
        bible generate-agents-md
        bible generate-agents-md --domain quantum
        bible generate-agents-md --stdout > AGENTS.md
        bible generate-agents-md --coverage 90
    """
    sys.exit(
        run_generate_agents_md(
            domain=domain,
            test_cmd=test_cmd,
            coverage=coverage,
            stdout=stdout,
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


@cli.group()
def registry() -> None:
    """Manage agent_registry.yaml for AI integration.

    The agent registry maps file patterns to required validators,
    helping AI agents apply appropriate checks automatically.

    Examples:
        bible registry show     # Show current registry
        bible registry init     # Create default registry
        bible registry check src/  # Check files use required validators
    """
    pass


@registry.command("show")
def registry_show() -> None:
    """Show current registry configuration."""
    sys.exit(run_registry("show"))


@registry.command("init")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing registry.",
)
def registry_init(force: bool) -> None:
    """Initialize agent_registry.yaml in current directory."""
    sys.exit(run_registry("init", force=force))


@registry.command("check")
@click.argument("path", type=click.Path(exists=True))
def registry_check(path: str) -> None:
    """Check if files use required validators."""
    sys.exit(run_registry("check", path=path))


if __name__ == "__main__":
    cli()
