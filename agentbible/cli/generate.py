"""Generate minimal, evidence-based AGENTS.md files.

Produces context files that follow arxiv:2602.11988 recommendations:
- Tool specifications only (test runner, linter, type checker)
- Domain-specific physics constraints when relevant
- No codebase overviews, no workflow checklists, no code examples
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()

# Domain-specific physics constraint lines
_DOMAIN_LINES: dict[str, list[str]] = {
    "quantum": [
        "- Validate quantum operators: `from agentbible.validators import validate_unitary`",
        "- Set seeds: `np.random.seed(seed)`",
    ],
    "ml": [
        "- Set seeds for reproducibility: `np.random.seed(seed); torch.manual_seed(seed)`",
    ],
    "atmospheric": [
        "- Validate physical bounds: `from agentbible.validators import validate_bounds`",
        "- Conserve mass/energy in all transforms",
    ],
}


def build_agents_md(
    domain: str = "none",
    test_cmd: str = "pytest -x",
    coverage: int = 80,
) -> str:
    """Build minimal AGENTS.md content.

    Args:
        domain: Physics domain (quantum/ml/atmospheric/none).
        test_cmd: Test command to use.
        coverage: Coverage threshold percentage.

    Returns:
        AGENTS.md file content as a string.
    """
    lines = [
        "# AGENTS.md",
        "",
        "## Tools",
        f"- Test: `{test_cmd}`",
        "- Lint: `ruff check .`",
        "- Type: `mypy src/`",
        f"- Coverage: {coverage}% minimum (`pytest --cov=src --cov-fail-under={coverage}`)",
    ]

    if domain in _DOMAIN_LINES:
        lines += ["", "## Physics"] + _DOMAIN_LINES[domain]

    lines += [
        "",
        "## Rules",
        "- Max 50 lines per function",
        "- No bare `except:`",
    ]

    return "\n".join(lines) + "\n"


def run_generate_agents_md(
    domain: str = "none",
    test_cmd: str = "pytest -x",
    coverage: int = 80,
    stdout: bool = False,
) -> int:
    """Generate a minimal AGENTS.md file.

    Args:
        domain: Physics domain (quantum/ml/atmospheric/none).
        test_cmd: Test command override.
        coverage: Coverage threshold percentage.
        stdout: Print to stdout instead of writing file.

    Returns:
        Exit code (0 = success).
    """
    content = build_agents_md(domain=domain, test_cmd=test_cmd, coverage=coverage)

    if stdout:
        print(content, end="")
        return 0

    output_path = Path("AGENTS.md")
    output_path.write_text(content, encoding="utf-8")
    console.print(
        f"[green]✓[/] Written {output_path} ({len(content.splitlines())} lines)"
    )
    console.print("[dim]Verify with: bible audit context AGENTS.md[/]")
    return 0
