"""Implementation of the bible registry command.

Manages agent_registry.yaml for AI agent integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from rich.console import Console
from rich.table import Table

console = Console()

# Default registry template
DEFAULT_REGISTRY = {
    "version": 1,
    "description": "AI agents should read this file to determine which validators to apply",
    "patterns": {
        "src/**/gates*.py": {
            "validators": ["check_unitarity"],
            "imports": ["from agentbible.validators import check_unitarity"],
            "rules": [
                "All gate functions must return unitary matrices",
                "Use check_unitarity() to validate gate outputs",
            ],
        },
        "src/**/states*.py": {
            "validators": ["check_normalized", "check_density_matrix"],
            "imports": [
                "from agentbible.validators import check_normalized, check_density_matrix"
            ],
            "rules": [
                "State vectors must be normalized",
                "Density matrices must be Hermitian with trace 1",
            ],
        },
        "src/**/data*.py": {
            "validators": ["check_no_nan_inf"],
            "imports": ["from agentbible.validators import check_no_nan_inf"],
            "rules": [
                "Check for NaN/Inf before processing",
                "Validate data shapes match expectations",
            ],
        },
    },
    "defaults": {
        "python": {
            "validators": ["check_no_nan_inf"],
            "imports": ["from agentbible.validators import check_no_nan_inf"],
            "rules": ["Always validate numerical outputs for NaN/Inf"],
        }
    },
}

REGISTRY_FILENAME = "agent_registry.yaml"


def find_registry(start_path: Path = Path.cwd()) -> Optional[Path]:
    """Find agent_registry.yaml in current or parent directories."""
    current = start_path.resolve()
    while current != current.parent:
        registry = current / REGISTRY_FILENAME
        if registry.exists():
            return registry
        current = current.parent
    return None


def load_registry(path: Path) -> Dict[str, Any]:
    """Load registry from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_registry(path: Path, registry: Dict[str, Any]) -> None:
    """Save registry to YAML file."""
    with open(path, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


def run_registry_show() -> int:
    """Show current registry configuration."""
    registry_path = find_registry()

    if registry_path is None:
        console.print("[yellow]No agent_registry.yaml found.[/]")
        console.print("Run 'bible registry init' to create one.")
        return 1

    console.print(f"[bold]Registry:[/] {registry_path}")
    console.print()

    registry = load_registry(registry_path)

    # Show version
    console.print(f"[bold]Version:[/] {registry.get('version', 'unknown')}")
    console.print()

    # Show patterns table
    patterns = registry.get("patterns", {})
    if patterns:
        table = Table(title="File Pattern Validators")
        table.add_column("Pattern", style="cyan")
        table.add_column("Validators", style="green")
        table.add_column("Rules", style="dim")

        for pattern, config in patterns.items():
            validators = ", ".join(config.get("validators", []))
            rules = "; ".join(config.get("rules", [])[:2])  # First 2 rules
            if len(config.get("rules", [])) > 2:
                rules += "..."
            table.add_row(pattern, validators, rules)

        console.print(table)

    # Show defaults
    defaults = registry.get("defaults", {})
    if defaults:
        console.print()
        console.print("[bold]Default Validators:[/]")
        for lang, config in defaults.items():
            validators = ", ".join(config.get("validators", []))
            console.print(f"  {lang}: {validators}")

    return 0


def run_registry_init(force: bool = False) -> int:
    """Initialize agent_registry.yaml in current directory."""
    registry_path = Path.cwd() / REGISTRY_FILENAME

    if registry_path.exists() and not force:
        console.print(f"[yellow]{REGISTRY_FILENAME} already exists.[/]")
        console.print("Use --force to overwrite.")
        return 1

    save_registry(registry_path, DEFAULT_REGISTRY)
    console.print(f"[green]Created {REGISTRY_FILENAME}[/]")
    console.print()
    console.print("Customize the patterns section for your project's needs.")
    console.print("AI agents will read this file to apply appropriate validators.")

    return 0


def run_registry_check(path: str) -> int:
    """Check if files in path use required validators."""
    import fnmatch
    import re

    registry_path = find_registry()
    if registry_path is None:
        console.print("[yellow]No agent_registry.yaml found.[/]")
        return 1

    registry = load_registry(registry_path)
    patterns = registry.get("patterns", {})

    target_path = Path(path)
    if not target_path.exists():
        console.print(f"[red]Path not found: {path}[/]")
        return 1

    # Collect Python files
    if target_path.is_file():
        files = [target_path]
    else:
        files = list(target_path.rglob("*.py"))

    issues: List[Dict[str, Any]] = []
    checked = 0

    for file in files:
        file_str = str(file)
        for pattern, config in patterns.items():
            # Convert glob pattern to work with fnmatch
            # Handle ** patterns
            pattern_regex = pattern.replace("**", "*")
            if fnmatch.fnmatch(file_str, f"*{pattern_regex}"):
                checked += 1
                validators = config.get("validators", [])
                imports = config.get("imports", [])

                # Read file and check for validators
                try:
                    content = file.read_text()
                except Exception:
                    continue

                missing = []
                for validator in validators:
                    # Check if validator is used (imported or called)
                    if validator not in content:
                        missing.append(validator)

                if missing:
                    issues.append(
                        {
                            "file": str(file),
                            "pattern": pattern,
                            "missing": missing,
                            "imports": imports,
                        }
                    )

    # Report results
    if not issues:
        console.print(f"[green]All {checked} matched files use required validators.[/]")
        return 0

    console.print(f"[yellow]Found {len(issues)} files missing validators:[/]")
    console.print()

    for issue in issues:
        console.print(f"[bold]{issue['file']}[/]")
        console.print(f"  Pattern: {issue['pattern']}")
        console.print(f"  Missing: {', '.join(issue['missing'])}")
        console.print(f"  Add: {issue['imports'][0] if issue['imports'] else 'N/A'}")
        console.print()

    return 1


def run_registry(
    action: str,
    path: Optional[str] = None,
    force: bool = False,
) -> int:
    """Run the registry command.

    Args:
        action: One of 'show', 'init', or 'check'.
        path: Path to check (for 'check' action).
        force: Force overwrite for 'init' action.

    Returns:
        Exit code (0 for success).
    """
    if action == "show":
        return run_registry_show()
    elif action == "init":
        return run_registry_init(force=force)
    elif action == "check":
        if not path:
            console.print("[red]Path required for check action.[/]")
            return 1
        return run_registry_check(path)
    else:
        console.print(f"[red]Unknown action: {action}[/]")
        console.print("Available actions: show, init, check")
        return 1
