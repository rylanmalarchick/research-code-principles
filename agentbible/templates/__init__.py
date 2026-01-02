"""Embedded templates for bible init command.

Templates are copied to new projects with variable substitution.
"""

from pathlib import Path

# Get the templates directory path
TEMPLATES_DIR = Path(__file__).parent

# Available templates
AVAILABLE_TEMPLATES = {
    "python-scientific": TEMPLATES_DIR / "python_research",
    "cpp-hpc-cuda": TEMPLATES_DIR / "cpp_hpc",
}

__all__ = ["TEMPLATES_DIR", "AVAILABLE_TEMPLATES"]
