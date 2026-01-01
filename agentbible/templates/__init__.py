"""Embedded templates for bible init command.

Templates are copied to new projects with variable substitution.
"""

from pathlib import Path

# Get the templates directory path
TEMPLATES_DIR = Path(__file__).parent

# Available templates
# Note: cpp-hpc-cuda template planned for future release
AVAILABLE_TEMPLATES = {
    "python-scientific": TEMPLATES_DIR / "python_research",
}

__all__ = ["TEMPLATES_DIR", "AVAILABLE_TEMPLATES"]
