#!/bin/bash
# Post-creation script for VS Code devcontainer
# Runs after the container is created to set up the development environment

set -e

echo "Setting up development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
if [ -f "pyproject.toml" ]; then
    echo "Installing from pyproject.toml..."
    pip install -e ".[dev]" 2>/dev/null || pip install -e "." 2>/dev/null || echo "No installable package found"
fi

if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
fi

if [ -f "requirements-dev.txt" ]; then
    echo "Installing dev dependencies..."
    pip install -r requirements-dev.txt
fi

# Install pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    echo "Installing pre-commit hooks..."
    pip install pre-commit
    pre-commit install
fi

# Set up opencode context manager if available
if [ -d "opencode-context" ]; then
    echo "Setting up opencode context manager..."
    OPENCODE_DIR="$HOME/.local/share/opencode"
    
    if [ ! -d "$OPENCODE_DIR" ]; then
        mkdir -p "$OPENCODE_DIR"
    fi
    
    # Copy context manager if not already installed
    if [ ! -f "$OPENCODE_DIR/oc_context.py" ]; then
        cp -r opencode-context/* "$OPENCODE_DIR/"
        
        # Create venv for context manager if needed
        if [ ! -d "$OPENCODE_DIR/venv" ]; then
            python -m venv "$OPENCODE_DIR/venv"
            source "$OPENCODE_DIR/venv/bin/activate"
            pip install chromadb openai tiktoken
            deactivate
        fi
    fi
fi

# Run a quick test to verify setup
echo ""
echo "Verifying setup..."
python --version
pip --version

if command -v pytest &> /dev/null; then
    echo "pytest: $(pytest --version 2>/dev/null | head -1)"
fi

if command -v ruff &> /dev/null; then
    echo "ruff: $(ruff --version)"
fi

echo ""
echo "========================================"
echo "  Development environment ready!"
echo "========================================"
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  pytest                    # Run tests"
echo "  ruff check .              # Lint code"
echo "  pre-commit run --all-files # Run all hooks"
echo ""
