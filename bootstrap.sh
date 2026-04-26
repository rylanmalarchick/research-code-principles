#!/usr/bin/env bash
#
# bootstrap.sh - One-command setup for agentbible development
#
# Usage:
#   ./bootstrap.sh           # Full setup (venv, deps, pre-commit)
#   ./bootstrap.sh --minimal # Skip local installation steps
#   ./bootstrap.sh --help    # Show help

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PYTHON_MIN_VERSION="3.9"
VENV_DIR=".venv"

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

show_help() {
    cat << EOF
agentbible bootstrap script

Usage: ./bootstrap.sh [OPTIONS]

Options:
    --minimal   Skip venv creation, dependency install, and pre-commit setup
    --help      Show this help message
EOF
    exit 0
}

check_python() {
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        error "Python not found. Please install Python ${PYTHON_MIN_VERSION}+."
    fi

    local version
    version=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$version" | sort -V | head -n1)" != "$PYTHON_MIN_VERSION" ]; then
        error "Python ${PYTHON_MIN_VERSION}+ required, found $version"
    fi

    success "Python $version found"
}

create_venv() {
    if [ -d "$VENV_DIR" ]; then
        warn "Virtual environment already exists at $VENV_DIR"
        return
    fi

    info "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR"
}

install_deps() {
    info "Installing dependencies..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet
    if [ -f "pyproject.toml" ]; then
        pip install -e ".[dev]" --quiet 2>/dev/null || pip install -e "." --quiet
    fi
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --quiet
    fi
    success "Dependencies installed"
}

setup_precommit() {
    if [ ! -f ".pre-commit-config.yaml" ]; then
        warn "No .pre-commit-config.yaml found, skipping pre-commit setup"
        return
    fi

    info "Setting up pre-commit hooks..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pre-commit install --quiet
    success "Pre-commit hooks installed"
}

print_next_steps() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Next steps:"
    echo -e "  ${BLUE}source $VENV_DIR/bin/activate${NC}"
    echo -e "  ${BLUE}pytest -x${NC}"
    echo -e "  ${BLUE}ruff check agentbible/${NC}"
    echo -e "  ${BLUE}mypy agentbible/${NC}"
    echo ""
}

print_minimal_steps() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Minimal Setup Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Nothing was installed. Run the standard local checks manually if needed."
    echo ""
}

main() {
    local minimal=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --minimal)
                minimal=true
                shift
                ;;
            --help|-h)
                show_help
                ;;
            *)
                error "Unknown option: $1. Use --help for usage."
                ;;
        esac
    done

    echo ""
    echo -e "${BLUE}agentbible bootstrap${NC}"
    echo "===================="
    echo ""

    check_python

    if [ "$minimal" = true ]; then
        print_minimal_steps
        return
    fi

    create_venv
    install_deps
    setup_precommit
    print_next_steps
}

main "$@"
