#!/usr/bin/env bash
#
# bootstrap.sh - One-command setup for research-code-principles
#
# Usage:
#   ./bootstrap.sh           # Full setup (venv, deps, pre-commit, vector DB)
#   ./bootstrap.sh --minimal # Docs only, no venv or hooks
#   ./bootstrap.sh --help    # Show help
#
# Requirements:
#   - Python 3.9+
#   - Git
#
# Author: Rylan Malarchick
# License: MIT

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.9"
VENV_DIR=".venv"
OPENCODE_DIR="$HOME/.local/share/opencode"

#######################################
# Print colored message
#######################################
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

#######################################
# Show help message
#######################################
show_help() {
    cat << EOF
research-code-principles bootstrap script

Usage: ./bootstrap.sh [OPTIONS]

Options:
    --minimal   Skip venv creation, dependencies, and pre-commit hooks
                Only installs opencode-context for doc retrieval
    --no-vector Skip vector DB setup (oc-update)
    --help      Show this help message

Examples:
    ./bootstrap.sh              # Full setup
    ./bootstrap.sh --minimal    # Just the docs/context tools
    ./bootstrap.sh --no-vector  # Skip embedding (faster, needs manual oc-update)

EOF
    exit 0
}

#######################################
# Detect operating system
#######################################
detect_os() {
    case "$(uname -s)" in
        Linux*)
            if grep -q Microsoft /proc/version 2>/dev/null; then
                echo "wsl"
            elif [ -f /etc/debian_version ]; then
                echo "debian"
            elif [ -f /etc/fedora-release ]; then
                echo "fedora"
            elif [ -f /etc/arch-release ]; then
                echo "arch"
            else
                echo "linux"
            fi
            ;;
        Darwin*)
            echo "macos"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

#######################################
# Check Python version
#######################################
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        error "Python not found. Please install Python $PYTHON_MIN_VERSION or higher."
    fi

    # Check version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    REQUIRED_VERSION=$PYTHON_MIN_VERSION

    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        error "Python $PYTHON_MIN_VERSION+ required, found $PYTHON_VERSION"
    fi

    success "Python $PYTHON_VERSION found"
}

#######################################
# Create virtual environment
#######################################
create_venv() {
    if [ -d "$VENV_DIR" ]; then
        warn "Virtual environment already exists at $VENV_DIR"
        return 0
    fi

    info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR"
}

#######################################
# Install dependencies
#######################################
install_deps() {
    info "Installing dependencies..."
    
    # Activate venv
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip --quiet
    
    # Install dev dependencies
    if [ -f "pyproject.toml" ]; then
        pip install -e ".[dev]" --quiet 2>/dev/null || pip install -e "." --quiet 2>/dev/null || true
    fi
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --quiet
    fi
    
    # Core tools
    pip install pre-commit ruff mypy pytest pytest-cov --quiet
    
    success "Dependencies installed"
}

#######################################
# Setup pre-commit hooks
#######################################
setup_precommit() {
    if [ ! -f ".pre-commit-config.yaml" ]; then
        warn "No .pre-commit-config.yaml found, skipping pre-commit setup"
        return 0
    fi

    info "Setting up pre-commit hooks..."
    source "$VENV_DIR/bin/activate"
    pre-commit install --quiet
    success "Pre-commit hooks installed"
}

#######################################
# Setup opencode-context
#######################################
setup_opencode() {
    info "Setting up opencode-context..."
    
    # Create opencode directory if needed
    mkdir -p "$OPENCODE_DIR"
    
    # Copy opencode-context if not already there
    if [ ! -d "$OPENCODE_DIR/oc_lib" ]; then
        if [ -d "opencode-context" ]; then
            cp -r opencode-context/oc_lib "$OPENCODE_DIR/"
            cp -r opencode-context/bin "$OPENCODE_DIR/" 2>/dev/null || mkdir -p "$OPENCODE_DIR/bin"
            cp opencode-context/bin/* "$OPENCODE_DIR/bin/" 2>/dev/null || true
            chmod +x "$OPENCODE_DIR/bin/"* 2>/dev/null || true
            success "opencode-context installed to $OPENCODE_DIR"
        else
            warn "opencode-context directory not found in repo"
        fi
    else
        success "opencode-context already installed"
    fi
    
    # Create venv for opencode if needed
    if [ ! -d "$OPENCODE_DIR/venv" ]; then
        info "Creating opencode virtual environment..."
        $PYTHON_CMD -m venv "$OPENCODE_DIR/venv"
        source "$OPENCODE_DIR/venv/bin/activate"
        pip install --upgrade pip --quiet
        
        if [ -f "opencode-context/requirements.txt" ]; then
            pip install -r opencode-context/requirements.txt --quiet
        else
            pip install chromadb openai tiktoken pyyaml click rich --quiet
        fi
        success "opencode-context venv created"
    fi
    
    # Check if config exists
    if [ ! -f "$OPENCODE_DIR/config.yaml" ]; then
        if [ -f "opencode-context/config.example.yaml" ]; then
            cp opencode-context/config.example.yaml "$OPENCODE_DIR/config.yaml"
            warn "Created default config at $OPENCODE_DIR/config.yaml - please edit with your paths"
        fi
    fi
}

#######################################
# Run vector DB update
#######################################
run_vector_update() {
    if [ ! -f "$OPENCODE_DIR/bin/oc-update" ]; then
        warn "oc-update not found, skipping vector DB setup"
        return 0
    fi
    
    # Check for OpenAI key
    if [ -z "$OPENAI_API_KEY" ]; then
        # Try to load from bashrc
        OPENAI_API_KEY=$(grep "OPENAI_API_KEY" ~/.bashrc 2>/dev/null | sed 's/export OPENAI_API_KEY=//' | tr -d '"' | tr -d "'" | head -1)
    fi
    
    if [ -z "$OPENAI_API_KEY" ]; then
        warn "OPENAI_API_KEY not set - skipping vector DB embedding"
        warn "Set the key and run: oc-update"
        return 0
    fi
    
    info "Embedding documentation in vector DB..."
    export OPENAI_API_KEY
    source "$OPENCODE_DIR/venv/bin/activate"
    "$OPENCODE_DIR/bin/oc-update" --status || true
    success "Vector DB ready"
}

#######################################
# Print next steps
#######################################
print_next_steps() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo -e "     ${BLUE}source $VENV_DIR/bin/activate${NC}"
    echo ""
    echo "  2. Run tests to verify:"
    echo -e "     ${BLUE}pytest${NC}"
    echo ""
    echo "  3. Load context for AI sessions:"
    echo -e "     ${BLUE}oc-context --all ./agent_docs${NC}"
    echo ""
    echo "  4. Copy a template to start a new project:"
    echo -e "     ${BLUE}cp -r templates/python_research ~/my-project${NC}"
    echo ""
    
    if [ ! -z "$OPENAI_API_KEY" ]; then
        echo -e "${GREEN}Vector DB is ready for semantic search.${NC}"
    else
        echo -e "${YELLOW}Set OPENAI_API_KEY and run 'oc-update' for semantic search.${NC}"
    fi
    echo ""
}

#######################################
# Print minimal setup message
#######################################
print_minimal_steps() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Minimal Setup Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Installed:"
    echo "  - opencode-context (vector-based doc retrieval)"
    echo ""
    echo "Usage:"
    echo -e "  ${BLUE}source $OPENCODE_DIR/venv/bin/activate${NC}"
    echo -e "  ${BLUE}oc-context --all ./agent_docs${NC}"
    echo ""
}

#######################################
# Main
#######################################
main() {
    local minimal=false
    local skip_vector=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --minimal)
                minimal=true
                shift
                ;;
            --no-vector)
                skip_vector=true
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
    echo -e "${BLUE}research-code-principles bootstrap${NC}"
    echo "=================================="
    echo ""

    # Detect OS
    OS=$(detect_os)
    info "Detected OS: $OS"

    # Check Python
    check_python

    if [ "$minimal" = true ]; then
        # Minimal setup - just opencode-context
        setup_opencode
        if [ "$skip_vector" = false ]; then
            run_vector_update
        fi
        print_minimal_steps
    else
        # Full setup
        create_venv
        install_deps
        setup_precommit
        setup_opencode
        if [ "$skip_vector" = false ]; then
            run_vector_update
        fi
        print_next_steps
    fi
}

main "$@"
