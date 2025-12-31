# Sprint 1: Foundation

**Sprint:** 1 of 6  
**Focus:** Invisible Infrastructure & AI Enforcement  
**Status:** Active  
**Estimated Sessions:** 2-3

---

## Objective

Make the repository a "one-click" development environment with aggressive AI agent enforcement. A developer should be able to:

1. Clone the repo
2. Run `./bootstrap.sh`
3. Open in VS Code (devcontainer) or terminal
4. Have AI agents automatically follow the rules

---

## Deliverables

### 1. bootstrap.sh

**Location:** `/bootstrap.sh`

**Purpose:** Single-command environment setup

**Requirements:**

```bash
#!/usr/bin/env bash
# Usage: ./bootstrap.sh [--minimal]

# 1. OS Detection
#    - Linux (Ubuntu/Debian, Fedora, Arch)
#    - macOS (Homebrew)
#    - WSL (treat as Linux)

# 2. Python Environment
#    - Check Python 3.9+ installed
#    - Create venv at .venv/
#    - Install dependencies from requirements.txt

# 3. Pre-commit Hooks
#    - pip install pre-commit
#    - pre-commit install

# 4. Vector DB Setup
#    - Copy opencode-context to ~/.local/share/opencode/ if not exists
#    - Run oc-update to embed docs

# 5. Verification
#    - Run pytest to verify setup
#    - Print success message with next steps

# --minimal flag:
#    - Skip venv creation
#    - Skip pre-commit
#    - Only copy docs/prompts
```

**Acceptance Criteria:**
- [ ] Runs on Ubuntu 22.04
- [ ] Runs on macOS 14+
- [ ] Runs on WSL2
- [ ] `--minimal` flag works
- [ ] Idempotent (safe to run multiple times)
- [ ] Prints clear error messages on failure

---

### 2. .cursorrules

**Location:** `/.cursorrules`

**Purpose:** Enforce coding standards in AI agents (Cursor, Claude Code, etc.)

**Content:**

```markdown
# Agent Bible: Coding Rules

You are working in a research code repository that enforces production-grade standards.

## MANDATORY RULES (NEVER VIOLATE)

### 1. Specification Before Code
- REFUSE to write implementation code without a test specification
- Ask: "What are the test cases?" before writing any function
- Write test file FIRST, then implementation

### 2. Rule of 50
- REFUSE to write functions longer than 50 lines (Python) or 60 lines (C++)
- If a function exceeds this limit, STOP and refactor into smaller functions
- Each function does ONE thing

### 3. Input Validation
- ALWAYS validate inputs at function boundaries
- NEVER assume inputs are valid
- Raise descriptive exceptions: what failed, what was expected, where

### 4. Physical Correctness (Research Code)
- ALWAYS check physical constraints:
  - Unitarity: U†U = I
  - Trace preservation: tr(ρ) = 1
  - Probability bounds: 0 ≤ p ≤ 1
  - Hermiticity: H = H†
- ALWAYS cite sources for physics equations (paper, textbook, DOI)

### 5. No Silent Failures
- NEVER catch exceptions and ignore them
- NEVER use bare `except:` clauses
- ALWAYS log or re-raise with context

### 6. Reproducibility
- ALWAYS set random seeds explicitly
- ALWAYS document seed values
- ALWAYS pin dependency versions

## CODE STYLE

### Python
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- snake_case for functions/variables, PascalCase for classes
- Use `ruff` formatting

### C++
- RAII for all resource management (no raw new/delete)
- `camelCase` for functions, `PascalCase` for classes
- Trailing underscore for private members: `member_`
- Zero-warning policy: treat warnings as errors

## WORKFLOW

1. Before writing code:
   - Ask about edge cases
   - Define test cases
   - Identify physical constraints

2. While writing code:
   - Write tests alongside implementation
   - Validate inputs at boundaries
   - Keep functions short

3. After writing code:
   - Run tests
   - Check coverage
   - Verify physical constraints

## REFUSAL PATTERNS

Say "I need to refactor this first" when:
- A function exceeds 50 lines
- There are no tests for the code being modified
- Error handling is missing
- Physical constraints are unchecked

Say "I need clarification" when:
- The expected behavior is ambiguous
- Edge cases are undefined
- Physical assumptions are unclear
```

**Acceptance Criteria:**
- [ ] File exists at repo root
- [ ] Cursor loads it automatically
- [ ] Rules are actionable (not vague)
- [ ] Includes REFUSE patterns

---

### 3. .devcontainer/

**Location:** `/.devcontainer/`

**Purpose:** VS Code development container for consistent environment

**Structure:**

```
.devcontainer/
├── devcontainer.json       # Main config
├── Dockerfile              # Container definition
└── post-create.sh          # Setup script run after container creation
```

**devcontainer.json:**
```json
{
  "name": "Research Code Principles",
  "build": {
    "dockerfile": "Dockerfile"
  },
  "postCreateCommand": ".devcontainer/post-create.sh",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "tamasfe.even-better-toml"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/workspace/.venv/bin/python",
        "python.testing.pytestEnabled": true,
        "editor.formatOnSave": true
      }
    }
  },
  "mounts": [
    "source=${localEnv:HOME}/.local/share/opencode,target=/home/vscode/.local/share/opencode,type=bind"
  ],
  "remoteUser": "vscode"
}
```

**Dockerfile:**
```dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python tools
RUN pip install --upgrade pip && \
    pip install ruff mypy pytest pytest-cov pre-commit

# Create workspace
WORKDIR /workspace
```

**post-create.sh:**
```bash
#!/bin/bash
set -e

# Create venv and install deps
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" 2>/dev/null || pip install -r requirements.txt

# Setup pre-commit
pre-commit install

echo "✓ Development environment ready"
```

**Acceptance Criteria:**
- [ ] `devcontainer.json` is valid
- [ ] Container builds successfully
- [ ] VS Code opens with correct extensions
- [ ] Python interpreter is set correctly
- [ ] Pre-commit hooks installed

---

### 4. agent_docs/

**Location:** `/agent_docs/`

**Purpose:** Meta-documentation for working on this repository itself

**Structure:**

```
agent_docs/
├── SCOPE_OF_WORK.md        # Overall vision and sprint overview
├── sprint-1-foundation.md  # This file - Sprint 1 details
├── ARCHITECTURE.md         # Target directory structure
└── README.md               # How to use these docs
```

**Acceptance Criteria:**
- [ ] All files created
- [ ] Added to opencode config
- [ ] Embedded in vector DB
- [ ] `oc-context --all ./agent_docs` works

---

### 5. Configuration Updates

**opencode config.yaml updates:**

```yaml
projects:
  research-code-principles:
    root: /home/rylan/Documents/career/code_bases/docs/workspace
    description: "Agent Bible - Research code principles and tools"
    docs:
      - path: agent_docs/SCOPE_OF_WORK.md
        always_include: true
        description: "Project scope and sprint overview"
      - path: agent_docs/sprint-1-foundation.md
        always_include: true
        description: "Current sprint details"
      - path: agent_docs/ARCHITECTURE.md
        always_include: true
        description: "Target directory structure"
```

**Acceptance Criteria:**
- [ ] Config updated
- [ ] `oc-update --project research-code-principles` succeeds
- [ ] `oc-context --all ./agent_docs` returns all docs

---

## Tasks

### Phase 1: Documentation (This Session)

| Task | Status | Notes |
|------|--------|-------|
| Create `agent_docs/SCOPE_OF_WORK.md` | ✅ | Complete |
| Create `agent_docs/sprint-1-foundation.md` | 🔄 | This file |
| Create `agent_docs/ARCHITECTURE.md` | ⬜ | Next |
| Update opencode config | ⬜ | Add project entry |
| Embed docs with oc-update | ⬜ | After config update |
| Commit and push | ⬜ | End of session |

### Phase 2: Infrastructure (Next Session)

| Task | Status | Notes |
|------|--------|-------|
| Create `bootstrap.sh` | ⬜ | OS detection, venv, deps |
| Create `.cursorrules` | ⬜ | Aggressive enforcement |
| Create `.devcontainer/` | ⬜ | Python base container |
| Test on fresh clone | ⬜ | Verify bootstrap works |
| Commit and push | ⬜ | End of session |

---

## Testing Plan

### bootstrap.sh Testing

```bash
# Test 1: Fresh clone
cd /tmp
git clone https://github.com/rylanmalarchick/research-code-principles.git
cd research-code-principles
./bootstrap.sh
# Expected: venv created, deps installed, pre-commit ready

# Test 2: Minimal mode
./bootstrap.sh --minimal
# Expected: Only docs copied, no venv

# Test 3: Idempotent
./bootstrap.sh
./bootstrap.sh
# Expected: No errors on second run
```

### .devcontainer Testing

1. Open repo in VS Code
2. Click "Reopen in Container" when prompted
3. Verify Python interpreter is `.venv/bin/python`
4. Verify `pytest` runs successfully
5. Verify ruff extension is active

---

## Definition of Done

Sprint 1 is complete when:

- [ ] `./bootstrap.sh` runs successfully on Linux/macOS/WSL
- [ ] `.cursorrules` is loaded by Cursor (verify in Cursor settings)
- [ ] `.devcontainer/` opens successfully in VS Code
- [ ] `agent_docs/` embedded in vector DB
- [ ] `oc-context --all ./agent_docs` returns all sprint docs
- [ ] All changes committed and pushed

---

## Notes

- Keep bootstrap.sh simple - don't try to handle every edge case
- .cursorrules should be opinionated - that's the point
- .devcontainer is optional for users but helps with consistency
- We eat our own dogfood: use these docs with oc-context while building
