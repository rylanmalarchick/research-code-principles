# Sprint 3: CLI Scaffolding

**Sprint:** 3 of 6  
**Focus:** `bible init` creates production-ready projects from templates  
**Status:** Active  
**Estimated Sessions:** 2-3

---

## Objective

Implement the `bible init` command that:

1. Copies project templates with customization (project name, author)
2. Optionally initializes git and creates a virtual environment
3. Includes `.cursorrules` in generated projects
4. Produces projects that pass `pytest` immediately

After this sprint: `bible init my-project` creates a working, testable project.

---

## Deliverables

### 1. Template Embedding

**Location:** `agentbible/templates/`

Templates are embedded in the package so they're available after `pip install agentbible`.

```
agentbible/templates/
├── python_research/     # Copy of templates/python_research
│   ├── pyproject.toml.template
│   ├── README.md.template
│   ├── .cursorrules
│   ├── src/
│   └── tests/
└── cpp_hpc/             # Copy of templates/cpp_hpc
    ├── CMakeLists.txt.template
    ├── README.md.template
    ├── .cursorrules
    ├── include/
    ├── src/
    └── tests/
```

---

### 2. Template Variables

Files with `.template` extension are processed with variable substitution:

| Variable | Description | Example |
|----------|-------------|---------|
| `{{PROJECT_NAME}}` | Project name (from CLI arg) | `my-quantum-sim` |
| `{{PROJECT_NAME_UNDERSCORE}}` | Python-safe name | `my_quantum_sim` |
| `{{AUTHOR_NAME}}` | Author name | `Rylan Malarchick` |
| `{{AUTHOR_EMAIL}}` | Author email | `rylan@example.com` |
| `{{YEAR}}` | Current year | `2025` |
| `{{DATE}}` | Current date | `2025-01-01` |

---

### 3. CLI Implementation

**Location:** `agentbible/cli/init.py`

```python
@cli.command()
@click.argument("name")
@click.option("--template", "-t", type=click.Choice(["python-scientific", "cpp-hpc-cuda"]))
@click.option("--author", "-a", type=str, help="Author name")
@click.option("--email", "-e", type=str, help="Author email")
@click.option("--no-git", is_flag=True, help="Skip git init")
@click.option("--no-venv", is_flag=True, help="Skip venv creation")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing directory")
def init(name, template, author, email, no_git, no_venv, force):
    """Initialize a new project from template."""
    ...
```

**Workflow:**

1. Validate project name (no spaces, valid Python identifier)
2. Check destination doesn't exist (or --force)
3. Copy template files to destination
4. Process `.template` files with variable substitution
5. Rename `.template` files (remove suffix)
6. Run `git init` (unless --no-git)
7. Run `python -m venv .venv` (unless --no-venv)
8. Print success message with next steps

---

### 4. .cursorrules for Templates

Add a simplified `.cursorrules` to each template that inherits from the main principles but is project-specific.

**templates/python_research/.cursorrules:**
```markdown
# Project Rules

This is a Python research project following AgentBible principles.

## Mandatory
- Write tests before implementation
- Functions ≤ 50 lines
- Type hints on all functions
- Validate physical constraints

## Project Structure
- Source code in `src/`
- Tests in `tests/`
- Run `pytest` before committing
```

---

## Tasks

| Task | Status | Notes |
|------|--------|-------|
| Create `agent_docs/sprint-3-scaffolding.md` | 🔄 | This file |
| Copy templates to `agentbible/templates/` | ⬜ | Embed in package |
| Create `.template` versions of config files | ⬜ | Variable substitution |
| Add `.cursorrules` to templates | ⬜ | Simplified rules |
| Create `agentbible/cli/init.py` | ⬜ | Main logic |
| Update `agentbible/cli/main.py` | ⬜ | Wire up init command |
| Add git init logic | ⬜ | Optional with --no-git |
| Add venv creation logic | ⬜ | Optional with --no-venv |
| Create `tests/test_init.py` | ⬜ | Test scaffolding |
| Test generated project runs pytest | ⬜ | Integration test |
| Update pyproject.toml package data | ⬜ | Include templates |
| Update ARCHITECTURE.md | ⬜ | Mark complete |
| Commit and push | ⬜ | End of sprint |

---

## Testing Plan

```bash
# Test basic init
bible init test-project --template python-scientific
cd test-project
pytest  # Should pass

# Test with options
bible init test-project-2 -t python-scientific --author "Test User" --email "test@example.com"

# Test no-git, no-venv
bible init test-project-3 -t python-scientific --no-git --no-venv
ls -la test-project-3/  # No .git, no .venv

# Test force overwrite
bible init test-project --force  # Should succeed

# Test cpp template
bible init cpp-project -t cpp-hpc-cuda
```

---

## Acceptance Criteria

- [ ] `bible init my-project` creates directory with template files
- [ ] `bible init my-project -t python-scientific` uses Python template
- [ ] `bible init my-project -t cpp-hpc-cuda` uses C++ template
- [ ] Project name substituted in pyproject.toml/CMakeLists.txt
- [ ] Generated Python project passes `pytest`
- [ ] `.cursorrules` included in generated project
- [ ] `--no-git` skips git initialization
- [ ] `--no-venv` skips virtual environment creation
- [ ] `--force` overwrites existing directory
- [ ] Clear error if directory exists without --force
- [ ] Clear error if invalid project name

---

## Design Decisions

### Template File Handling

1. **Static files**: Copied as-is (e.g., `.py`, `.cpp`, `.hpp`)
2. **Template files**: Processed with variable substitution, then renamed
   - `pyproject.toml.template` → `pyproject.toml`
   - `README.md.template` → `README.md`

### Project Name Validation

- Must be valid directory name
- Converted to underscore version for Python imports
- Examples:
  - `my-project` → `my_project` (Python)
  - `QuantumSim` → `quantumsim` (normalized)

### Default Values

- Author: Read from `git config user.name` or prompt
- Email: Read from `git config user.email` or prompt
- Template: `python-scientific` (default)

---

## Notes

- Templates are copied, not symlinked (works after pip install)
- Use `importlib.resources` for Python 3.9+ compatible resource access
- Consider adding `--dry-run` flag for preview
