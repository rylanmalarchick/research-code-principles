# Sprint 5: CI/CD & Security

**Status:** Complete  
**Started:** December 2025  
**Completed:** January 2026

---

## Objectives

Production-ready package publishing with security-first CI/CD pipeline.

---

## Deliverables

### 1. Trusted Publishing (OIDC)

**Status:** Complete

GitHub Actions publishes to PyPI without stored credentials using OpenID Connect.

**Implementation:** `.github/workflows/ci.yml` (lines 111-131)

```yaml
publish:
  runs-on: ubuntu-latest
  needs: [build]
  if: startsWith(github.ref, 'refs/tags/v')
  environment:
    name: pypi
    url: https://pypi.org/project/agentbible/
  permissions:
    id-token: write  # Required for trusted publishing

  steps:
    - name: Download artifacts
      uses: actions/download-artifact@v4
      with:
        name: dist
        path: dist/

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      # No password needed - uses trusted publishing via OIDC
```

**Verification:**
- v0.1.0 successfully published to PyPI
- No API tokens stored in repository secrets

---

### 2. Dependency Vulnerability Scanning (pip-audit)

**Status:** Complete

CI fails on known vulnerabilities in dependencies.

**Implementation:** `.github/workflows/ci.yml` (lines 55-77)

```yaml
security:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.12"

    - name: Install pip-audit
      run: pip install pip-audit

    - name: Install package dependencies only
      run: |
        pip install click>=8.0 rich>=13.0 pyyaml>=6.0 numpy>=1.20
        pip install h5py>=3.0

    - name: Run pip-audit
      run: pip-audit --desc on
```

**Features:**
- Scans all runtime dependencies
- Fails CI on any known vulnerability
- Provides descriptions for found issues

---

### 3. Security Policy (SECURITY.md)

**Status:** Complete

Comprehensive security disclosure policy.

**File:** `SECURITY.md` (70 lines)

**Contents:**
- Supported versions table
- Vulnerability reporting process (email, not public issues)
- Response timeline (Critical: 24-72h, High: 1-2 weeks, etc.)
- Security measures documentation (pip-audit, trusted publishing)
- User best practices
- Scope definition

---

### 4. Automated Dependency Updates (Dependabot)

**Status:** Complete

Automated PRs for dependency updates.

**File:** `.github/dependabot.yml`

**Configuration:**
- Weekly schedule (Monday)
- Grouped updates (dev dependencies, runtime, GitHub Actions)
- 5 open PR limit
- Automatic labels for categorization

---

### 5. Version Automation

**Status:** Complete (Manual Workflow)

Version management via git tags:

1. Update version in `pyproject.toml`
2. Commit and push
3. Create and push tag: `git tag v0.1.1 && git push origin v0.1.1`
4. CI automatically publishes to PyPI

**Note:** Version is defined in `pyproject.toml` and must match the git tag. The CI workflow only triggers publishing for tags matching `v*`.

---

## CI Pipeline Overview

```
push to main/PR
      │
      ├──► test (matrix: 3.9, 3.10, 3.11, 3.12)
      │     ├── ruff lint
      │     ├── ruff format check
      │     ├── mypy type check
      │     └── pytest with 85% coverage
      │
      ├──► security
      │     └── pip-audit vulnerability scan
      │
      └──► build (after test + security pass)
            ├── python -m build
            └── twine check

push tag v*
      │
      └──► publish (after build passes)
            └── PyPI trusted publishing (OIDC)
```

---

## Success Criteria

| Criterion | Status |
|-----------|--------|
| Package publishes to PyPI via GitHub release | ✅ v0.1.0 published |
| CI fails on known vulnerabilities | ✅ pip-audit in security job |
| 2FA enabled on PyPI | ✅ Required for trusted publishing |
| Security disclosure process documented | ✅ SECURITY.md |
| Automated dependency updates | ✅ Dependabot configured |

---

## Files Modified/Created

| File | Action |
|------|--------|
| `.github/workflows/ci.yml` | Pre-existing (131 lines) |
| `SECURITY.md` | Created (70 lines) |
| `.github/dependabot.yml` | Created |
| `agent_docs/sprint-5-cicd.md` | Created (this file) |

---

## Verification Commands

```bash
# Check recent CI runs
gh run list --limit 5

# Verify package on PyPI
pip index versions agentbible

# Check security locally
pip install pip-audit
pip-audit --desc on

# Verify trusted publishing setup
gh api repos/rylanmalarchick/research-code-principles/environments/pypi
```

---

## Notes

- The CI was already well-configured from initial package setup
- Sprint 5 was primarily about verification and documentation
- Dependabot will create weekly PRs for dependency updates
- Version bumps require manual update to `pyproject.toml` before tagging
