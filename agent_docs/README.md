# Agent Docs

**Meta-documentation for developing the research-code-principles repository.**

These docs are embedded in the vector database and used with `oc-context` when working on the repo itself. This is dogfooding - we use our own tools to build our tools.

## Usage

```bash
# Load all sprint docs for an AI session
oc-context --all ./agent_docs

# Or with the future CLI
bible context --all ./agent_docs
```

## Files

| File | Purpose |
|------|---------|
| `SCOPE_OF_WORK.md` | Overall vision, sprint overview, technical decisions |
| `sprint-1-foundation.md` | Sprint 1: bootstrap.sh, .cursorrules, .devcontainer |
| `ARCHITECTURE.md` | Target directory structure for v3.0 |

## Adding New Sprints

When starting a new sprint:

1. Create `sprint-N-name.md` following the existing format
2. Update `SCOPE_OF_WORK.md` sprint table
3. Run `oc-update` to embed the new docs
4. Mark previous sprint as complete

## Format

Each sprint doc should include:

- **Objective** - What we're trying to achieve
- **Deliverables** - Specific files/features to create
- **Tasks** - Checklist of work items
- **Acceptance Criteria** - How we know it's done
- **Testing Plan** - How to verify it works
