# Agent Prompts Directory

Modular prompt snippets for AI-assisted research coding sessions. Load only what you need to reduce token usage.

## Usage

Instead of loading the entire `docs/agent-coding-context.md` (500+ lines), load specific prompts for your task:

```bash
# For test generation
cat agent_prompts/test-generation.md

# For physics validation
cat agent_prompts/physics-validation.md

# For kernel optimization
cat agent_prompts/kernel-optimization.md
```

## Available Prompts

| File | Purpose | Lines |
|------|---------|-------|
| `core-principles.md` | The 5 core principles (minimal version) | ~50 |
| `test-generation.md` | Writing tests for research code | ~40 |
| `physics-validation.md` | Quantum/physics constraint checking | ~40 |
| `kernel-optimization.md` | CUDA kernel best practices | ~40 |
| `code-review.md` | Quick review checklist | ~30 |
| `error-handling.md` | Fail-fast patterns | ~30 |

## Combining Prompts

For a full session, combine prompts:

```bash
cat agent_prompts/core-principles.md agent_prompts/test-generation.md > session_context.md
```

Or use the context mapper:

```bash
./scripts/map_repo.sh  # Generate repo map
cat repo_map.txt agent_prompts/core-principles.md > session_context.md
```
