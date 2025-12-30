# Research Code Principles

**Research code doesn't have to be garbage.**

A complete system for building production-grade research software using AI agents.

## The Problem

Research code defaults to being terrible:
- No tests
- No validation
- No reproducibility
- Results can't be trusted

**Why?** Scientists and AI agents both optimize for "code that compiles," not "code that's correct."

## The Solution

**5 Core Principles:**
1. **Correctness First** — Physical accuracy is non-negotiable
2. **Specification Before Code** — Tests define the contract
3. **Fail Fast with Clarity** — Validate at boundaries, error immediately
4. **Simplicity by Design** — Each function does one thing
5. **Infrastructure Enables Speed** — Tests, CI/CD, and tooling aren't overhead

## Who This Is For

- PhD students learning to code scientifically
- Researchers pairing Claude/ChatGPT with their work
- Labs wanting to enforce quality without micromanaging
- Anyone tired of inheriting garbage research code

## Quick Start

1. **Understand why:** Read `docs/research-code-principles.md` (15 min)
2. **Reference standards:** Use `docs/style-guide-reference.md` when coding
3. **Prompt AI agents:** Use `docs/agent-coding-context.md` + `docs/prompting-research-code.md`
4. **See examples:** Check `examples/quantum-gate-example/`

## Key Files

| File | Purpose |
|------|---------|
| `research-code-principles.md` | Philosophy: why good code matters |
| `style-guide-reference.md` | Standards: C++, Python, CMake, testing |
| `repo-standards.md` | Workflow: git, CI/CD, infrastructure |
| `agent-coding-context.md` | Paste into Claude for coding sessions |
| `prompting-research-code.md` | How to prompt for research-grade code |

## Tools

### OpenCode Context Manager

**Reduce token usage by 60-70%** when loading documentation into AI sessions.

Instead of loading entire 50k+ token doc files, use vector search to load only relevant chunks:

```bash
# Load all docs from a directory (~20k tokens instead of 60k)
oc-context --all ./agent_docs

# Semantic search for specific topics
oc-context --query "error handling validation"
```

See [`opencode-context/README.md`](opencode-context/README.md) for installation and usage.

## Using With Claude/ChatGPT

1. Paste `docs/agent-coding-context.md` into your session
2. Reference `docs/prompting-research-code.md` for patterns
3. Provide one working example (few-shot)
4. Use Chain of Thought: "Think step by step about edge cases, then write tests, then implement"

## Examples

- `examples/quantum-gate-example/` — Full quantum gate implementation with tests

## Status

- ✅ Philosophy complete
- ✅ Detailed standards complete
- ✅ Prompt engineering guide complete
- ✅ Quantum gate example complete
- ✅ OpenCode Context Manager (vector-based context retrieval)
- 🔄 Community examples (accepting contributions)

## License

MIT — Use and adapt freely.

## Contributing

Found issues? Better prompts? New examples? See `CONTRIBUTING.md`.

---

**Latest:** v1.0 (Dec 2025)  
**Author:** Rylan Malarchick  
**Contact:** rylan1012@gmail.com
