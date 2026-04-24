# AgentBible Philosophy

AgentBible exists to make scientific numerical code fail loudly when it is wrong and to leave a reproducible record when it is right.

## Core Principles

1. Correctness first. Numerical and mathematical validity come before convenience or speed.
2. Specification before code. Define checks and edge cases before trusting implementation.
3. Fail fast with clarity. Invalid inputs and invalid results should raise immediately with enough context to debug.
4. Simplicity by design. Small functions and narrow responsibilities reduce hidden numerical bugs.
5. Infrastructure enables speed. Tests, linting, type checks, and provenance records are acceleration tools, not overhead.

## Practical Reading

- Use `SPEC.md` for the exact mathematical definitions.
- Use `schema/provenance_v1.json` for the cross-language provenance contract.
- Treat README and other docs as secondary to those two files.
