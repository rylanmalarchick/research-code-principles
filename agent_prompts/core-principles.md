# Core Principles (Minimal Version)

Load this into every AI coding session. ~50 lines for token efficiency.

---

## The 5 Principles

1. **Correctness First** — Physical accuracy is non-negotiable. Validate against known values.
2. **Specification Before Code** — Tests define the contract. Write them first.
3. **Fail Fast with Clarity** — Validate inputs at boundaries. Descriptive errors.
4. **Simplicity by Design** — Functions ≤50 lines. Single responsibility.
5. **Infrastructure Enables Speed** — CI, tests, linting from day one.

---

## You MUST

- Write or describe test cases BEFORE implementing
- Validate all inputs at function boundaries
- Cite sources for physics equations and algorithms
- Check physical constraints (unitarity, normalization, bounds)
- Keep functions short and focused (≤50 lines Python, ≤60 lines C++)
- Ensure reproducibility (set seeds, pin dependencies)

## You MUST NOT

- Skip tests "to save time"
- Assume inputs are valid
- Implement algorithms without citing source
- Write functions > 60 lines
- Use magic numbers without documentation
- Ignore warnings

---

## Quick Checklist

Before marking any task complete:

- [ ] All tests pass?
- [ ] Zero warnings?
- [ ] Every public function has docstring?
- [ ] Edge cases tested (empty, single, max, invalid)?
- [ ] Physical constraints validated?

---

**Full context:** See `docs/agent-coding-context.md`
