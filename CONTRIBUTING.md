# Contributing

This is a living document. Help improve it.

## Types of Contributions

### 1. Bug Reports / Clarifications
Found an unclear section? Confusing example? Open an issue.

### 2. Better Prompting Patterns
Discovered a prompt that works better? Add it to `docs/prompting-research-code.md`.

Format:
```
## Pattern: [Name]

**When to use:** [Situation]

**Prompt:**
[Exact prompt]

**Result:** [What worked, what didn't]

**Why it works:** [Explanation]
```

### 3. New Examples
Implemented something following these principles? Create `examples/[domain]-[task]/`.

Include:
- Source code (≤200 lines)
- Full test suite
- README explaining design
- `prompting-log.md` showing how you prompted for it

### 4. Improvements to Standards
- "This C++ standard doesn't work for X, suggest Y"
- "Added validation pattern that prevents Z bug"

## Before Submitting
- [ ] Is it applicable widely? (Not just your project)
- [ ] Is it correct? (Tested on real code)
- [ ] Is it clear? (Examples provided)
- [ ] Is it research-grade? (Follows the 5 principles)

## Discussion
Start in **Issues** before PRs for big changes.

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(examples): add molecular dynamics example
fix(docs): correct rotation gate formula
docs: update prompting patterns
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
