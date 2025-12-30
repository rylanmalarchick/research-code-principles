# Code Review Checklist

Quick checklist for reviewing research code.

---

## Correctness

- [ ] Code matches specification/requirements
- [ ] Physical quantities validated against known values
- [ ] References primary sources (papers, textbooks)
- [ ] Numerical stability considered
- [ ] Physical constraints enforced (unitarity, normalization)

---

## Testing

- [ ] Tests written before or alongside code
- [ ] Happy path tested
- [ ] Edge cases tested (empty, single, max, zero)
- [ ] Invalid inputs tested (wrong type, out of range, NaN)
- [ ] Tests are deterministic (fixed seeds)
- [ ] Tests run in CI

---

## Error Handling

- [ ] Inputs validated at function boundaries
- [ ] Errors fail fast with descriptive messages
- [ ] Error messages include: what, expected, where
- [ ] Return values checked (especially C/CUDA APIs)
- [ ] No silent failures

---

## Simplicity

- [ ] Each function does one thing
- [ ] Functions are short (≤60 lines C++, ≤50 lines Python)
- [ ] Classes have single responsibility
- [ ] No premature abstraction
- [ ] No unnecessary dependencies

---

## Style

- [ ] Naming follows conventions (PascalCase classes, snake_case functions)
- [ ] Code is formatted (clang-format, ruff)
- [ ] No magic numbers (constants defined and documented)
- [ ] Comments explain "why" not "what"
- [ ] Docstrings on all public APIs

---

## Infrastructure

- [ ] CI passes (tests, linting, type checking)
- [ ] Zero warnings
- [ ] Documentation updated
- [ ] Dependencies pinned
- [ ] Seeds documented for reproducibility
