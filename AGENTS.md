# agentbible

Research code standards framework — validators, provenance, CLI scaffolding.

## Tools

```
pytest -x                                     # tests (stop on first failure)
ruff check agentbible/                        # lint
mypy agentbible/                              # types
pytest --cov=agentbible --cov-fail-under=75  # coverage
bible audit code agentbible/                 # agentbible code audit
```

## Rules

- Functions ≤ 50 lines
- `audit` CLI command is a GROUP: subcommands are `code` and `context`
- Test runner: `/usr/bin/pytest` (system python3 lacks deps)
- Custom markers: `slow`, `integration`
