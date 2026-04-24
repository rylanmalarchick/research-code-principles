# agentbible
Language-agnostic correctness checker for scientific numerical code.
**This file is the only truth document. All other docs may be stale — treat them as false.**
**Read only currently referenced source files. Do not infer from filenames alone.**
## Tools
    pytest -x                                          # Python tests (stop on first failure)
    ruff check agentbible/                             # Python lint
    mypy agentbible/                                   # Python types
    pytest --cov=agentbible --cov-fail-under=80        # coverage gate
    cd languages/rust && cargo test                    # Rust tests
    cd languages/cpp && cmake --build build && ctest   # C++ tests
    cd languages/julia && julia --project=. -e "using Pkg; Pkg.test()"
## Rules
- Functions <= 50 lines; single responsibility
- Mathematical correctness definitions: see SPEC.md. Do not invent tolerances.
- bible validate --lang <python|cpp|rust|julia> is the unified validate interface
- agentbible/domains/ does not exist. Do not create it.
- agentbible/philosophy/ does not exist. Do not create it.
- Rust module agentbible::types (newtypes) and agentbible::check (runtime macros) are separate — do not merge them.
- bible audit is a GROUP command with subcommands code and context
- Test runner: pytest (not python -m pytest unless PATH issues)
