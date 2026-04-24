# AgentBible C++

Header-only C++ validation helpers for the AgentBible v1 correctness specification.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Macros

- `AGENTBIBLE_VALIDATE_FINITE(ptr, n)`
- `AGENTBIBLE_VALIDATE_POSITIVE(ptr, n)`
- `AGENTBIBLE_VALIDATE_NON_NEGATIVE(ptr, n)`
- `AGENTBIBLE_VALIDATE_PROBABILITY(ptr, n)`
- `AGENTBIBLE_VALIDATE_NORMALIZED_L1(ptr, n, atol)`
- `AGENTBIBLE_VALIDATE_SYMMETRIC(mat, n, atol)`
- `AGENTBIBLE_VALIDATE_UNITARY(mat, n, rtol, atol)`
- `AGENTBIBLE_VALIDATE_POSITIVE_DEFINITE(mat, n)`

Define `AGENTBIBLE_DISABLE` to compile the checks out entirely. Set
`AGENTBIBLE_ON_FAIL=ABORT` for abort-on-failure HPC builds; the default mode throws.
