# AgentBible

Language-agnostic correctness checker for scientific numerical code.

[SPEC.md](SPEC.md) is the mathematical source of truth for every implementation in this repository.

## Install

| Language | Command |
| --- | --- |
| Python | `pip install agentbible` |
| Rust | `cargo add agentbible` |
| Julia | `julia -e 'using Pkg; Pkg.add("AgentBible")'` |
| C++ | `CPMAddPackage("gh:rylanmalarchick/research-code-principles@cpp-v0.1.0")` |

## Python Example

Without validation, a normalization bug can quietly propagate:

```python
import numpy as np


def probabilities(logits: np.ndarray) -> np.ndarray:
    return np.exp(logits)
```

With AgentBible, the same bug fails at the boundary where it matters:

```python
import numpy as np
from agentbible import validate_finite, validate_normalized_l1, validate_probabilities


@validate_finite
@validate_probabilities
@validate_normalized_l1()
def probabilities(logits: np.ndarray) -> np.ndarray:
    return np.exp(logits)
```

The invalid output raises immediately instead of silently contaminating later results.

## CLI

Validate Python-produced data directly:

```bash
bible validate results.npy --check normalized_l1
```

Inspect a non-Python provenance record through the unified interface:

```bash
bible validate --lang rust results.json
```

Generate a Markdown provenance summary:

```bash
bible report results.json
```

## Repository Layout

- `agentbible/`: Python package, CLI, provenance, context retrieval
- `languages/rust/`: Rust workspace
- `languages/cpp/`: header-only C++ library
- `languages/julia/`: Julia package
- `schema/provenance_v1.json`: cross-language provenance schema
