# Prompting Log: Quantum Gate Example

This document shows how the `gate.py` module was developed using AI-assisted coding, following the research code principles. It includes a realistic simulated session and templates for future use.

## Session: Building gate.py

### Initial Context Provided

```
I'm building a quantum gate representation for a circuit simulator.

Requirements:
- Represent common gates: X, Z, H, RZ (rotation), CNOT
- Generate unitary matrices for simulation
- Validate inputs (qubit indices, rotation angles)
- This is research code - correctness matters more than performance

Physics context:
- Quantum gates are unitary matrices acting on qubit state vectors
- X, Z, H are 2x2 matrices; CNOT is 4x4
- RZ(angle) is a parametric rotation gate
- Reference: Nielsen & Chuang, Chapter 4

Please create a Gate class following these principles:
1. Correctness first - validate everything
2. Simple over clever - readable code
3. Inspectable - good repr, logging
4. Fail fast - errors at construction, not use
5. Reproducible - deterministic, well-documented
```

### Prompt 1: Core Design

```
Create a Gate class as a frozen dataclass with:
- gate_type: str (e.g., "X", "Z", "H", "RZ", "CNOT")
- qubits: tuple[int, ...] for qubit indices
- parameter: Optional[float] for rotation angles

Add validation in __post_init__:
- Qubits must be non-negative integers
- No duplicate qubits
- Parameter must be finite (no NaN/inf)

Use factory methods for clean construction:
- Gate.x(qubit) -> X gate
- Gate.z(qubit) -> Z gate
- Gate.h(qubit) -> Hadamard
- Gate.rz(qubit, angle) -> Z rotation
- Gate.cnot(control, target) -> controlled-NOT

Include docstrings with Nielsen & Chuang references.
```

**Result**: Agent generated the Gate class with validation and factories.

**Review notes**:
- Added explicit error for CNOT with same control/target
- Improved error messages to include the actual invalid value
- Added type hints for all parameters

### Prompt 2: Matrix Generation

```
Add to_matrix() method that returns numpy arrays:

X = [[0, 1], [1, 0]]
Z = [[1, 0], [0, -1]]
H = (1/sqrt(2)) * [[1, 1], [1, -1]]
RZ(θ) = [[exp(-iθ/2), 0], [0, exp(iθ/2)]]
CNOT = [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]

After generating each matrix, verify unitarity:
- Compute U @ U.conj().T
- Check max deviation from identity
- Raise RuntimeError if > 1e-10

Add logging for debugging.
```

**Result**: Matrix generation with unitarity verification.

**Review notes**:
- Changed to use assert for RZ parameter (guaranteed by factory)
- Made tolerance a module constant
- Added context to RuntimeError message

### Prompt 3: Tests

```
Write pytest tests for gate.py covering:

1. TestGateConstruction - factory methods work correctly
2. TestInvalidInputs - bad inputs raise with clear messages
   - negative qubit, NaN angle, inf angle
   - CNOT same qubit, duplicate qubits, empty qubits
3. TestMatrixGeneration - matrices match textbook values
4. TestEdgeCases - boundary values
   - RZ(0) = identity
   - RZ(2π) = -identity (global phase)
   - negative angles
   - large qubit indices
5. TestPhysicalProperties - quantum mechanics is correct
   - X² = I, Z² = I, H² = I
   - All gates are unitary
   - XZ = -ZX (anticommutation)
   - H|0> creates superposition

Use numpy.testing.assert_array_almost_equal for matrix comparisons.
```

**Result**: Comprehensive test suite with 25+ tests.

**Review notes**:
- Added test for custom tolerance in is_unitary
- Added test for unknown gate type
- Grouped related tests into clear classes

---

## Templates for Future Sessions

### Template 1: New Module from Scratch

```
I'm implementing [MODULE_NAME] for [PROJECT_CONTEXT].

Requirements:
- [Functional requirement 1]
- [Functional requirement 2]
- [Functional requirement 3]

Physics/Domain context:
- [Key concept 1]
- [Key concept 2]
- Reference: [Paper/textbook citation]

Apply these principles:
1. Correctness first - validate all inputs
2. Simple over clever - prioritize readability
3. Inspectable - good repr, logging for debugging
4. Fail fast - errors at construction not use
5. Reproducible - deterministic, documented

Start with the core data structure and validation.
```

### Template 2: Adding a Feature

```
Add [FEATURE_NAME] to [EXISTING_MODULE].

Current state:
- [What exists]
- [Relevant code patterns already used]

New requirements:
- [What the feature should do]
- [Edge cases to handle]
- [Physical constraints to enforce]

Follow the existing patterns for:
- Validation style
- Error message format
- Docstring format with citations
- Test organization

Include tests for happy path, edge cases, and invalid inputs.
```

### Template 3: Writing Tests

```
Write tests for [MODULE_NAME] covering:

1. Happy paths
   - [Normal use case 1]
   - [Normal use case 2]

2. Edge cases
   - [Boundary value 1]
   - [Boundary value 2]
   - [Limit case]

3. Invalid inputs (should raise)
   - [Invalid input type 1] -> [Expected error]
   - [Invalid input type 2] -> [Expected error]

4. Physical/domain properties
   - [Invariant 1]
   - [Invariant 2]

Use pytest. Group related tests into classes.
Match error messages with pytest.raises(Error, match="...").
```

### Template 4: Review Request

```
Review this code for research code principles:

[PASTE CODE]

Check for:
1. Input validation - are all inputs checked?
2. Error messages - do they include context?
3. Fail fast - do errors happen early?
4. Reproducibility - any hidden state or randomness?
5. Documentation - are citations included?
6. Testability - is this easy to test?

Suggest specific improvements with code examples.
```

---

## Key Lessons from This Session

1. **Front-load context**: The initial prompt included physics background, which prevented incorrect implementations.

2. **Request validation explicitly**: Without asking for validation, the agent might skip it or implement it poorly.

3. **Specify error handling style**: "Errors at construction, not use" guided the design toward early validation.

4. **Include citations in prompts**: Mentioning "Nielsen & Chuang, Chapter 4" ensured docstrings included proper references.

5. **Iterate in focused chunks**: Separate prompts for structure, implementation, and tests kept each response focused.

6. **Review everything**: Each agent response was reviewed and refined. The agent got 80% right; the 20% required domain expertise.
