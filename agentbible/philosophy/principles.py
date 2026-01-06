"""The 5 Core Principles for AI-assisted research software.

These principles guide how AI agents should write scientific code:
1. Correctness First - Physical accuracy is non-negotiable
2. Specification Before Code - Tests define the contract
3. Fail Fast with Clarity - Detect errors at boundaries
4. Simplicity by Design - Simple code is correct code
5. Infrastructure Enables Speed - Invest in tooling
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Principle:
    """A core principle for research software development.

    Attributes:
        name: Short name for the principle
        summary: One-line summary
        description: Full description of what this means
        why_agents_fail: Common ways AI agents violate this principle
        how_to_recognize: Signs you're following this principle
    """

    name: str
    summary: str
    description: str
    why_agents_fail: str
    how_to_recognize: str


# =============================================================================
# The 5 Core Principles
# =============================================================================

CORRECTNESS_FIRST = Principle(
    name="Correctness First",
    summary="Physical accuracy and mathematical correctness are non-negotiable.",
    description=(
        "Research code must be *right* before it's fast, flexible, or elegant. "
        "A quantum gate that achieves 99.9% fidelity is worthless if the gate "
        "definition is wrong. An ML model with high accuracy is meaningless if "
        "the metric is miscalculated."
    ),
    why_agents_fail=(
        "Agents optimize for 'code compiles and runs' not 'code produces correct "
        "results.' They'll happily implement a rotation gate with the wrong sign, "
        "or calculate fidelity without normalization, because the code *works* - "
        "it just doesn't work *correctly*."
    ),
    how_to_recognize=(
        "Every non-trivial calculation has a test comparing against known values. "
        "Every physical quantity has bounds checking. Every equation references "
        "the paper/textbook it came from."
    ),
)

SPECIFICATION_BEFORE_CODE = Principle(
    name="Specification Before Code",
    summary="You can't write code until you define what 'correct' means.",
    description=(
        "Tests are not something you 'add later.' Tests *are* your specification. "
        "They define the contract: given these inputs, produce these outputs. "
        "Writing tests first forces you to think about edge cases before writing code."
    ),
    why_agents_fail=(
        "Agents default to feature-first development: 'write function, then add tests.' "
        "This leads to code that handles the case the agent thought about, but misses "
        "the 12 edge cases the agent didn't consider. Tests written after implementation "
        "are validation theater - they test what the code *does*, not what it *should do*."
    ),
    how_to_recognize=(
        "Test files exist before feature code. Edge cases are enumerated in test names. "
        "Parameterized tests cover boundaries. When you discover a bug, you write a test "
        "that fails, *then* fix the code."
    ),
)

FAIL_FAST_WITH_CLARITY = Principle(
    name="Fail Fast with Clarity",
    summary="Detect errors at boundaries and report them immediately with context.",
    description=(
        "Silent failures are the enemy of correctness. If a function receives invalid "
        "input, it must fail *now* with a clear message - not propagate garbage through "
        "the system and fail cryptically 10 function calls later."
    ),
    why_agents_fail=(
        "Agents often write 'optimistic code' that assumes inputs are valid. They skip "
        "input validation because 'the caller should handle that.' This leads to subtle "
        "bugs that are hard to trace back to the source."
    ),
    how_to_recognize=(
        "Every function validates its inputs at the boundary. Error messages include "
        "what went wrong, what was expected, and where it happened. Assertions check "
        "invariants that should never be false."
    ),
)

SIMPLICITY_BY_DESIGN = Principle(
    name="Simplicity by Design",
    summary="Simple code is correct code. Complex code hides bugs.",
    description=(
        "Research code should be readable by someone who didn't write it. This means: "
        "short functions (<=50 lines), clear names, minimal nesting, no clever tricks. "
        "If you need a comment to explain what code does, the code is too complex."
    ),
    why_agents_fail=(
        "Agents love to show off. They'll write a one-liner list comprehension when a "
        "simple for loop would be clearer. They'll use advanced language features when "
        "basic constructs would suffice. They optimize for 'impressiveness' over clarity."
    ),
    how_to_recognize=(
        "Functions do one thing. Names are descriptive. Control flow is linear. "
        "A new team member can understand the code without asking questions."
    ),
)

INFRASTRUCTURE_ENABLES_SPEED = Principle(
    name="Infrastructure Enables Speed",
    summary="Invest in tooling so you can move fast with confidence.",
    description=(
        "Good infrastructure (CI/CD, linting, formatting, testing) pays for itself. "
        "It catches errors before they become bugs. It enforces consistency. It lets "
        "you refactor with confidence because you know the tests will catch regressions."
    ),
    why_agents_fail=(
        "Agents skip infrastructure because it's 'not the feature.' They'll write code "
        "without tests, skip CI configuration, ignore linting errors. This leads to "
        "technical debt that compounds over time."
    ),
    how_to_recognize=(
        "CI runs on every push. Pre-commit hooks enforce formatting. Tests run fast "
        "and reliably. You can deploy/publish with a single command."
    ),
)


@dataclass(frozen=True)
class PrinciplesNamespace:
    """Namespace containing all 5 core principles."""

    CORRECTNESS_FIRST: Principle = CORRECTNESS_FIRST
    SPECIFICATION_BEFORE_CODE: Principle = SPECIFICATION_BEFORE_CODE
    FAIL_FAST_WITH_CLARITY: Principle = FAIL_FAST_WITH_CLARITY
    SIMPLICITY_BY_DESIGN: Principle = SIMPLICITY_BY_DESIGN
    INFRASTRUCTURE_ENABLES_SPEED: Principle = INFRASTRUCTURE_ENABLES_SPEED


# Singleton instance for convenient access
PRINCIPLES = PrinciplesNamespace()


def get_all_principles() -> list[Principle]:
    """Return all 5 principles as a list."""
    return [
        CORRECTNESS_FIRST,
        SPECIFICATION_BEFORE_CODE,
        FAIL_FAST_WITH_CLARITY,
        SIMPLICITY_BY_DESIGN,
        INFRASTRUCTURE_ENABLES_SPEED,
    ]


def get_agent_context(
    topics: list[str] | None = None,
    include_all: bool = False,
) -> str:
    """Generate context text for AI agent sessions.

    Args:
        topics: List of topics to include. Options:
            - "correctness", "test-first", "validation", "simplicity", "infrastructure"
            - If None and include_all is False, returns core summary
        include_all: If True, include all principles in full detail

    Returns:
        Formatted markdown text suitable for AI agent context window.

    Example:
        >>> context = get_agent_context(topics=["test-first", "validation"])
        >>> # Use this in your AI session context
    """
    lines = ["# AgentBible Core Principles\n"]

    if include_all:
        principles = get_all_principles()
    elif topics:
        topic_map = {
            "correctness": CORRECTNESS_FIRST,
            "test-first": SPECIFICATION_BEFORE_CODE,
            "specification": SPECIFICATION_BEFORE_CODE,
            "validation": FAIL_FAST_WITH_CLARITY,
            "fail-fast": FAIL_FAST_WITH_CLARITY,
            "simplicity": SIMPLICITY_BY_DESIGN,
            "infrastructure": INFRASTRUCTURE_ENABLES_SPEED,
            "ci": INFRASTRUCTURE_ENABLES_SPEED,
            "cicd": INFRASTRUCTURE_ENABLES_SPEED,
        }
        principles = []
        for topic in topics:
            topic_lower = topic.lower().replace("_", "-")
            if topic_lower in topic_map:
                p = topic_map[topic_lower]
                if p not in principles:
                    principles.append(p)
    else:
        # Just return summaries
        lines.append("Quick reference for AI-assisted research coding:\n")
        for i, p in enumerate(get_all_principles(), 1):
            lines.append(f"{i}. **{p.name}**: {p.summary}")
        return "\n".join(lines)

    # Full format for selected principles
    for p in principles:
        lines.append(f"## {p.name}\n")
        lines.append(f"**{p.summary}**\n")
        lines.append(p.description + "\n")
        lines.append(f"**Why agents fail:** {p.why_agents_fail}\n")
        lines.append(f"**How to recognize success:** {p.how_to_recognize}\n")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "Principle",
    "PRINCIPLES",
    "CORRECTNESS_FIRST",
    "SPECIFICATION_BEFORE_CODE",
    "FAIL_FAST_WITH_CLARITY",
    "SIMPLICITY_BY_DESIGN",
    "INFRASTRUCTURE_ENABLES_SPEED",
    "get_all_principles",
    "get_agent_context",
]
