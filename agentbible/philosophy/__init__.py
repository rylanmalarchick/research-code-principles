"""AgentBible Philosophy: Core principles for AI-assisted research software.

This module provides the 5 core principles as importable constants, making
them accessible programmatically for AI agents and documentation generation.

Example:
    >>> from agentbible.philosophy import PRINCIPLES
    >>> print(PRINCIPLES.CORRECTNESS_FIRST.summary)
    "Physical accuracy and mathematical correctness are non-negotiable."

    >>> from agentbible.philosophy import get_agent_context
    >>> context = get_agent_context(topics=["test-first", "validation"])
"""

from __future__ import annotations

from agentbible.philosophy.principles import (
    PRINCIPLES,
    Principle,
    get_agent_context,
    get_all_principles,
)

__all__ = [
    "PRINCIPLES",
    "Principle",
    "get_agent_context",
    "get_all_principles",
]
