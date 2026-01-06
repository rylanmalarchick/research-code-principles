"""AgentBible Context Module.

Provides vector-based document retrieval for AI coding sessions.
This module enables semantic search over project documentation,
allowing AI agents to retrieve relevant context based on queries.

Example usage:
    from agentbible.context import ContextManager

    # Load context for current project
    ctx = ContextManager()
    context = ctx.build_context(query="error handling patterns")
    print(context)

    # Load all docs from a directory
    context = ctx.build_context(all_from_dir="./agent_docs")

Dependencies:
    Core: pyyaml, tiktoken
    Full: chromadb, openai (or sentence-transformers for local embeddings)

Install with: pip install agentbible[context]
"""

from __future__ import annotations

from .chunker import Chunk, DocumentChunker
from .config import ContextConfig, DocConfig, ProjectConfig
from .manager import ContextManager

__all__ = [
    "ContextManager",
    "ContextConfig",
    "ProjectConfig",
    "DocConfig",
    "DocumentChunker",
    "Chunk",
]
