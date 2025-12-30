"""
OpenCode Context Manager Library

Provides vector-based document retrieval for OpenCode sessions.
"""

from .config import Config, OPENCODE_DIR, CONFIG_PATH, VECTORDB_DIR
from .embed import DocumentChunker, EmbeddingManager, Chunk
from .retrieve import ContextRetriever, RetrievedChunk

__all__ = [
    "Config",
    "OPENCODE_DIR",
    "CONFIG_PATH",
    "VECTORDB_DIR",
    "DocumentChunker",
    "EmbeddingManager",
    "Chunk",
    "ContextRetriever",
    "RetrievedChunk",
]
__version__ = "1.0.0"
