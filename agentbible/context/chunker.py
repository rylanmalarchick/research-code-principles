"""Document chunking for AgentBible Context.

Handles intelligent chunking of markdown documents for embedding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    """A chunk of document content with metadata."""

    content: str
    source: str
    doc_type: str
    section: str
    line_start: int
    line_end: int
    always_include: bool
    description: str
    chunk_index: int
    total_chunks: int

    def to_metadata(self) -> dict[str, Any]:
        """Convert to Chroma metadata dict."""
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "section": self.section,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "always_include": self.always_include,
            "description": self.description,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }

    def get_id(self) -> str:
        """Generate unique ID for this chunk."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{Path(self.source).stem}_{self.line_start}_{content_hash}"


class DocumentChunker:
    """Chunks markdown documents intelligently.

    Uses tiktoken for accurate token counting when available,
    falls back to character-based estimation otherwise.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder: Any = None

    @property
    def encoder(self) -> Any:
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken  # type: ignore[import-not-found]

                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                # Fall back to character-based estimation
                self._encoder = None
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder is not None:
            return len(self.encoder.encode(text))
        # Fallback: estimate ~4 chars per token
        return len(text) // 4

    def chunk_document(
        self,
        content: str,
        source: str,
        doc_type: str = "reference",
        always_include: bool = False,
        description: str = "",
    ) -> list[Chunk]:
        """Split document into chunks, preserving markdown structure."""
        lines = content.split("\n")
        chunks: list[Chunk] = []

        current_section = "Introduction"
        current_chunk_lines: list[tuple[int, str]] = []
        current_tokens = 0

        for line_num, line in enumerate(lines, start=1):
            # Track section headers
            if line.startswith("## "):
                current_section = line[3:].strip()
            elif line.startswith("# ") and not line.startswith("##"):
                current_section = line[2:].strip()

            line_tokens = self.count_tokens(line + "\n")

            # Check if adding this line would exceed chunk size
            if current_tokens + line_tokens > self.chunk_size and current_chunk_lines:
                # Save current chunk
                chunk = self._create_chunk(
                    current_chunk_lines,
                    source,
                    doc_type,
                    current_section,
                    always_include,
                    description,
                    len(chunks),
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk_lines)
                current_chunk_lines = overlap_lines
                current_tokens = sum(
                    self.count_tokens(ln + "\n") for _, ln in current_chunk_lines
                )

            current_chunk_lines.append((line_num, line))
            current_tokens += line_tokens

        # Don't forget the last chunk
        if current_chunk_lines:
            chunk = self._create_chunk(
                current_chunk_lines,
                source,
                doc_type,
                current_section,
                always_include,
                description,
                len(chunks),
            )
            chunks.append(chunk)

        # Update total_chunks in all chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _create_chunk(
        self,
        lines: list[tuple[int, str]],
        source: str,
        doc_type: str,
        section: str,
        always_include: bool,
        description: str,
        chunk_index: int,
    ) -> Chunk:
        """Create a Chunk from lines."""
        content = "\n".join(ln for _, ln in lines)
        line_start = lines[0][0]
        line_end = lines[-1][0]

        return Chunk(
            content=content,
            source=source,
            doc_type=doc_type,
            section=section,
            line_start=line_start,
            line_end=line_end,
            always_include=always_include,
            description=description,
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated later
        )

    def _get_overlap_lines(self, lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """Get lines for overlap from the end of current chunk."""
        overlap_tokens = 0
        overlap_lines: list[tuple[int, str]] = []

        for line_num, line in reversed(lines):
            line_tokens = self.count_tokens(line + "\n")
            if overlap_tokens + line_tokens > self.chunk_overlap:
                break
            overlap_lines.insert(0, (line_num, line))
            overlap_tokens += line_tokens

        return overlap_lines
