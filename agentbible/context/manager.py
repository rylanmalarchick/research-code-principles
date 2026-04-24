"""Source-tree context retrieval across the AgentBible language targets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentbible.context.config import ContextConfig


@dataclass
class RetrievedChunk:
    """A ranked chunk of source context."""

    content: str
    source: str
    section: str
    line_start: int
    line_end: int
    doc_type: str
    always_include: bool
    description: str
    distance: float
    chunk_index: int
    total_chunks: int

    @property
    def score(self) -> float:
        """Convert the stored distance into a similarity score."""
        return max(0.0, 1.0 - self.distance)

    def format_reference(self) -> str:
        """Format a compact source reference."""
        return f"{Path(self.source).name}:{self.line_start}-{self.line_end}"


SOURCE_ROOTS = {
    "python": [Path("agentbible")],
    "cpp": [Path("languages/cpp/include")],
    "rust": [Path("languages/rust/src"), Path("languages/rust/agentbible/src")],
    "julia": [Path("languages/julia/src")],
}

SOURCE_SUFFIXES = {
    "python": {".py"},
    "cpp": {".hpp", ".h", ".cpp"},
    "rust": {".rs"},
    "julia": {".jl"},
}


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z_]+", text.lower()) if token]


def _score_chunk(query_tokens: list[str], content: str, source: Path) -> int:
    haystack = content.lower()
    filename = source.name.lower()
    return sum(haystack.count(token) + (2 * filename.count(token)) for token in query_tokens)


def _chunk_source(path: Path, language: str, chunk_size: int = 40) -> list[RetrievedChunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks: list[RetrievedChunk] = []
    total = max(1, (len(lines) + chunk_size - 1) // chunk_size)
    for index, start in enumerate(range(0, len(lines), chunk_size)):
        end = min(start + chunk_size, len(lines))
        chunks.append(
            RetrievedChunk(
                content="\n".join(lines[start:end]),
                source=str(path),
                section=path.name,
                line_start=start + 1,
                line_end=end,
                doc_type=language,
                always_include=False,
                description=f"{language} source",
                distance=1.0,
                chunk_index=index,
                total_chunks=total,
            )
        )
    return chunks


class ContextManager:
    """Lightweight source-code retrieval for the current repository."""

    def __init__(self, config: ContextConfig | None = None):
        self.config = config
        self.repo_root = Path(__file__).resolve().parents[2]

    def load_directory(self, directory: str | Path) -> str:
        """Load markdown files from a directory for compatibility."""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        md_files = sorted(dir_path.glob("**/*.md"))
        parts = [f"# Context from {dir_path.name}\n", f"**Files loaded:** {len(md_files)}\n", "---\n"]
        for md_file in md_files:
            parts.append(f"\n## Source: {md_file.relative_to(dir_path)}\n")
            parts.append(md_file.read_text(encoding="utf-8"))
            parts.append("\n---\n")
        return "".join(parts)

    def query(
        self,
        query: str,
        project_name: str | None = None,
        top_k: int | None = None,
        include_global: bool = True,
        lang: str = "all",
    ) -> list[RetrievedChunk]:
        """Query the repository source trees and return ranked chunks."""
        del project_name, include_global
        query_tokens = _tokenize(query)
        languages = SOURCE_ROOTS.keys() if lang == "all" else [lang]
        ranked: list[tuple[int, RetrievedChunk]] = []
        for language in languages:
            for root in SOURCE_ROOTS.get(language, []):
                absolute = self.repo_root / root
                if not absolute.exists():
                    continue
                for path in absolute.rglob("*"):
                    if path.suffix not in SOURCE_SUFFIXES[language]:
                        continue
                    for chunk in _chunk_source(path, language):
                        score = _score_chunk(query_tokens, chunk.content, path)
                        if score <= 0:
                            continue
                        chunk.distance = 1.0 / (score + 1.0)
                        ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].source, item[1].line_start))
        limit = top_k or 5
        return [chunk for _, chunk in ranked[:limit]]

    def build_context(
        self,
        query: str | None = None,
        project_name: str | None = None,
        all_from_dir: str | Path | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        lang: str = "all",
    ) -> str:
        """Build a formatted context string."""
        del project_name, max_tokens
        if all_from_dir:
            return self.load_directory(all_from_dir)
        if not query:
            return "# Context\n\nUse --query to search the repository source trees."
        chunks = self.query(query=query, top_k=top_k, lang=lang)
        if not chunks:
            return "# Context\n\nNo relevant source chunks found."
        parts = ["# Context\n", f"**Query:** {query}\n", f"**Language filter:** {lang}\n", "---\n"]
        for chunk in chunks:
            parts.append(
                f"\n## {chunk.doc_type}: {Path(chunk.source).relative_to(self.repo_root)}"
                f" ({chunk.line_start}-{chunk.line_end})\n"
            )
            parts.append(chunk.content)
            parts.append("\n")
        return "".join(parts)

    def embed_directory(self, directory: str | Path, collection_name: str = "local_docs") -> int:
        """Compatibility shim for the old CLI surface."""
        del collection_name
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        return len(list(dir_path.rglob("*")))
