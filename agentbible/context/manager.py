"""Context Manager for AgentBible.

High-level API for retrieving relevant context for AI coding sessions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import DocumentChunker
from .config import ContextConfig, get_vectordb_path


@dataclass
class RetrievedChunk:
    """A retrieved chunk with relevance score."""

    content: str
    source: str
    section: str
    line_start: int
    line_end: int
    doc_type: str
    always_include: bool
    description: str
    distance: float  # Lower is more similar
    chunk_index: int
    total_chunks: int

    @property
    def score(self) -> float:
        """Convert distance to similarity score (0-1, higher is better)."""
        # Cosine distance ranges from 0 (identical) to 2 (opposite)
        # Convert to similarity: 1 - (distance / 2)
        return max(0.0, 1.0 - (self.distance / 2.0))

    @classmethod
    def from_query_result(
        cls,
        content: str,
        metadata: dict[str, Any],
        distance: float,
    ) -> RetrievedChunk:
        """Create from ChromaDB query result."""
        return cls(
            content=content,
            source=str(metadata.get("source", "")),
            section=str(metadata.get("section", "")),
            line_start=int(metadata.get("line_start", 0)),
            line_end=int(metadata.get("line_end", 0)),
            doc_type=str(metadata.get("doc_type", "reference")),
            always_include=bool(metadata.get("always_include", False)),
            description=str(metadata.get("description", "")),
            distance=distance,
            chunk_index=int(metadata.get("chunk_index", 0)),
            total_chunks=int(metadata.get("total_chunks", 1)),
        )

    def format_reference(self) -> str:
        """Format source reference for context output."""
        source_name = Path(self.source).name
        return f"{source_name}:{self.line_start}-{self.line_end}"


class ContextManager:
    """High-level API for context retrieval.

    This class provides a simple interface for:
    - Loading documents from directories
    - Embedding documents into a vector database
    - Querying for relevant context
    - Building formatted context strings for AI sessions

    Example:
        >>> ctx = ContextManager()
        >>> context = ctx.build_context(query="error handling")
        >>> print(context)

        >>> # Load all docs from a directory
        >>> context = ctx.load_directory("./agent_docs")
    """

    def __init__(self, config: ContextConfig | None = None):
        """Initialize context manager.

        Args:
            config: Configuration to use. If None, loads from default path.
        """
        self.config = config or ContextConfig.load()
        self.chunker = DocumentChunker(
            chunk_size=self.config.embedding.chunk_size,
            chunk_overlap=self.config.embedding.chunk_overlap,
        )
        self._client: Any = None
        self._embedding_fn: Any = None

    def _check_chromadb(self) -> None:
        """Check if chromadb is available."""
        try:
            import chromadb  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "chromadb is required for vector search. "
                "Install with: pip install agentbible[context]"
            ) from e

    def get_embedding_function(self) -> Any:
        """Get embedding function based on config."""
        if self._embedding_fn is not None:
            return self._embedding_fn

        api_key = os.environ.get("OPENAI_API_KEY")

        if api_key and self.config.embedding.provider == "openai":
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

            self._embedding_fn = OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=self.config.embedding.model,
            )
        else:
            # Fallback to local embeddings
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )

                self._embedding_fn = SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            except ImportError as e:
                raise ImportError(
                    "No embedding function available. Set OPENAI_API_KEY or "
                    "install sentence-transformers."
                ) from e

        return self._embedding_fn

    def get_persistent_client(self, path: Path) -> Any:
        """Get a persistent ChromaDB client for a specific path."""
        self._check_chromadb()
        import chromadb
        from chromadb.config import Settings

        path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )

    def load_directory(
        self,
        directory: str | Path,
    ) -> str:
        """Load all markdown files from a directory and return as context.

        This is a simple, non-vector approach that just reads and
        concatenates all markdown files from the directory.

        Args:
            directory: Path to directory containing docs

        Returns:
            Formatted context string
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all markdown files
        md_files = sorted(dir_path.glob("**/*.md"))

        if not md_files:
            return f"# Context from {dir_path.name}\n\nNo markdown files found."

        parts = [f"# Context from {dir_path.name}\n"]
        parts.append(f"**Files loaded:** {len(md_files)}\n")
        parts.append("---\n")

        for md_file in md_files:
            content = md_file.read_text()
            relative_path = md_file.relative_to(dir_path)

            parts.append(f"\n## Source: {relative_path}\n")
            parts.append(content)
            parts.append("\n---\n")

        return "".join(parts)

    def query(
        self,
        query: str,
        project_name: str | None = None,
        top_k: int | None = None,
        include_global: bool = True,
    ) -> list[RetrievedChunk]:
        """Query for relevant chunks using semantic search.

        Args:
            query: The semantic search query
            project_name: Optional project to search within
            top_k: Number of results per collection
            include_global: Whether to include global docs

        Returns:
            List of retrieved chunks, sorted by relevance
        """
        self._check_chromadb()

        if top_k is None:
            top_k = self.config.retrieval.default_top_k

        results: list[RetrievedChunk] = []

        # Query global docs
        if include_global:
            try:
                global_collection = self._get_collection(None)
                global_results = global_collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                results.extend(self._parse_query_results(global_results))
            except FileNotFoundError:
                pass  # Global docs not embedded yet

        # Query project docs
        if project_name:
            try:
                project_collection = self._get_collection(project_name)
                project_results = project_collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                results.extend(self._parse_query_results(project_results))
            except FileNotFoundError:
                pass  # Project not embedded yet

        # Sort by distance (lower is better)
        results.sort(key=lambda x: x.distance)
        return results

    def build_context(
        self,
        query: str | None = None,
        project_name: str | None = None,
        all_from_dir: str | Path | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Build context string for AI coding session.

        Args:
            query: Optional semantic query to find relevant chunks
            project_name: Optional project to include
            all_from_dir: Load all docs from this directory (simple mode)
            top_k: Number of query results to include
            max_tokens: Optional max token limit

        Returns:
            Formatted context string with source references
        """
        # Simple mode: just load directory contents
        if all_from_dir:
            return self.load_directory(all_from_dir)

        # Vector search mode
        if query:
            chunks = self.query(
                query=query,
                project_name=project_name,
                top_k=top_k,
            )

            # Filter by similarity threshold
            threshold = self.config.retrieval.similarity_threshold
            chunks = [c for c in chunks if c.score >= threshold]

            return self._format_chunks(chunks, query=query, max_tokens=max_tokens)

        # Default: return always_include docs
        return self._get_always_include_context(project_name, max_tokens)

    def embed_directory(
        self,
        directory: str | Path,
        collection_name: str = "local_docs",
    ) -> int:
        """Embed all documents from a directory into vector database.

        Args:
            directory: Path to directory containing docs
            collection_name: Name for the ChromaDB collection

        Returns:
            Number of chunks created
        """
        self._check_chromadb()

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        db_path = get_vectordb_path(collection_name)
        client = self.get_persistent_client(db_path)

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.get_embedding_function(),
            metadata={"description": f"Docs from {dir_path}"},
        )

        # Clear existing
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        total_chunks = 0
        md_files = list(dir_path.glob("**/*.md"))

        for md_file in md_files:
            content = md_file.read_text()
            chunks = self.chunker.chunk_document(
                content=content,
                source=str(md_file),
                doc_type="reference",
            )

            if chunks:
                collection.add(
                    ids=[c.get_id() for c in chunks],
                    documents=[c.content for c in chunks],
                    metadatas=[c.to_metadata() for c in chunks],
                )
                total_chunks += len(chunks)

        return total_chunks

    def _get_collection(self, project_name: str | None) -> Any:
        """Get ChromaDB collection for global or project docs."""
        import chromadb
        from chromadb.config import Settings

        db_path = get_vectordb_path(project_name)

        if not db_path.exists():
            raise FileNotFoundError(
                f"Vector database not found at {db_path}. "
                "Run 'bible context --embed' to create embeddings first."
            )

        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )

        collection_name = f"project_{project_name}" if project_name else "global_docs"

        return client.get_collection(
            name=collection_name,
            embedding_function=self.get_embedding_function(),
        )

    def _parse_query_results(self, results: Any) -> list[RetrievedChunk]:
        """Parse ChromaDB query results into RetrievedChunk objects."""
        chunks: list[RetrievedChunk] = []

        documents = results.get("documents")
        if not documents or not documents[0]:
            return chunks

        docs = documents[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc in enumerate(docs):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0

            chunks.append(
                RetrievedChunk.from_query_result(
                    content=str(doc),
                    metadata=metadata or {},
                    distance=float(distance),
                )
            )

        return chunks

    def _format_chunks(
        self,
        chunks: list[RetrievedChunk],
        query: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Format chunks into context string."""
        if not chunks:
            return "# Context\n\nNo relevant documents found."

        # Group by source
        chunks_by_source: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            source = chunk.source
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)

        # Sort within each source
        for source_chunks in chunks_by_source.values():
            source_chunks.sort(key=lambda x: x.line_start)

        parts = ["# Context\n"]
        if query:
            parts.append(f"**Query:** {query}\n")
        parts.append(f"**Chunks loaded:** {len(chunks)}\n")
        parts.append("---\n")

        current_tokens = 0

        for source, source_chunks in chunks_by_source.items():
            source_name = Path(source).name
            parts.append(f"\n## Source: {source_name}\n")

            for chunk in source_chunks:
                chunk_header = (
                    f"\n### {chunk.section} "
                    f"(lines {chunk.line_start}-{chunk.line_end})\n"
                )
                chunk_content = chunk.content + "\n"
                chunk_text = chunk_header + chunk_content

                # Estimate tokens
                chunk_tokens = self.chunker.count_tokens(chunk_text)

                if max_tokens and current_tokens + chunk_tokens > max_tokens:
                    parts.append("\n*[Truncated due to token limit]*\n")
                    break

                parts.append(chunk_text)
                current_tokens += chunk_tokens

        return "".join(parts)

    def _get_always_include_context(
        self,
        project_name: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Get context from always_include documents."""
        # Unused parameters - kept for API compatibility
        _ = project_name
        _ = max_tokens
        # For now, just return a message - full implementation would
        # query the vector DB for always_include docs
        return (
            "# Context\n\n"
            "No query provided. Use --query to search for relevant context, "
            "or --all to load all documents from a directory."
        )
