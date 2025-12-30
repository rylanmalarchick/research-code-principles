"""
OpenCode Context Manager - Retrieval Module

Handles semantic search and context retrieval from embedded documents.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rich.console import Console

from .config import (
    Config,
    DocConfig,
    ProjectConfig,
    VECTORDB_DIR,
    get_vectordb_path,
)

console = Console()


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
        metadata: Dict[str, Any],
        distance: float,
    ) -> "RetrievedChunk":
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


class ContextRetriever:
    """Retrieves relevant context from embedded documents."""
    
    def __init__(self, config: Config):
        self.config = config
        self._embedding_fn: Any = None
    
    def get_embedding_function(self) -> Any:
        """Get embedding function (shared with EmbeddingManager)."""
        if self._embedding_fn is not None:
            return self._embedding_fn
        
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if api_key and self.config.embedding.provider == "openai":
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            
            self._embedding_fn = OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=self.config.embedding.model,
            )
        else:
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )
                
                self._embedding_fn = SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            except ImportError:
                raise RuntimeError(
                    "No embedding function available. Set OPENAI_API_KEY or "
                    "install sentence-transformers."
                )
        
        return self._embedding_fn
    
    def get_collection(self, project_name: Optional[str] = None) -> Any:
        """Get ChromaDB collection for global or project docs."""
        import chromadb
        from chromadb.config import Settings
        
        db_path = get_vectordb_path(project_name)
        
        if not db_path.exists():
            raise FileNotFoundError(
                f"Vector database not found at {db_path}. "
                f"Run 'oc-update' to create embeddings first."
            )
        
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        
        collection_name = (
            f"project_{project_name}" if project_name else "global_docs"
        )
        
        return client.get_collection(
            name=collection_name,
            embedding_function=self.get_embedding_function(),
        )
    
    def query(
        self,
        query: str,
        project_name: Optional[str] = None,
        top_k: Optional[int] = None,
        include_global: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Query for relevant chunks.
        
        Args:
            query: The semantic search query
            project_name: Optional project to search within
            top_k: Number of results per collection
            include_global: Whether to include global docs
            
        Returns:
            List of retrieved chunks, sorted by relevance
        """
        if top_k is None:
            top_k = self.config.retrieval.default_top_k
        
        results: List[RetrievedChunk] = []
        
        # Query global docs
        if include_global:
            try:
                global_collection = self.get_collection(None)
                global_results = global_collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                results.extend(self._parse_query_results(global_results))
            except FileNotFoundError:
                console.print("[yellow]Warning: Global docs not embedded yet[/yellow]")
        
        # Query project docs
        if project_name:
            try:
                project_collection = self.get_collection(project_name)
                project_results = project_collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                results.extend(self._parse_query_results(project_results))
            except FileNotFoundError:
                console.print(
                    f"[yellow]Warning: Project {project_name} not embedded yet[/yellow]"
                )
        
        # Sort by distance (lower is better) and return
        results.sort(key=lambda x: x.distance)
        return results
    
    def get_all_chunks_from_dir(
        self,
        directory: Path,
        project_name: Optional[str] = None,
        include_global: bool = False,
    ) -> List[RetrievedChunk]:
        """
        Get all chunks from files within a specific directory.
        
        Args:
            directory: Directory path to filter by (matches if source contains this path)
            project_name: Optional project to search within
            include_global: Whether to include global docs
            
        Returns:
            List of all chunks from files in the directory
        """
        results: List[RetrievedChunk] = []
        dir_str = str(directory.resolve())
        
        # Get from global docs if requested
        if include_global:
            try:
                global_collection = self.get_collection(None)
                global_all = global_collection.get(
                    include=["documents", "metadatas"],
                )
                
                documents = global_all.get("documents") or []
                metadatas = global_all.get("metadatas") or []
                
                for i, doc in enumerate(documents):
                    if i < len(metadatas):
                        metadata = metadatas[i] or {}
                    else:
                        metadata = {}
                    
                    source = str(metadata.get("source", ""))
                    # Check if source is within the directory
                    if dir_str in source:
                        results.append(RetrievedChunk(
                            content=str(doc),
                            source=source,
                            section=str(metadata.get("section", "")),
                            line_start=int(metadata.get("line_start", 0)),
                            line_end=int(metadata.get("line_end", 0)),
                            doc_type=str(metadata.get("doc_type", "reference")),
                            always_include=bool(metadata.get("always_include", False)),
                            description=str(metadata.get("description", "")),
                            distance=0.0,
                            chunk_index=int(metadata.get("chunk_index", 0)),
                            total_chunks=int(metadata.get("total_chunks", 1)),
                        ))
            except FileNotFoundError:
                pass
        
        # Get from project docs
        if project_name:
            try:
                project_collection = self.get_collection(project_name)
                project_all = project_collection.get(
                    include=["documents", "metadatas"],
                )
                
                documents = project_all.get("documents") or []
                metadatas = project_all.get("metadatas") or []
                
                for i, doc in enumerate(documents):
                    if i < len(metadatas):
                        metadata = metadatas[i] or {}
                    else:
                        metadata = {}
                    
                    source = str(metadata.get("source", ""))
                    # Check if source is within the directory
                    if dir_str in source:
                        results.append(RetrievedChunk(
                            content=str(doc),
                            source=source,
                            section=str(metadata.get("section", "")),
                            line_start=int(metadata.get("line_start", 0)),
                            line_end=int(metadata.get("line_end", 0)),
                            doc_type=str(metadata.get("doc_type", "reference")),
                            always_include=bool(metadata.get("always_include", False)),
                            description=str(metadata.get("description", "")),
                            distance=0.0,
                            chunk_index=int(metadata.get("chunk_index", 0)),
                            total_chunks=int(metadata.get("total_chunks", 1)),
                        ))
            except FileNotFoundError:
                pass
        
        # Sort by source then chunk_index for consistent ordering
        results.sort(key=lambda x: (x.source, x.chunk_index))
        return results
    
    def get_always_include_chunks(
        self,
        project_name: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Get all chunks from always_include documents.
        
        This is the default context loading behavior - load all chunks
        from documents marked as always_include.
        """
        results: List[RetrievedChunk] = []
        
        # Get global always_include docs
        try:
            global_collection = self.get_collection(None)
            global_all = global_collection.get(
                where={"always_include": True},
                include=["documents", "metadatas"],
            )
            
            documents = global_all.get("documents") or []
            metadatas = global_all.get("metadatas") or []
            
            for i, doc in enumerate(documents):
                if i < len(metadatas):
                    metadata = metadatas[i] or {}
                else:
                    metadata = {}
                    
                results.append(RetrievedChunk(
                    content=str(doc),
                    source=str(metadata.get("source", "")),
                    section=str(metadata.get("section", "")),
                    line_start=int(metadata.get("line_start", 0)),
                    line_end=int(metadata.get("line_end", 0)),
                    doc_type=str(metadata.get("doc_type", "reference")),
                    always_include=True,
                    description=str(metadata.get("description", "")),
                    distance=0.0,  # Always included, so "perfect" relevance
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    total_chunks=int(metadata.get("total_chunks", 1)),
                ))
        except FileNotFoundError:
            pass
        
        # Get project always_include docs
        if project_name:
            try:
                project_collection = self.get_collection(project_name)
                project_all = project_collection.get(
                    where={"always_include": True},
                    include=["documents", "metadatas"],
                )
                
                documents = project_all.get("documents") or []
                metadatas = project_all.get("metadatas") or []
                
                for i, doc in enumerate(documents):
                    if i < len(metadatas):
                        metadata = metadatas[i] or {}
                    else:
                        metadata = {}
                        
                    results.append(RetrievedChunk(
                        content=str(doc),
                        source=str(metadata.get("source", "")),
                        section=str(metadata.get("section", "")),
                        line_start=int(metadata.get("line_start", 0)),
                        line_end=int(metadata.get("line_end", 0)),
                        doc_type=str(metadata.get("doc_type", "reference")),
                        always_include=True,
                        description=str(metadata.get("description", "")),
                        distance=0.0,
                        chunk_index=int(metadata.get("chunk_index", 0)),
                        total_chunks=int(metadata.get("total_chunks", 1)),
                    ))
            except FileNotFoundError:
                pass
        
        # Sort by source then chunk_index for consistent ordering
        results.sort(key=lambda x: (x.source, x.chunk_index))
        return results
    
    def build_context(
        self,
        query: Optional[str] = None,
        project_name: Optional[str] = None,
        top_k: Optional[int] = None,
        include_always: bool = True,
        max_tokens: Optional[int] = None,
        all_from_dir: Optional[Path] = None,
    ) -> str:
        """
        Build context string for OpenCode session.
        
        Args:
            query: Optional semantic query to find relevant chunks
            project_name: Optional project to include
            top_k: Number of query results to include
            include_always: Include always_include docs
            max_tokens: Optional max token limit
            all_from_dir: Load all chunks from this directory
            
        Returns:
            Formatted context string with source references
        """
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        
        chunks: List[RetrievedChunk] = []
        seen_ids: Set[str] = set()
        
        # If --all directory specified, load everything from that dir
        if all_from_dir:
            dir_chunks = self.get_all_chunks_from_dir(
                directory=all_from_dir,
                project_name=project_name,
                include_global=True,
            )
            for chunk in dir_chunks:
                chunk_id = f"{chunk.source}:{chunk.line_start}"
                if chunk_id not in seen_ids:
                    chunks.append(chunk)
                    seen_ids.add(chunk_id)
        
        # Add always_include chunks (unless --all was used, then skip to avoid dupes)
        elif include_always:
            always_chunks = self.get_always_include_chunks(project_name)
            for chunk in always_chunks:
                chunk_id = f"{chunk.source}:{chunk.line_start}"
                if chunk_id not in seen_ids:
                    chunks.append(chunk)
                    seen_ids.add(chunk_id)
        
        # Add query results
        if query:
            query_chunks = self.query(
                query=query,
                project_name=project_name,
                top_k=top_k or self.config.retrieval.default_top_k,
            )
            
            for chunk in query_chunks:
                chunk_id = f"{chunk.source}:{chunk.line_start}"
                if chunk_id not in seen_ids:
                    # Apply similarity threshold
                    if chunk.score >= self.config.retrieval.similarity_threshold:
                        chunks.append(chunk)
                        seen_ids.add(chunk_id)
        
        # Build output
        current_tokens = 0
        
        # Group chunks by source for cleaner output
        chunks_by_source: Dict[str, List[RetrievedChunk]] = {}
        for chunk in chunks:
            source = chunk.source
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
        
        # Sort chunks within each source by line number
        for source_chunks in chunks_by_source.values():
            source_chunks.sort(key=lambda x: x.line_start)
        
        output_parts: List[str] = []
        truncated = False
        
        for source, source_chunks in chunks_by_source.items():
            source_name = Path(source).name
            
            # Build source section
            section_parts = [f"\n## Source: {source_name}\n"]
            if source_chunks[0].description:
                section_parts.append(f"*{source_chunks[0].description}*\n")
            
            for chunk in source_chunks:
                chunk_header = f"\n### {chunk.section} (lines {chunk.line_start}-{chunk.line_end})\n"
                chunk_content = chunk.content + "\n"
                
                chunk_text = chunk_header + chunk_content
                chunk_tokens = len(encoder.encode(chunk_text))
                
                if max_tokens and current_tokens + chunk_tokens > max_tokens:
                    section_parts.append("\n*[Truncated due to token limit]*\n")
                    truncated = True
                    break
                
                section_parts.append(chunk_text)
                current_tokens += chunk_tokens
            
            output_parts.append("".join(section_parts))
            
            if truncated:
                break
        
        # Build final output
        header = "# OpenCode Context\n\n"
        if project_name:
            header += f"**Project:** {project_name}\n"
        if query:
            header += f"**Query:** {query}\n"
        header += f"**Chunks loaded:** {len(chunks)}\n"
        header += f"**Approximate tokens:** {current_tokens}\n"
        header += "\n---\n"
        
        return header + "".join(output_parts)
    
    def _parse_query_results(self, results: Any) -> List[RetrievedChunk]:
        """Parse ChromaDB query results into RetrievedChunk objects."""
        chunks: List[RetrievedChunk] = []
        
        documents = results.get("documents")
        if not documents or not documents[0]:
            return chunks
        
        docs = documents[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for i, doc in enumerate(docs):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            
            chunks.append(RetrievedChunk.from_query_result(
                content=str(doc),
                metadata=metadata or {},
                distance=float(distance),
            ))
        
        return chunks
