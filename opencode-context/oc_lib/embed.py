"""
OpenCode Context Manager - Embedding Module

Handles document chunking and embedding generation.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import tiktoken
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import (
    Config,
    DocConfig,
    ProjectConfig,
    VECTORDB_DIR,
    get_vectordb_path,
)

console = Console()


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
    
    def to_metadata(self) -> Dict[str, Any]:
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
    """Chunks markdown documents intelligently."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
    
    def chunk_document(
        self,
        content: str,
        source: str,
        doc_type: str = "reference",
        always_include: bool = False,
        description: str = "",
    ) -> List[Chunk]:
        """Split document into chunks, preserving markdown structure."""
        lines = content.split("\n")
        chunks = []
        
        current_section = "Introduction"
        current_chunk_lines: List[Tuple[int, str]] = []
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
                    self.count_tokens(l + "\n") for _, l in current_chunk_lines
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
        lines: List[Tuple[int, str]],
        source: str,
        doc_type: str,
        section: str,
        always_include: bool,
        description: str,
        chunk_index: int,
    ) -> Chunk:
        """Create a Chunk from lines."""
        content = "\n".join(line for _, line in lines)
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
    
    def _get_overlap_lines(
        self, lines: List[Tuple[int, str]]
    ) -> List[Tuple[int, str]]:
        """Get lines for overlap from the end of current chunk."""
        overlap_tokens = 0
        overlap_lines = []
        
        for line_num, line in reversed(lines):
            line_tokens = self.count_tokens(line + "\n")
            if overlap_tokens + line_tokens > self.chunk_overlap:
                break
            overlap_lines.insert(0, (line_num, line))
            overlap_tokens += line_tokens
        
        return overlap_lines


class EmbeddingManager:
    """Manages document embedding and storage in ChromaDB."""
    
    def __init__(self, config: Config):
        self.config = config
        self.chunker = DocumentChunker(
            chunk_size=config.embedding.chunk_size,
            chunk_overlap=config.embedding.chunk_overlap,
        )
        self._client = None
        self._embedding_fn = None
    
    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ))
        return self._client
    
    def get_persistent_client(self, path: Path):
        """Get a persistent ChromaDB client for a specific path."""
        import chromadb
        from chromadb.config import Settings
        
        path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
    
    def get_embedding_function(self):
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
            console.print(
                f"[green]Using OpenAI embeddings ({self.config.embedding.model})[/green]"
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
                console.print(
                    "[yellow]⚠ No OpenAI key found, using local embeddings "
                    "(all-MiniLM-L6-v2)[/yellow]"
                )
            except ImportError:
                console.print(
                    "[red]Error: sentence-transformers not installed and no "
                    "OpenAI key found.[/red]"
                )
                console.print(
                    "Install with: pip install sentence-transformers"
                )
                raise
        
        return self._embedding_fn
    
    def embed_global_docs(self) -> int:
        """Embed all global documents. Returns number of chunks created."""
        db_path = get_vectordb_path()
        client = self.get_persistent_client(db_path)
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name="global_docs",
            embedding_function=self.get_embedding_function(),
            metadata={"description": "Global standards and reference documents"},
        )
        
        # Clear existing documents
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        
        total_chunks = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding global docs...", total=None)
            
            for doc_config in self.config.global_docs:
                doc_path = Path(doc_config.path)
                
                if not doc_path.exists():
                    console.print(f"[yellow]Warning: {doc_path} not found[/yellow]")
                    continue
                
                progress.update(task, description=f"Processing {doc_path.name}...")
                
                content = doc_path.read_text()
                chunks = self.chunker.chunk_document(
                    content=content,
                    source=str(doc_path),
                    doc_type=doc_config.type,
                    always_include=doc_config.always_include,
                    description=doc_config.description,
                )
                
                if chunks:
                    collection.add(
                        ids=[c.get_id() for c in chunks],
                        documents=[c.content for c in chunks],
                        metadatas=[c.to_metadata() for c in chunks],
                    )
                    total_chunks += len(chunks)
                    console.print(
                        f"  [dim]{doc_path.name}: {len(chunks)} chunks[/dim]"
                    )
        
        # Save metadata
        self._save_metadata(db_path, {
            "type": "global",
            "updated_at": datetime.now().isoformat(),
            "total_chunks": total_chunks,
            "doc_count": len(self.config.global_docs),
        })
        
        return total_chunks
    
    def embed_project_docs(self, project_name: str) -> int:
        """Embed documents for a specific project. Returns number of chunks."""
        project = self.config.get_project(project_name)
        if not project:
            raise ValueError(f"Unknown project: {project_name}")
        
        db_path = get_vectordb_path(project_name)
        client = self.get_persistent_client(db_path)
        
        collection = client.get_or_create_collection(
            name=f"project_{project_name}",
            embedding_function=self.get_embedding_function(),
            metadata={"description": project.description},
        )
        
        # Clear existing
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        
        total_chunks = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Embedding {project_name}...", total=None)
            
            for doc_config in project.docs:
                doc_path = doc_config.resolve_path(project.root)
                
                if not doc_path.exists():
                    console.print(f"[yellow]Warning: {doc_path} not found[/yellow]")
                    continue
                
                progress.update(task, description=f"Processing {doc_path.name}...")
                
                content = doc_path.read_text()
                chunks = self.chunker.chunk_document(
                    content=content,
                    source=str(doc_path),
                    doc_type=doc_config.type,
                    always_include=doc_config.always_include,
                    description=doc_config.description,
                )
                
                if chunks:
                    collection.add(
                        ids=[c.get_id() for c in chunks],
                        documents=[c.content for c in chunks],
                        metadatas=[c.to_metadata() for c in chunks],
                    )
                    total_chunks += len(chunks)
                    console.print(
                        f"  [dim]{doc_path.name}: {len(chunks)} chunks[/dim]"
                    )
        
        self._save_metadata(db_path, {
            "type": "project",
            "project": project_name,
            "updated_at": datetime.now().isoformat(),
            "total_chunks": total_chunks,
            "doc_count": len(project.docs),
        })
        
        return total_chunks
    
    def embed_all(self) -> Dict[str, int]:
        """Embed all global and project documents."""
        results = {}
        
        console.print("\n[bold]Embedding global documents...[/bold]")
        results["global"] = self.embed_global_docs()
        
        for project_name in self.config.projects:
            console.print(f"\n[bold]Embedding project: {project_name}[/bold]")
            results[project_name] = self.embed_project_docs(project_name)
        
        console.print("\n[bold green]Embedding complete![/bold green]")
        for name, count in results.items():
            console.print(f"  {name}: {count} chunks")
        
        return results
    
    def _save_metadata(self, db_path: Path, metadata: Dict[str, Any]) -> None:
        """Save metadata file alongside database."""
        meta_path = db_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
