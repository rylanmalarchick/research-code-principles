"""
OpenCode Context Manager - Configuration Module

Handles loading and validating the config.yaml file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Default paths
OPENCODE_DIR = Path.home() / ".local" / "share" / "opencode"
CONFIG_PATH = OPENCODE_DIR / "config.yaml"
VECTORDB_DIR = OPENCODE_DIR / "vectordb"


@dataclass
class DocConfig:
    """Configuration for a single document."""
    
    path: str
    always_include: bool = False
    description: str = ""
    type: str = "reference"
    
    def resolve_path(self, project_root: Optional[Path] = None) -> Path:
        """Resolve document path (absolute or relative to project root)."""
        p = Path(self.path)
        if p.is_absolute():
            return p
        if project_root:
            return project_root / p
        raise ValueError(f"Relative path {self.path} requires project_root")


@dataclass
class ProjectConfig:
    """Configuration for a project."""
    
    name: str
    root: Path
    description: str = ""
    docs: List[DocConfig] = field(default_factory=list)
    
    def get_always_include_docs(self) -> List[DocConfig]:
        """Get docs marked as always_include."""
        return [d for d in self.docs if d.always_include]


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    
    default_top_k: int = 5
    similarity_threshold: float = 0.7


@dataclass
class Config:
    """Main configuration container."""
    
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    global_docs: List[DocConfig]
    projects: Dict[str, ProjectConfig]
    
    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Parse config from dictionary."""
        # Embedding config
        embed_data = data.get("embedding", {})
        embedding = EmbeddingConfig(
            provider=embed_data.get("provider", "openai"),
            model=embed_data.get("model", "text-embedding-3-small"),
            chunk_size=embed_data.get("chunk_size", 1000),
            chunk_overlap=embed_data.get("chunk_overlap", 200),
        )
        
        # Retrieval config
        retr_data = data.get("retrieval", {})
        retrieval = RetrievalConfig(
            default_top_k=retr_data.get("default_top_k", 5),
            similarity_threshold=retr_data.get("similarity_threshold", 0.7),
        )
        
        # Global docs
        global_docs = []
        for doc_data in data.get("global_docs", []):
            global_docs.append(DocConfig(
                path=doc_data["path"],
                always_include=doc_data.get("always_include", False),
                description=doc_data.get("description", ""),
                type=doc_data.get("type", "reference"),
            ))
        
        # Projects
        projects = {}
        for name, proj_data in data.get("projects", {}).items():
            docs = []
            for doc_data in proj_data.get("docs", []):
                docs.append(DocConfig(
                    path=doc_data["path"],
                    always_include=doc_data.get("always_include", False),
                    description=doc_data.get("description", ""),
                    type=doc_data.get("type", "reference"),
                ))
            
            projects[name] = ProjectConfig(
                name=name,
                root=Path(proj_data["root"]),
                description=proj_data.get("description", ""),
                docs=docs,
            )
        
        return cls(
            embedding=embedding,
            retrieval=retrieval,
            global_docs=global_docs,
            projects=projects,
        )
    
    def get_project(self, name: str) -> Optional[ProjectConfig]:
        """Get project by name."""
        return self.projects.get(name)
    
    def detect_project(self, cwd: Path) -> Optional[ProjectConfig]:
        """Detect project based on current working directory."""
        cwd = cwd.resolve()
        
        # Check if cwd is under any registered project root
        for project in self.projects.values():
            try:
                cwd.relative_to(project.root)
                return project
            except ValueError:
                continue
        
        # Check for agent_docs directory
        for parent in [cwd] + list(cwd.parents):
            if (parent / "agent_docs").is_dir():
                # Try to match against registered projects
                for project in self.projects.values():
                    if project.root == parent:
                        return project
        
        return None
    
    def get_global_always_include(self) -> List[DocConfig]:
        """Get global docs marked as always_include."""
        return [d for d in self.global_docs if d.always_include]


def get_vectordb_path(project_name: Optional[str] = None) -> Path:
    """Get path to vector database."""
    if project_name:
        return VECTORDB_DIR / "projects" / project_name
    return VECTORDB_DIR / "global"
