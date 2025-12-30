# OpenCode Context Manager

**Vector-based context retrieval for AI coding sessions.**

Reduce token usage by loading only relevant document chunks instead of entire files. Uses semantic search to find the most relevant context for your current task.

## The Problem

When using AI agents for coding:
- Large documentation files (50-60k tokens) quickly fill context windows
- Loading everything wastes tokens on irrelevant content
- Manual context selection is tedious and error-prone

## The Solution

Chunk your documentation, embed it in a vector database, and retrieve only what's relevant:

```bash
# Load all docs from a directory (~20k tokens instead of 60k)
oc-context --all ./agent_docs

# Semantic search for specific topics
oc-context --query "error handling validation"

# Preview what would be loaded
oc-context --dry-run
```

## Quick Start

### 1. Install

```bash
# Clone or copy to ~/.local/share/opencode/
mkdir -p ~/.local/share/opencode
cp -r opencode-context/* ~/.local/share/opencode/

# Create virtual environment
cd ~/.local/share/opencode
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Make CLI scripts executable
chmod +x bin/oc-context bin/oc-update

# Add to PATH (add to ~/.bashrc)
export PATH="$HOME/.local/share/opencode/bin:$PATH"
```

### 2. Configure

```bash
# Copy and edit config
cp config.example.yaml ~/.local/share/opencode/config.yaml
# Edit config.yaml with your docs and projects
```

### 3. Embed Documents

```bash
# Set OpenAI API key (or install sentence-transformers for local embeddings)
export OPENAI_API_KEY="your-key"

# Embed all configured documents
source ~/.local/share/opencode/venv/bin/activate
oc-update

# Check status
oc-update --status
```

### 4. Generate Context

```bash
# Navigate to your project
cd /path/to/your-project

# Generate context file
oc-context --all ./agent_docs

# Load .opencode-context.md into your AI session
```

## Commands

### oc-update

Update vector embeddings for configured documents.

```bash
oc-update                    # Embed everything
oc-update --status           # Show embedding status
oc-update --project foo      # Embed specific project only
oc-update --global-only      # Embed global docs only
```

### oc-context

Generate context files for AI sessions.

```bash
oc-context                          # Load always_include docs
oc-context --all ./agent_docs       # Load ALL files from directory
oc-context --query "topic"          # Semantic search
oc-context --dry-run                # Preview without generating
oc-context --stdout                 # Print to stdout
oc-context --max-tokens 8000        # Limit output size
oc-context --project foo            # Specify project explicitly
```

## Configuration

Edit `~/.local/share/opencode/config.yaml`:

```yaml
embedding:
  provider: openai
  model: text-embedding-3-small
  chunk_size: 1000      # tokens per chunk
  chunk_overlap: 200    # overlap for continuity

retrieval:
  default_top_k: 5              # query results count
  similarity_threshold: 0.7     # minimum relevance score

global_docs:
  - path: /path/to/principles.md
    always_include: true        # auto-load every session
    description: "Core principles"

projects:
  my-project:
    root: /path/to/project
    docs:
      - path: agent_docs/guide.md   # relative to root
        always_include: true
```

## How It Works

1. **Chunking**: Documents are split into ~1000 token chunks with 200 token overlap, preserving markdown structure (headers, sections)

2. **Embedding**: Chunks are embedded using OpenAI's `text-embedding-3-small` (or local sentence-transformers as fallback)

3. **Storage**: Embeddings stored in ChromaDB (local SQLite-based vector DB)

4. **Retrieval**: 
   - `--all`: Load every chunk from specified directory
   - `--query`: Semantic similarity search
   - Default: Load `always_include` documents

5. **Output**: Formatted markdown with source references and line numbers

## Architecture

```
~/.local/share/opencode/
├── bin/
│   ├── oc-context      # Context generation CLI
│   └── oc-update       # Embedding update CLI
├── oc_lib/
│   ├── config.py       # Configuration loading
│   ├── embed.py        # Document chunking & embedding
│   └── retrieve.py     # Semantic search & retrieval
├── vectordb/
│   ├── global/         # Global docs database
│   └── projects/       # Per-project databases
├── config.yaml         # Your configuration
└── venv/               # Python virtual environment
```

## Token Estimates

| Context Type | Approximate Tokens |
|-------------|-------------------|
| Single always_include doc | ~3,000-5,000 |
| Full agent_docs/ directory | ~15,000-25,000 |
| Query results (5 chunks) | ~5,000 |

Use `--max-tokens` to cap output size.

## Requirements

- Python 3.9+
- OpenAI API key (or `sentence-transformers` for local embeddings)
- Dependencies: `chromadb`, `openai`, `tiktoken`, `pyyaml`, `click`, `rich`

## License

MIT
