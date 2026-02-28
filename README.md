# AIMemory

Intelligent memory management MCP server for AI assistants. Automatically stores, retrieves, and manages conversational memories using RL-based policies, knowledge graphs, and semantic search.

## Features

- **Automatic memory management** — RL contextual bandit decides when to save, skip, or retrieve
- **Semantic search** — ChromaDB + multilingual sentence-transformer embeddings
- **Knowledge graph** — Entity-relation graph for hybrid retrieval
- **Multi-resolution text** — Full text, summary, and entity triple compression levels
- **Forgetting pipeline** — Decay-based memory aging with consolidation
- **Multilingual** — Korean (ko) and English (en) pattern support via i18n module
- **MCP server** — Drop-in integration with OpenClaw, Claude Desktop, and any MCP client

## Installation

### Basic (English mode)

```bash
uv sync
```

### With Korean support (MeCab)

```bash
uv sync --extra ko
```

### Development

```bash
uv sync --extra dev --extra ko
```

## Usage

### As MCP server

```bash
# Run directly
uv run python -m aimemory.mcp

# Or via entry point
uv run aimemory-mcp
```

### OpenClaw integration

```bash
bash scripts/install_openclaw.sh
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "aimemory": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/AIMemory", "python", "-m", "aimemory.mcp"],
      "env": {
        "AIMEMORY_DB_PATH": "/path/to/AIMemory/memory_db"
      }
    }
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AIMEMORY_DB_PATH` | `./memory_db` | ChromaDB persistence directory |
| `AIMEMORY_LANGUAGE` | `ko` | Language for pattern matching (`ko` or `en`) |
| `AIMEMORY_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Sentence-transformer model |
| `AIMEMORY_LOG_LEVEL` | `INFO` | Logging level |

## Development

```bash
# Run tests
uv run pytest tests/ -q

# Run i18n tests
uv run pytest tests/test_i18n.py -v

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

## Architecture

```
src/aimemory/
├── i18n/           # Multilingual patterns (ko, en)
├── memory/         # Graph store, retriever, resolution, forgetting
├── mcp/            # MCP server and bridge
├── online/         # RL policy, encoder, reranker
├── reward/         # Feedback detection, implicit rewards
├── selfplay/       # Training data generation
├── extractor/      # DualHeadDQN feature extractor
└── dataset/        # Dataset building and splitting
```

## License

MIT
