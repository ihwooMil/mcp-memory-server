# AIMemory

**Intelligent memory for AI assistants that actually remembers.**

> Drop-in MCP server that gives Claude (and any MCP client) persistent, searchable, self-organizing memory — powered by semantic search, knowledge graphs, and reinforcement learning.

[![CI](https://github.com/ihwooMil/mcp-memory-server/actions/workflows/ci.yml/badge.svg)](https://github.com/ihwooMil/mcp-memory-server/actions)
[![PyPI](https://img.shields.io/pypi/v/mcp-memory-server)](https://pypi.org/project/mcp-memory-server/)
[![Python](https://img.shields.io/pypi/pyversions/mcp-memory-server)](https://pypi.org/project/mcp-memory-server/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why AIMemory?

Current AI memory tools have two critical problems:

| Problem | How AIMemory solves it |
|---------|----------------------|
| **Manual retrieval** — you must ask "do you remember X?" | `auto_search` runs every turn, injecting relevant memories automatically |
| **Token waste** — entire memory dump inserted into context | Multi-resolution composer selects top-K memories within a token budget |

## Key Features

- **RL-powered policy** — Contextual bandit decides when to save, skip, or retrieve (not just keyword matching)
- **Semantic search** — ChromaDB + multilingual sentence-transformer embeddings (`intfloat/multilingual-e5-small`)
- **Knowledge graph** — Entity-relation graph (NetworkX) for multi-hop reasoning ("who likes what?")
- **GraphRAG hybrid retrieval** — Vector similarity + graph traversal, fused and re-ranked by an RL re-ranker
- **Multi-resolution text** — Full text → summary → entity triples, composed within token budget
- **Forgetting pipeline** — Decay-based aging with consolidation, pinning, and immutable protection
- **Sleep cycle** — Periodic maintenance: dedup, compress, forget, checkpoint
- **Multilingual** — Korean and English pattern support out of the box

---

## Quick Start (2 minutes)

### 1. Install

```bash
pip install mcp-memory-server
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install mcp-memory-server
```

<details>
<summary>Optional: Korean NLP support</summary>

```bash
pip install mcp-memory-server[ko]
```
</details>

### 2. Connect to OpenClaw

```bash
mcporter config add aimemory --command aimemory-mcp --scope home
```

### 3. Connect to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "aimemory": {
      "command": "aimemory-mcp"
    }
  }
}
```

That's it. Claude now has persistent memory across all conversations.

<details>
<summary>Advanced: custom data path or uv project mode</summary>

```json
{
  "mcpServers": {
    "aimemory": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/AIMemory", "aimemory-mcp"],
      "env": {
        "AIMEMORY_DB_PATH": "/path/to/memory_db"
      }
    }
  }
}
```
</details>

### 4. Connect to Claude Code

```bash
claude mcp add aimemory -- aimemory-mcp
```

---

## MCP Tools (12)

| Tool | Description |
|------|-------------|
| `auto_search` | Auto-retrieve relevant memories at turn start (multi-resolution context) |
| `memory_save` | Save a new memory with keywords, category, and relations |
| `memory_search` | Semantic similarity search |
| `memory_update` | Update content or keywords of an existing memory |
| `memory_delete` | Delete a memory (respects immutability) |
| `memory_get_related` | BFS graph traversal for related memories |
| `memory_pin` / `memory_unpin` | Protect memories from forgetting |
| `memory_stats` | Total count and category breakdown |
| `sleep_cycle_run` | Trigger maintenance (consolidation + forgetting + checkpoint) |
| `policy_status` | RL policy state (epsilon, action distribution, updates) |
| `policy_decide` | Ask the RL policy for a SAVE/SKIP/RETRIEVE decision with reasoning |

---

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AIMEMORY_DB_PATH` | `./memory_db` | ChromaDB persistence directory |
| `AIMEMORY_LANGUAGE` | `ko` | Language for pattern matching (`ko` / `en`) |
| `AIMEMORY_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Sentence-transformer model |
| `AIMEMORY_LOG_LEVEL` | `INFO` | Logging level |
| `AIMEMORY_ENHANCED_POLICY` | `0` | Enable 778d enhanced RL policy (`1` to enable) |
| `AIMEMORY_GRAPH_RAG` | `0` | Enable GraphRAG hybrid retrieval (`1` to enable) |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   MCP Client                     │
│          (Claude Desktop / Claude Code)          │
└────────────────────┬────────────────────────────┘
                     │ stdio (JSON-RPC)
┌────────────────────▼────────────────────────────┐
│              FastMCP Server (12 tools)           │
├──────────────────────────────────────────────────┤
│              MemoryBridge (orchestrator)          │
├──────────┬──────────┬──────────┬─────────────────┤
│ RL Policy│ Retrieval│ Storage  │ Maintenance      │
│          │          │          │                  │
│ Rule-    │ ChromaDB │ Graph    │ Sleep Cycle      │
│ Based +  │ vector + │ Memory   │ (consolidation,  │
│ MLP      │ Knowledge│ Store    │  forgetting,     │
│ Bandit   │ Graph    │          │  checkpoints)    │
│          │ (GraphRAG)│         │                  │
│ Re-ranker│          │          │                  │
│ (11d MLP)│          │          │                  │
└──────────┴──────────┴──────────┴─────────────────┘
```

---

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/ihwooMil/mcp-memory-server.git
cd mcp-memory-server
uv sync --extra dev

# Run tests (611 tests)
uv run pytest tests/ -q

# Lint & format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
