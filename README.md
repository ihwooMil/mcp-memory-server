# Long-Term Memory

**Persistent, self-organizing memory for AI assistants.**

> Drop-in MCP server that gives Claude (and any MCP client) long-term memory — powered by semantic search, knowledge graphs, and reinforcement learning.

[![CI](https://github.com/ihwooMil/long-term-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/ihwooMil/long-term-memory/actions)
[![PyPI](https://img.shields.io/pypi/v/long-term-memory)](https://pypi.org/project/long-term-memory/)
[![Python](https://img.shields.io/pypi/pyversions/long-term-memory)](https://pypi.org/project/long-term-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note:** This package was previously published as [`mcp-memory-server`](https://pypi.org/project/mcp-memory-server/). That package is deprecated — please use `long-term-memory` going forward.

---

## Why Long-Term Memory?

Current AI memory tools have two critical problems:

| Problem | How we solve it |
|---------|----------------|
| **Manual retrieval** — you must ask "do you remember X?" | `auto_search` runs every turn, injecting relevant memories automatically |
| **Token waste** — entire memory dump inserted into context | Multi-resolution composer selects top-K memories within a token budget |

## Key Features

- **RL-powered policy** — Contextual bandit decides when to save, skip, or retrieve (not just keyword matching)
- **Semantic search** — ChromaDB + multilingual sentence-transformer embeddings (`intfloat/multilingual-e5-small`)
- **Knowledge graph** — Entity-relation graph (NetworkX) for multi-hop reasoning
- **GraphRAG hybrid retrieval** — Vector similarity + graph traversal, fused and re-ranked by an RL re-ranker
- **Auto-linking** — New memories automatically link to similar existing ones (similarity ≥ 0.92)
- **Multi-resolution text** — Full text → summary → entity triples, composed within token budget
- **Forgetting pipeline** — Decay-based aging with consolidation, pinning, and immutable protection
- **Sleep cycle** — Periodic maintenance: dedup, compress, forget, checkpoint
- **Live graph** — Real-time WebSocket visualization of the memory graph
- **Multilingual** — Korean and English pattern support out of the box

---

## Quick Start (2 minutes)

### 1. Install

```bash
pip install long-term-memory
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install long-term-memory
```

<details>
<summary>Optional extras</summary>

```bash
pip install long-term-memory[ko]     # Korean NLP support
pip install long-term-memory[live]   # Real-time graph visualization
pip install long-term-memory[viz]    # Static graph visualization
```
</details>

### 2. Setup client instructions

```bash
# For OpenClaw
aimemory-setup openclaw

# For Claude Code
aimemory-setup claude
```

This injects memory usage instructions into your client's configuration files (`SOUL.md`/`TOOLS.md` for OpenClaw, `CLAUDE.md` for Claude Code). Re-run anytime to update.

### 3. Connect to OpenClaw

```bash
mcporter config add aimemory --command aimemory-mcp --scope home
```

### 4. Connect to Claude Desktop

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
<summary>With live graph visualization</summary>

```json
{
  "mcpServers": {
    "aimemory": {
      "command": "aimemory-mcp",
      "args": ["--with-live"]
    }
  }
}
```

Then open `http://127.0.0.1:8765` to see the live memory graph.
</details>

<details>
<summary>Advanced: custom data path or uv project mode</summary>

```json
{
  "mcpServers": {
    "aimemory": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/long-term-memory", "aimemory-mcp", "--with-live"],
      "env": {
        "AIMEMORY_DB_PATH": "/path/to/memory_db"
      }
    }
  }
}
```
</details>

### 5. Connect to Claude Code

```bash
claude mcp add aimemory -- aimemory-mcp
```

Or with live graph:

```bash
claude mcp add aimemory -- aimemory-mcp --with-live
```

---

## Live Graph Visualization

Real-time WebSocket-based memory graph that updates as memories are saved, searched, or deleted.

```bash
# Option 1: auto-start with MCP server
aimemory-mcp --with-live

# Option 2: standalone server
aimemory-live --port 8765

# Option 3: via environment variable
AIMEMORY_LIVE=1 aimemory-mcp
```

Open `http://127.0.0.1:8765` in a browser. Requires the `[live]` extra (`pip install long-term-memory[live]`). Features:

- Force-directed graph layout with category-based coloring
- New nodes glow green on save, blue on search
- Event log sidebar with hover-to-highlight (hover a log entry to highlight related nodes)
- Persistent event history across browser refreshes
- Cross-process events — MCP server pushes events to the live graph via WebSocket

---

## MCP Tools (13)

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
| `memory_visualize` | Generate interactive graph HTML |
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
| `AIMEMORY_LIVE_HOST` | `127.0.0.1` | Live graph server host (for event push) |
| `AIMEMORY_LIVE_PORT` | `8765` | Live graph server port (for event push) |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   MCP Client                     │
│          (Claude Desktop / Claude Code)          │
└────────────────────┬────────────────────────────┘
                     │ stdio (JSON-RPC)
┌────────────────────▼────────────────────────────┐
│              FastMCP Server (13 tools)           │
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
         ↕ WebSocket (cross-process)
┌──────────────────────────────────────────────────┐
│          Live Graph Server (aimemory-live)        │
│     vis.js force-directed graph + event log      │
└──────────────────────────────────────────────────┘
```

---

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/ihwooMil/long-term-memory.git
cd long-term-memory
uv sync --extra dev

# Run tests (611+ tests)
uv run pytest tests/ -q

# Lint & format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

---

## Migrating from mcp-memory-server

```bash
pip uninstall mcp-memory-server
pip install long-term-memory
```

No code changes needed — the Python import name (`aimemory`) and CLI commands (`aimemory-mcp`, `aimemory-viz`, `aimemory-live`) remain the same.

---

## License

MIT — see [LICENSE](LICENSE) for details.
