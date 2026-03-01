# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-01

### Changed

- **Project renamed** from `mcp-memory-server` to `long-term-memory`. Python import name (`aimemory`) and CLI commands unchanged.
- **GitHub repository** renamed to `long-term-memory` (old URL auto-redirects)

### Added

- **Auto-linking** — New memories automatically link to similar existing ones (similarity ≥ 0.92) via bidirectional graph edges
- **Live graph visualization** (`aimemory-live`) — Real-time WebSocket-based memory graph in the browser
  - Force-directed layout (vis.js ForceAtlas2) with category-based coloring
  - Glow effects on save (green) and search (blue)
  - Event log sidebar with hover-to-highlight related nodes
  - Persistent event history across browser refreshes
  - Cross-process event push (MCP → live server via WebSocket `/event` path)
- **`memory_visualize` tool** — Generate interactive static HTML graph (13th MCP tool)
- **`[live]` optional dependency** — `websockets>=12.0`

### Deprecated

- PyPI package `mcp-memory-server` — use `long-term-memory` instead

## [0.2.0] - 2026-02-28

First public release.

### Added

- **MCP Server** with 12 tools: `auto_search`, `memory_save`, `memory_search`, `memory_update`, `memory_delete`, `memory_get_related`, `memory_pin`, `memory_unpin`, `memory_stats`, `sleep_cycle_run`, `policy_status`, `policy_decide`
- **RL Memory Policy** — Rule-based importance scoring + MLP contextual bandit (SAVE/SKIP/RETRIEVE)
- **Enhanced Policy** (opt-in) — 778d state encoder (SentenceTransformer 768d + 10d hand-crafted), experience replay buffer, progressive autonomy
- **Semantic Search** — ChromaDB vector store with `intfloat/multilingual-e5-small` embeddings
- **Knowledge Graph** — NetworkX-based entity-relation graph with multi-hop traversal
- **GraphRAG Hybrid Retrieval** (opt-in) — Vector similarity + graph traversal with RL re-ranker (11d features)
- **Multi-Resolution Text** — 3 levels (full text, keyword summary, entity triples) with token-budget-aware composition
- **Forgetting Pipeline** — Decay-based aging with compress → deactivate → delete stages
- **Sleep Cycle** — Periodic maintenance: consolidation, resolution regeneration, forgetting, checkpoint saving
- **Immutable Memories** — Protected from modification/deletion with SHA-256 rule hash verification
- **Pin/Unpin** — User-controlled forgetting protection
- **Multilingual Support** — Korean (ko) and English (en) i18n patterns
- **P2P Federated Learning** — Gossip protocol with differential privacy and Krum aggregation
- **A/B Comparison Framework** — Baseline vs re-ranked retrieval comparison
- **CI/CD** — GitHub Actions with Python 3.11/3.12/3.13 matrix
- **611 tests** passing across 31 test files

### Removed

- Deprecated DualHeadDQN offline training pipeline (replaced by online MLP bandit)
- `ExtractorConfig` (unused after DQN removal)
