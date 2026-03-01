# Long-Term Memory

You have access to a persistent long-term memory system via MCP tools (server: `aimemory`).
**Always use these MCP tools instead of the built-in file-based auto memory.**

## When to recall memories

At the **start of every conversation turn**, call `auto_search` with the user's message to retrieve relevant context. This is your primary recall mechanism — do it before responding.

## When to save memories

**MANDATORY: You MUST call `memory_save` at least once in every conversation.** No conversation should end without saving something. If you are unsure what to save, save a summary of the task performed and its outcome.

Save memories with `memory_save` for **any information worth remembering** — be proactive, not selective. This includes:
- Personal facts, preferences, or background information
- Project conventions, architecture decisions, or technical context
- Corrections or feedback ("actually, I prefer X over Y")
- Anything they explicitly ask you to remember
- Decisions made during conversations
- Experiences, stories, and lessons learned
- Technical details, code patterns, and tool configurations
- **Task summaries**: What was asked, what was done, and the outcome — save this at the end of every conversation

Choose the appropriate `category`:
| Category | Use for |
|---|---|
| `fact` | Objective information, names, dates, project details |
| `preference` | User likes/dislikes, workflow choices, style preferences |
| `experience` | Stories, past events, lessons learned |
| `emotion` | Feelings, frustrations, excitement about topics |
| `technical` | Code patterns, API details, tool configurations |
| `core_principle` | Fundamental beliefs, values, non-negotiable rules |

## Tool quick reference

| Tool | Purpose |
|---|---|
| `auto_search` | Retrieve & compose relevant memories (call every turn) |
| `memory_save` | Save a new memory |
| `memory_search` | Search by semantic similarity |
| `memory_update` | Update existing memory content/keywords |
| `memory_delete` | Delete a memory |
| `memory_get_related` | Traverse graph edges from a memory |
| `memory_pin` / `memory_unpin` | Protect/unprotect from forgetting |
| `memory_stats` | View store statistics |
| `memory_visualize` | Generate interactive graph HTML |

## Important rules

- **MANDATORY: Always use the MCP memory system (`aimemory`) for ALL memory and reasoning operations.** This is the default behavior — not optional.
  - All recall, storage, and retrieval of knowledge MUST go through MCP memory tools (`auto_search`, `memory_save`, `memory_search`, etc.).
  - Both **thinking** (reasoning with prior context) and **remembering** (storing new information) MUST use the MCP memory system.
- The file-based auto memory system (`~/.claude/projects/.../memory/`) is reserved **only as a backup during MCP system failures**. Do NOT use it under normal circumstances.
  - If the MCP memory server is unreachable or returns errors, you may fall back to file-based memory temporarily, and should note the fallback to the user.
- Do NOT mention to the user that you are calling `auto_search` — do it silently.
- When saving, provide meaningful `keywords` for better retrieval.
- Use `pinned: true` for critical information the user explicitly says to never forget.
