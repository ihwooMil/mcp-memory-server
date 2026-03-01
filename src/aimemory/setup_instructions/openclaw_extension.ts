import { spawn, type ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";

// ── MCP stdio client ────────────────────────────────────────────────

const MCP_COMMAND = "__AIMEMORY_MCP_COMMAND__";

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

let mcpProcess: ChildProcess | null = null;
let requestId = 0;
const pendingRequests = new Map<
  number,
  { resolve: (v: unknown) => void; reject: (e: Error) => void }
>();
let initialized = false;
let initPromise: Promise<void> | null = null;

function ensureMcp(): ChildProcess {
  if (mcpProcess && !mcpProcess.killed) return mcpProcess;

  mcpProcess = spawn(MCP_COMMAND, [], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env },
  });

  const rl = createInterface({ input: mcpProcess.stdout! });
  rl.on("line", (line) => {
    try {
      const msg = JSON.parse(line) as JsonRpcResponse;
      if (msg.id != null && pendingRequests.has(msg.id)) {
        const p = pendingRequests.get(msg.id)!;
        pendingRequests.delete(msg.id);
        if (msg.error) {
          p.reject(new Error(msg.error.message));
        } else {
          p.resolve(msg.result);
        }
      }
    } catch {
      /* ignore non-JSON lines */
    }
  });

  mcpProcess.on("exit", () => {
    mcpProcess = null;
    initialized = false;
    initPromise = null;
    for (const [id, p] of pendingRequests) {
      p.reject(new Error("MCP process exited"));
      pendingRequests.delete(id);
    }
  });

  // Initialize MCP handshake
  initPromise = sendRpc("initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "openclaw-aimemory", version: "0.1.0" },
  }).then(() => {
    const notification = JSON.stringify({
      jsonrpc: "2.0",
      method: "notifications/initialized",
    });
    mcpProcess?.stdin?.write(notification + "\n");
    initialized = true;
  });

  return mcpProcess;
}

function sendRpc(method: string, params?: Record<string, unknown>): Promise<unknown> {
  const proc = ensureMcp();
  const id = ++requestId;
  const req: JsonRpcRequest = { jsonrpc: "2.0", id, method, params };

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pendingRequests.delete(id);
      reject(new Error(`MCP request timed out: ${method}`));
    }, 30_000);

    pendingRequests.set(id, {
      resolve: (v) => {
        clearTimeout(timeout);
        resolve(v);
      },
      reject: (e) => {
        clearTimeout(timeout);
        reject(e);
      },
    });

    proc.stdin!.write(JSON.stringify(req) + "\n");
  });
}

async function callTool(
  toolName: string,
  args: Record<string, unknown>,
): Promise<string> {
  // Wait for initialization before calling tools
  if (!initialized && initPromise) {
    await initPromise;
  }

  const result = (await sendRpc("tools/call", {
    name: toolName,
    arguments: args,
  })) as { content?: { type: string; text: string }[] };

  if (result?.content?.[0]?.text) {
    return result.content[0].text;
  }
  return JSON.stringify(result);
}

// ── Tool executor helpers ───────────────────────────────────────────

async function execTool(
  _toolCallId: string,
  params: Record<string, unknown>,
  toolName: string,
) {
  try {
    const text = await callTool(toolName, params);
    return { content: [{ type: "text" as const, text }] };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      content: [{ type: "text" as const, text: JSON.stringify({ success: false, error: msg }) }],
    };
  }
}

// ── Plugin registration ─────────────────────────────────────────────

const aimemoryPlugin = {
  id: "aimemory",
  name: "AIMemory",
  description:
    "Persistent long-term memory with semantic search and knowledge graphs",
  configSchema: { parse: (v: unknown) => v ?? {} },
  register(api: any) {
    api.registerTool({
      name: "auto_search",
      label: "Auto Search Memories",
      description:
        "Automatically retrieve and compose relevant memories for a user message. Call this at the start of every conversation turn.",
      parameters: {
        type: "object",
        properties: {
          user_message: { type: "string", description: "The user's current message" },
          token_budget: { type: "number", description: "Max tokens for context (default: 1024)" },
          top_k: { type: "number", description: "Number of memories to retrieve (default: 10)" },
        },
        required: ["user_message"],
      },
      execute: (id: string, p: any) => execTool(id, p, "auto_search"),
    } as any);

    api.registerTool({
      name: "memory_save",
      label: "Save Memory",
      description: "Save a new memory to the knowledge graph.",
      parameters: {
        type: "object",
        properties: {
          content: { type: "string", description: "The memory content to save" },
          keywords: {
            type: "array",
            items: { type: "string" },
            description: "Keywords for retrieval",
          },
          category: {
            type: "string",
            enum: ["fact", "preference", "experience", "emotion", "technical", "core_principle"],
            description: "Memory category (default: fact)",
          },
          related_ids: {
            type: "array",
            items: { type: "string" },
            description: "IDs of related memories to link",
          },
          pinned: { type: "boolean", description: "Protect from forgetting" },
        },
        required: ["content"],
      },
      execute: (id: string, p: any) => execTool(id, p, "memory_save"),
    } as any);

    api.registerTool({
      name: "memory_search",
      label: "Search Memories",
      description: "Search memories by semantic similarity.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query text" },
          top_k: { type: "number", description: "Number of results (default: 5)" },
          category: { type: "string", description: "Optional category filter" },
        },
        required: ["query"],
      },
      execute: (id: string, p: any) => execTool(id, p, "memory_search"),
    } as any);

    api.registerTool({
      name: "memory_update",
      label: "Update Memory",
      description: "Update an existing memory's content and/or keywords.",
      parameters: {
        type: "object",
        properties: {
          memory_id: { type: "string", description: "The memory ID to update" },
          content: { type: "string", description: "New content (optional)" },
          keywords: {
            type: "array",
            items: { type: "string" },
            description: "New keywords (optional)",
          },
        },
        required: ["memory_id"],
      },
      execute: (id: string, p: any) => execTool(id, p, "memory_update"),
    } as any);

    api.registerTool({
      name: "memory_delete",
      label: "Delete Memory",
      description: "Delete a memory from the knowledge graph.",
      parameters: {
        type: "object",
        properties: {
          memory_id: { type: "string", description: "The memory ID to delete" },
        },
        required: ["memory_id"],
      },
      execute: (id: string, p: any) => execTool(id, p, "memory_delete"),
    } as any);

    api.registerTool({
      name: "memory_get_related",
      label: "Get Related Memories",
      description: "Get memories related to a given memory via graph edges.",
      parameters: {
        type: "object",
        properties: {
          memory_id: { type: "string", description: "The starting memory ID" },
          depth: { type: "number", description: "Hops to traverse (default: 1, max: 3)" },
        },
        required: ["memory_id"],
      },
      execute: (id: string, p: any) => execTool(id, p, "memory_get_related"),
    } as any);

    api.registerTool({
      name: "memory_pin",
      label: "Pin Memory",
      description: "Pin a memory to protect it from forgetting.",
      parameters: {
        type: "object",
        properties: {
          memory_id: { type: "string", description: "The memory ID to pin" },
        },
        required: ["memory_id"],
      },
      execute: (id: string, p: any) => execTool(id, p, "memory_pin"),
    } as any);

    api.registerTool({
      name: "memory_unpin",
      label: "Unpin Memory",
      description: "Remove pin protection from a memory.",
      parameters: {
        type: "object",
        properties: {
          memory_id: { type: "string", description: "The memory ID to unpin" },
        },
        required: ["memory_id"],
      },
      execute: (id: string, p: any) => execTool(id, p, "memory_unpin"),
    } as any);

    api.registerTool({
      name: "memory_stats",
      label: "Memory Statistics",
      description: "Get statistics about the memory store.",
      parameters: {
        type: "object",
        properties: {},
      },
      execute: (id: string, p: any) => execTool(id, p, "memory_stats"),
    } as any);

    api.registerTool({
      name: "memory_visualize",
      label: "Visualize Memory Graph",
      description: "Generate an interactive HTML visualization of the memory graph.",
      parameters: {
        type: "object",
        properties: {
          output_path: { type: "string", description: "Output file path (optional)" },
          include_inactive: {
            type: "boolean",
            description: "Include forgotten memories (default: false)",
          },
        },
      },
      execute: (id: string, p: any) => execTool(id, p, "memory_visualize"),
    } as any);
  },
};

export default aimemoryPlugin;
