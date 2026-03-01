"""CLI tool to inject AIMemory instructions into AI client configuration files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib import resources
from pathlib import Path

START_MARKER = "<!-- aimemory:start -->"
END_MARKER = "<!-- aimemory:end -->"


def _load_template(name: str) -> str:
    """Load an instruction template from the setup_instructions package."""
    ref = resources.files("aimemory.setup_instructions").joinpath(name)
    return ref.read_text(encoding="utf-8")


def _inject_block(file_path: Path, block_content: str) -> None:
    """Inject or replace a managed block in a file.

    If the file contains START_MARKER...END_MARKER, the block between them
    is replaced. Otherwise the block is appended to the end of the file.
    If the file does not exist, it is created.
    """
    managed_block = f"{START_MARKER}\n{block_content}\n{END_MARKER}\n"

    if file_path.exists():
        text = file_path.read_text(encoding="utf-8")
        start_idx = text.find(START_MARKER)
        end_idx = text.find(END_MARKER)

        if start_idx != -1 and end_idx != -1:
            # Replace existing block
            before = text[:start_idx]
            after = text[end_idx + len(END_MARKER) :]
            # Strip trailing newline from after to avoid double newlines
            if after.startswith("\n"):
                after = after[1:]
            new_text = before + managed_block + after
        else:
            # Append to end
            if text and not text.endswith("\n"):
                text += "\n"
            new_text = text + "\n" + managed_block
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        new_text = managed_block

    file_path.write_text(new_text, encoding="utf-8")


def _find_aimemory_mcp_command() -> str:
    """Find the absolute path to the ``aimemory-mcp`` executable."""
    mcp_bin = shutil.which("aimemory-mcp")
    if mcp_bin:
        return str(Path(mcp_bin).resolve())
    # Fallback: look for it in the current venv
    import sysconfig

    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        candidate = Path(scripts_dir) / "aimemory-mcp"
        if candidate.exists():
            return str(candidate.resolve())
    return "aimemory-mcp"


def _install_openclaw_extension(openclaw_dir: Path) -> None:
    """Install the AIMemory extension into OpenClaw's extensions directory.

    Creates ~/.openclaw/extensions/aimemory/ with the plugin files that
    bridge aimemory MCP tools into OpenClaw's native tool system.
    """
    ext_dir = openclaw_dir / "extensions" / "aimemory"
    ext_dir.mkdir(parents=True, exist_ok=True)

    mcp_command = _find_aimemory_mcp_command()

    # package.json
    package_json = {
        "name": "aimemory",
        "version": "0.1.0",
        "description": "AIMemory long-term memory tools for OpenClaw",
        "type": "module",
        "main": "index.ts",
        "files": ["index.ts", "openclaw.plugin.json"],
        "openclaw": {"extensions": ["./index.ts"]},
    }
    (ext_dir / "package.json").write_text(
        json.dumps(package_json, indent=2) + "\n", encoding="utf-8"
    )

    # openclaw.plugin.json
    plugin_json = {
        "id": "aimemory",
        "name": "AIMemory",
        "description": "Persistent long-term memory with semantic search and knowledge graphs",
        "configSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    }
    (ext_dir / "openclaw.plugin.json").write_text(
        json.dumps(plugin_json, indent=2) + "\n", encoding="utf-8"
    )

    # index.ts — bridge plugin
    index_ts = _load_template("openclaw_extension.ts")
    # Inject the resolved MCP command path
    index_ts = index_ts.replace("__AIMEMORY_MCP_COMMAND__", mcp_command)
    (ext_dir / "index.ts").write_text(index_ts, encoding="utf-8")

    print(f"  Installed extension to {ext_dir}")


def setup_openclaw(workspace: Path) -> None:
    """Inject AIMemory instructions into OpenClaw workspace files."""
    workspace = workspace.expanduser()

    # SOUL.md — Memory Continuity section
    soul_path = workspace / "SOUL.md"
    soul_content = _load_template("openclaw_soul.md")
    _inject_block(soul_path, soul_content)
    print(f"  Updated {soul_path}")

    # TOOLS.md — AIMemory tool section
    tools_path = workspace / "TOOLS.md"
    tools_content = _load_template("openclaw_tools.md")
    _inject_block(tools_path, tools_content)
    print(f"  Updated {tools_path}")

    # Install OpenClaw extension
    openclaw_dir = workspace.parent
    _install_openclaw_extension(openclaw_dir)


def setup_claude(claude_dir: Path) -> None:
    """Inject AIMemory instructions into Claude Code's CLAUDE.md."""
    claude_dir = claude_dir.expanduser()
    claude_md = claude_dir / "CLAUDE.md"
    claude_content = _load_template("claude.md")
    _inject_block(claude_md, claude_content)
    print(f"  Updated {claude_md}")


def main() -> None:
    """Entry point for aimemory-setup CLI."""
    parser = argparse.ArgumentParser(
        prog="aimemory-setup",
        description="Setup AIMemory instructions for AI clients",
    )
    subparsers = parser.add_subparsers(dest="client")

    # openclaw subcommand
    oc = subparsers.add_parser("openclaw", help="Setup for OpenClaw")
    oc.add_argument(
        "--workspace",
        default="~/.openclaw/workspace",
        help="OpenClaw workspace path (default: ~/.openclaw/workspace)",
    )

    # claude subcommand
    subparsers.add_parser("claude", help="Setup for Claude Code")

    args = parser.parse_args()

    if args.client is None:
        parser.print_help()
        sys.exit(1)

    print(f"Setting up AIMemory for {args.client}...")

    if args.client == "openclaw":
        setup_openclaw(Path(args.workspace))
    elif args.client == "claude":
        setup_claude(Path("~/.claude"))

    print("Done!")


if __name__ == "__main__":
    main()
