# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""A standalone MCP server over stdio (JSON-RPC 2.0, one message per line).

Tools:

* ``omnievolve_run``: evolve a project from its config file and return the summary.
* ``omnievolve_leaderboard``: the best candidates of a run in a ledger.
* ``omnievolve_export_winner``: the seed-to-winner unified diff (optionally written to a file).

Every path must resolve inside ``--root`` (default: the working directory), because running a
project executes its evaluator. Standard output carries protocol messages only; diagnostics go
to standard error.

    omnievolve-mcp --root /path/to/projects
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from ._version import __version__
from .config import DEFAULT_LEDGER, ConfigError, load_config
from .engine import EvolutionError
from .ledger import Ledger, LedgerError
from .llm import LLMError

SUPPORTED_PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = (
    -32700,
    -32600,
    -32601,
    -32602,
    -32603,
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "omnievolve_run",
        "title": "Run an OmniEvolve search",
        "description": (
            "Evolve the target file of an OmniEvolve project (omnievolve.json) locally: "
            "island-model search scored by the project's evaluator. Records every candidate in "
            "the SQLite ledger and returns the best score, the improvement over the seed and the "
            "ledger path. Runs the project's evaluator, so only use it on projects you trust."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "config_path": {"type": "string", "description": "Path to omnievolve.json"},
                "generations": {"type": "integer", "minimum": 0, "maximum": 1000},
                "islands": {"type": "integer", "minimum": 1, "maximum": 64},
                "population": {"type": "integer", "minimum": 2, "maximum": 1000},
                "seed": {"type": "integer"},
                "workers": {"type": "integer", "minimum": 1, "maximum": 64},
                "ledger_path": {"type": "string", "description": "Ledger database path"},
                "html_path": {"type": "string", "description": "Also write the HTML report here"},
            },
            "required": ["config_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "omnievolve_leaderboard",
        "title": "OmniEvolve leaderboard",
        "description": "The best candidates of a run (default: the latest) in an OmniEvolve "
        "ledger.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ledger_path": {"type": "string", "description": f"Default {DEFAULT_LEDGER}"},
                "run_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "include_source": {"type": "boolean", "description": "Include each file's source"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "omnievolve_export_winner",
        "title": "Export the OmniEvolve winner",
        "description": (
            "The unified diff from the seed to the best candidate of a run, ready for git apply. "
            "Writes it to output_path when given."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ledger_path": {"type": "string", "description": f"Default {DEFAULT_LEDGER}"},
                "run_id": {"type": "string"},
                "output_path": {"type": "string", "description": "Write the patch here"},
            },
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """A tool call failed in a way the model should see (returned with ``isError``)."""


class Server:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "omnievolve_run": self.tool_run,
            "omnievolve_leaderboard": self.tool_leaderboard,
            "omnievolve_export_winner": self.tool_export_winner,
        }

    # -- paths ------------------------------------------------------------------------------

    def path(self, value: Any, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ToolError(f"{name} must be a non-empty string")
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else self.root / path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ToolError(f"{name} must be inside the server root {self.root}")
        return path

    # -- tools ------------------------------------------------------------------------------

    def tool_run(self, args: dict[str, Any]) -> dict[str, Any]:
        from .runner import run_project

        config = self.path(args.get("config_path"), "config_path")
        overrides = {
            k: args.get(k) for k in ("generations", "islands", "population", "seed", "workers")
        }
        for key, value in overrides.items():
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ToolError(f"{key} must be an integer")
        project = load_config(config, overrides)
        ledger = self.path(args["ledger_path"], "ledger_path") if args.get("ledger_path") else None
        html = self.path(args["html_path"], "html_path") if args.get("html_path") else None
        if ledger is None:
            self.path(str(project.ledger), "the config's ledger")
        if html is None and project.report is not None:
            self.path(str(project.report), "the config's report")
        result, report = run_project(project, ledger_path=ledger, html_path=html)
        summary = result.summary()
        summary["report"] = str(report) if report else None
        return summary

    def _ledger(self, args: dict[str, Any]) -> Ledger:
        return Ledger(
            self.path(args.get("ledger_path") or DEFAULT_LEDGER, "ledger_path"), create=False
        )

    def tool_leaderboard(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = args.get("limit", 10)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ToolError("limit must be an integer from 1 to 100")
        with self._ledger(args) as ledger:
            run = ledger.run(args.get("run_id"))
            rows = ledger.leaderboard(run["run_id"], limit=limit)
            seed = ledger.candidate(run["run_id"], "seed")
        if not args.get("include_source"):
            rows = [{k: v for k, v in row.items() if k != "source"} for row in rows]
        return {
            "run_id": run["run_id"],
            "name": run["name"],
            "objective": run["objective"],
            "status": run["status"],
            "seed_score": seed["score"],
            "leaderboard": rows,
        }

    def tool_export_winner(self, args: dict[str, Any]) -> dict[str, Any]:
        with self._ledger(args) as ledger:
            run = ledger.run(args.get("run_id"))
            patch, best = ledger.winner_patch(run["run_id"])
            seed = ledger.candidate(run["run_id"], "seed")
        written = None
        if args.get("output_path"):
            output = self.path(args["output_path"], "output_path")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(patch, encoding="utf-8")
            written = str(output)
        return {
            "run_id": run["run_id"],
            "target": run["target"],
            "winner": best["candidate_id"],
            "seed_score": seed["score"],
            "best_score": best["score"],
            "patch": patch,
            "output_path": written,
        }

    # -- protocol ---------------------------------------------------------------------------

    def handle(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "expected a JSON-RPC 2.0 object")
        method = message.get("method")
        has_id = "id" in message
        msg_id = message.get("id")
        if not isinstance(method, str):
            return _error(msg_id, INVALID_REQUEST, "missing method") if has_id else None
        params = message.get("params") or {}
        if not has_id:
            return None  # notifications (initialized, cancelled, ...) need no reply
        try:
            if method == "initialize":
                requested = params.get("protocolVersion")
                version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
                result = {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "omnievolve",
                        "title": "OmniEvolve",
                        "version": __version__,
                    },
                    "instructions": (
                        "Local evolutionary optimization of code and prompts. Run a project with "
                        "omnievolve_run, inspect it with omnievolve_leaderboard and fetch the "
                        "winning patch with omnievolve_export_winner."
                    ),
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self.call_tool(params)
            else:
                return _error(msg_id, METHOD_NOT_FOUND, f"unknown method {method}")
        except ToolError as exc:
            return _error(msg_id, INVALID_PARAMS, str(exc))
        except Exception as exc:  # never let one request kill the server
            traceback.print_exc(file=sys.stderr)
            return _error(msg_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        handler = self.handlers.get(name)  # type: ignore[arg-type]
        if handler is None:
            raise ToolError(f"unknown tool {name!r}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be an object")
        try:
            payload = handler(arguments)
        except (
            ToolError,
            ConfigError,
            LedgerError,
            EvolutionError,
            LLMError,
            ValueError,
            OSError,
        ) as exc:
            return {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True}
        return {
            "content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}],
            "structuredContent": payload,
            "isError": False,
        }

    def serve(self, stdin: TextIO, stdout: TextIO) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                response: dict[str, Any] | None = _error(None, PARSE_ERROR, "invalid JSON")
            else:
                response = self.handle(message)
            if response is not None:
                stdout.write(json.dumps(response, default=str) + "\n")
                stdout.flush()


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnievolve-mcp", description="OmniEvolve MCP server (stdio, JSON-RPC 2.0)."
    )
    parser.add_argument("--root", default=".", help="directory every tool path must stay inside")
    parser.add_argument("--version", action="version", version=f"omnievolve-mcp {__version__}")
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"omnievolve-mcp: --root {root} is not a directory", file=sys.stderr)
        return 2
    try:
        Server(root).serve(sys.stdin, sys.stdout)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
