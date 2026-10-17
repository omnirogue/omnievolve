# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _support import ROOT

from omnievolve.mcp_server import Server
from omnievolve.templates import FILES


class McpTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="omnievolve-mcp-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, True)
        for name, text in FILES.items():
            (self.dir / name).write_text(text)
        self.server = Server(self.dir)

    def call(self, method, params=None, msg_id=1):
        return self.server.handle(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}
        )

    def tool(self, name, **arguments):
        return self.call("tools/call", {"name": name, "arguments": arguments})["result"]

    def test_handshake_and_tool_list(self):
        init = self.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
        self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(init["result"]["serverInfo"]["name"], "omnievolve")
        newer = self.call("initialize", {"protocolVersion": "2099-01-01"})
        self.assertEqual(newer["result"]["protocolVersion"], "2025-11-25")
        self.assertIsNone(
            self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        )
        names = [t["name"] for t in self.call("tools/list")["result"]["tools"]]
        self.assertEqual(
            names, ["omnievolve_run", "omnievolve_leaderboard", "omnievolve_export_winner"]
        )
        self.assertEqual(self.call("ping")["result"], {})
        self.assertEqual(self.call("nope")["error"]["code"], -32601)
        self.assertEqual(self.call("tools/call", {"name": "nope"})["error"]["code"], -32602)

    def test_run_leaderboard_and_export(self):
        run = self.tool("omnievolve_run", config_path="omnievolve.json", generations=2, workers=2)
        self.assertFalse(run["isError"], run["content"][0]["text"])
        summary = run["structuredContent"]
        self.assertTrue(summary["improved"])
        self.assertEqual(json.loads(run["content"][0]["text"])["run_id"], summary["run_id"])

        board = self.tool("omnievolve_leaderboard", limit=3)["structuredContent"]
        self.assertEqual(board["run_id"], summary["run_id"])
        self.assertNotIn("source", board["leaderboard"][0])

        export = self.tool("omnievolve_export_winner", output_path="out/best.patch")
        data = export["structuredContent"]
        self.assertTrue(data["patch"].startswith("--- a/program.py"))
        self.assertTrue((self.dir / "out" / "best.patch").is_file())

    def test_paths_must_stay_inside_the_root(self):
        for args in (
            {"config_path": "../elsewhere/omnievolve.json"},
            {"config_path": "/etc/hosts"},
            {"config_path": "omnievolve.json", "ledger_path": "../x.db"},
        ):
            with self.subTest(args=args):
                result = self.tool("omnievolve_run", **args)
                self.assertTrue(result["isError"])
                self.assertIn("inside the server root", result["content"][0]["text"])
        missing = self.tool("omnievolve_leaderboard", ledger_path="none.db")
        self.assertTrue(missing["isError"])

    def test_stdio_transport(self):
        lines = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
        stdin = "\n".join(json.dumps(line) for line in lines) + "\nnot json\n"
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
        done = subprocess.run(
            [sys.executable, "-m", "omnievolve.mcp_server", "--root", str(self.dir)],
            input=stdin,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        replies = [json.loads(line) for line in done.stdout.splitlines()]
        self.assertEqual([r.get("id") for r in replies], [1, 2, None])
        self.assertEqual(replies[2]["error"]["code"], -32700)
        self.assertEqual(len(replies[1]["result"]["tools"]), 3)


if __name__ == "__main__":
    unittest.main()
