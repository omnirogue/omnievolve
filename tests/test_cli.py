# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _support import EXAMPLES

from omnievolve.cli import main


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="omnievolve-cli-"))
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_init_run_and_inspect(self):
        project = self.dir / "proj"
        code, out, _ = cli("init", str(project))
        self.assertEqual(code, 0)
        self.assertTrue((project / "omnievolve.json").is_file())
        self.assertEqual(cli("init", str(project))[0], 1)  # refuses to overwrite

        db = self.dir / "ledger.db"
        html = self.dir / "report.html"
        code, out, err = cli(
            "run",
            "--config",
            str(project / "omnievolve.json"),
            "--generations",
            "3",
            "--ledger",
            str(db),
            "--html",
            str(html),
            "--json",
        )
        self.assertEqual(code, 0, err)
        summary = json.loads(out)
        self.assertTrue(summary["improved"])
        self.assertGreater(summary["best_score"], summary["seed_score"])
        self.assertTrue(html.is_file())
        self.assertIn("gen   3", err)  # progress goes to stderr with --json

        code, out, _ = cli("leaderboard", "--ledger", str(db), "--json", "-n", "3")
        board = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(board["leaderboard"][0]["candidate_id"], summary["best_candidate_id"])

        patch, best = self.dir / "best.patch", self.dir / "best.py"
        code, out, _ = cli(
            "export-winner", "--ledger", str(db), "-o", str(patch), "--source-output", str(best)
        )
        self.assertEqual(code, 0)
        self.assertTrue(patch.read_text().startswith("--- a/program.py"))
        self.assertIn("EVOLVE-BLOCK-START", best.read_text())

        code, out, _ = cli("report", "--ledger", str(db), "-o", str(self.dir / "r2.html"))
        self.assertEqual(code, 0)
        code, out, _ = cli("runs", "--ledger", str(db))
        self.assertIn(summary["run_id"], out)

    def test_errors_are_reported_not_raised(self):
        code, _, err = cli("leaderboard", "--ledger", str(self.dir / "missing.db"))
        self.assertEqual(code, 1)
        self.assertIn("no ledger", err)
        code, _, err = cli("run", "--config", str(self.dir / "missing.json"))
        self.assertEqual(code, 1)
        self.assertIn("no config", err)

    def test_cloud_dry_run_sends_nothing(self):
        project = self.dir / "heuristic"
        shutil.copytree(EXAMPLES / "heuristic_tuning", project)
        (project / ".env").write_text("SECRET=1\n")
        code, out, err = cli(
            "run", "--config", str(project / "omnievolve.json"), "--cloud", "--dry-run"
        )
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        files = payload["spec"]["files"]
        self.assertIn("evaluate.py", files)
        self.assertIsInstance(files["evaluate.py"], int)  # sizes only, never contents
        self.assertNotIn(".env", files)
        self.assertIn(".env", payload["spec"]["skipped_files"])
        self.assertEqual(payload["spec"]["target"]["path"], "triage.py")
        self.assertIn("nothing was sent", err)

    def test_version(self):
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(io.StringIO()):
            main(["--version"])
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
