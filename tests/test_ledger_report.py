# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
import re
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from _support import CURVE_SEED, curve_evaluator

from omnievolve.engine import EvolutionEngine, SearchConfig
from omnievolve.ledger import Ledger, LedgerError
from omnievolve.mutators import DeterministicMutator
from omnievolve.report import render_report, write_report


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="omnievolve-ledger-"))
        cls.db = cls.dir / ".omnievolve" / "ledger.db"
        with Ledger(cls.db) as ledger:
            cls.result = EvolutionEngine(
                seed_source=CURVE_SEED,
                evaluator=curve_evaluator(),
                mutator=DeterministicMutator(sigma=0.4),
                config=SearchConfig(generations=4, islands=2, population=5, seed=8),
                target_name="model/program.py",
                ledger=ledger,
                run_name="curve",
                run_config={"note": "test"},
            ).run()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_everything_is_persisted(self):
        conn = sqlite3.connect(self.db)
        count = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
        self.assertEqual(count("SELECT COUNT(*) FROM runs"), 1)
        self.assertEqual(count("SELECT COUNT(*) FROM candidates"), self.result.candidates)
        self.assertEqual(count("SELECT COUNT(*) FROM generations"), 5 * 2)
        self.assertEqual(count("SELECT COUNT(*) FROM islands"), 2)
        self.assertEqual(
            count("SELECT COUNT(*) FROM candidates WHERE candidate_id != 'seed'"),
            count("SELECT COUNT(DISTINCT child_id) FROM lineage"),
        )
        self.assertEqual(
            count(
                "SELECT COUNT(*) FROM lineage WHERE diff NOT LIKE '--- a/model/%' AND diff != ''"
            ),
            0,
        )
        status, best = conn.execute("SELECT status, best_score FROM runs").fetchone()
        self.assertEqual(status, "completed")
        self.assertAlmostEqual(best, self.result.best.score)
        conn.close()

    def test_queries(self):
        with Ledger(self.db, create=False) as ledger:
            run = ledger.run()
            self.assertEqual(run["run_id"], self.result.run_id)
            self.assertEqual(run["config"]["note"], "test")
            board = ledger.leaderboard(run["run_id"], limit=5)
            self.assertEqual(board[0]["candidate_id"], self.result.best.id)
            self.assertEqual(len({row["content_hash"] for row in board}), len(board))
            chain = ledger.ancestry(run["run_id"], self.result.best.id)
            self.assertEqual(chain[0]["candidate_id"], "seed")
            self.assertEqual(chain[-1]["candidate_id"], self.result.best.id)
            patch, best = ledger.winner_patch()
            self.assertEqual(best["source"], self.result.best.source)
            self.assertTrue(patch.startswith("--- a/model/program.py\n+++ b/model/program.py\n"))
            with self.assertRaises(LedgerError):
                ledger.resolve_run("run-missing")

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_patch_applies_with_git(self):
        work = self.dir / "apply"
        (work / "model").mkdir(parents=True)
        (work / "model" / "program.py").write_text(CURVE_SEED)
        with Ledger(self.db, create=False) as ledger:
            patch, best = ledger.winner_patch()
        (work / "best.patch").write_text(patch)
        subprocess.run(["git", "init", "-q"], cwd=work, check=True)
        subprocess.run(["git", "apply", "best.patch"], cwd=work, check=True)
        self.assertEqual((work / "model" / "program.py").read_text(), best["source"])

    def test_missing_or_foreign_ledger(self):
        with self.assertRaises(LedgerError):
            Ledger(self.dir / "nope.db", create=False)
        foreign = self.dir / "foreign.db"
        sqlite3.connect(foreign).execute("CREATE TABLE t (x)").connection.close()
        with self.assertRaises(LedgerError):
            Ledger(foreign, create=False)

    def test_report_is_self_contained(self):
        with Ledger(self.db, create=False) as ledger:
            page = render_report(ledger)
            out = write_report(ledger, self.dir / "out" / "report.html")
        self.assertTrue(out.is_file())
        for section in (
            "Fitness by generation",
            "Lineage",
            "Island leaderboard",
            "Top candidates",
            "Ancestry of the winner",
            "Winning patch",
        ):
            self.assertIn(section, page)
        self.assertEqual(page.count("<svg"), 2)
        self.assertIn("color-scheme: dark", page)
        self.assertIsNone(re.search(r"""(src|href)=["']?(https?:)?//""", page))
        self.assertNotIn("<script", page)
        self.assertIn(self.result.best.id, page)


if __name__ == "__main__":
    unittest.main()
