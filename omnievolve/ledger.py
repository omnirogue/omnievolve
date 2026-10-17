# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""The local SQLite lineage ledger.

One file (by default ``.omnievolve/ledger.db``) holds every run: its configuration, each
generation's per-island statistics, every candidate's full source, score, metrics and latency,
and the lineage DAG as parent-to-child edges, each carrying the unified diff between the two.
Plain SQL, so ``sqlite3 .omnievolve/ledger.db`` works for ad hoc questions.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .engine import unified_diff

if TYPE_CHECKING:
    from .engine import Candidate

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    target TEXT NOT NULL,
    objective TEXT NOT NULL,
    engine TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    stopped_reason TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_s REAL,
    evaluations INTEGER,
    cache_hits INTEGER,
    best_candidate_id TEXT,
    best_score REAL
);
CREATE TABLE IF NOT EXISTS candidates (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    candidate_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    island INTEGER NOT NULL,
    operator TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source TEXT NOT NULL,
    score REAL,
    fitness REAL,
    ok INTEGER NOT NULL,
    error TEXT,
    metrics_json TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    cached INTEGER NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, candidate_id)
);
CREATE TABLE IF NOT EXISTS lineage (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    role TEXT NOT NULL,
    diff TEXT NOT NULL,
    PRIMARY KEY (run_id, parent_id, child_id)
);
CREATE TABLE IF NOT EXISTS generations (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    generation INTEGER NOT NULL,
    island INTEGER NOT NULL,
    best_score REAL,
    mean_score REAL,
    best_candidate_id TEXT,
    evaluated INTEGER NOT NULL,
    cache_hits INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    migrants_in INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (run_id, generation, island)
);
CREATE TABLE IF NOT EXISTS islands (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    island INTEGER NOT NULL,
    size INTEGER NOT NULL,
    best_candidate_id TEXT,
    best_score REAL,
    mean_score REAL,
    PRIMARY KEY (run_id, island)
);
CREATE INDEX IF NOT EXISTS candidates_by_fitness ON candidates (run_id, fitness DESC);
CREATE INDEX IF NOT EXISTS lineage_by_child ON lineage (run_id, child_id);
"""


class LedgerError(RuntimeError):
    """The ledger is missing, empty or does not contain the requested run."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


class Ledger:
    """Read and write access to one ledger file. Use as a context manager or call ``close()``."""

    def __init__(self, path: Path | str, create: bool = True):
        self.path = Path(path)
        if not create and not self.path.is_file():
            raise LedgerError(f"no ledger at {self.path} (run `omnievolve run` first)")
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if create:
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.conn.commit()
        else:
            tables = {
                row[0]
                for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "runs" not in tables:
                self.conn.close()
                raise LedgerError(f"{self.path} is not an OmniEvolve ledger")

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    # -- writes -----------------------------------------------------------------------------

    def start_run(
        self,
        *,
        run_id: str,
        name: str,
        target: str,
        objective: str,
        config: dict[str, Any],
        engine: str,
        engine_version: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO runs (run_id, name, target, objective, engine, engine_version,"
            " config_json, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)",
            (
                run_id,
                name,
                target,
                objective,
                engine,
                engine_version,
                json.dumps(config, sort_keys=True, default=str),
                _now(),
            ),
        )
        self.conn.commit()

    def record_candidate(
        self, run_id: str, candidate: Candidate, edges: Sequence[tuple[str, str, str]] = ()
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO candidates (run_id, candidate_id, generation, island, operator,"
            " content_hash, source, score, fitness, ok, error, metrics_json, latency_ms, cached,"
            " note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                candidate.id,
                candidate.generation,
                candidate.island,
                candidate.operator,
                candidate.digest,
                candidate.source,
                _finite(candidate.score),
                _finite(candidate.fitness),
                int(candidate.ok),
                candidate.error,
                json.dumps(candidate.metrics, sort_keys=True),
                round(candidate.latency_ms, 3),
                int(candidate.cached),
                candidate.note or None,
                _now(),
            ),
        )
        for parent_id, role, diff in edges:
            self.conn.execute(
                "INSERT OR REPLACE INTO lineage (run_id, parent_id, child_id, role, diff)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, parent_id, candidate.id, role, diff),
            )

    def record_generation(self, run_id: str, generation: int, row: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO generations (run_id, generation, island, best_score,"
            " mean_score, best_candidate_id, evaluated, cache_hits, failed, migrants_in,"
            " recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                generation,
                row["island"],
                _finite(row.get("best_score")),
                _finite(row.get("mean_score")),
                row.get("best_candidate_id"),
                row.get("evaluated", 0),
                row.get("cache_hits", 0),
                row.get("failed", 0),
                row.get("migrants_in", 0),
                _now(),
            ),
        )
        self.conn.commit()

    def record_islands(self, run_id: str, rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            self.conn.execute(
                "INSERT OR REPLACE INTO islands (run_id, island, size, best_candidate_id,"
                " best_score, mean_score) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    row["island"],
                    row["size"],
                    row.get("best_candidate_id"),
                    _finite(row.get("best_score")),
                    _finite(row.get("mean_score")),
                ),
            )
        self.conn.commit()

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        stopped_reason: str,
        best_candidate_id: str | None,
        best_score: float | None,
        evaluations: int,
        cache_hits: int,
        duration_s: float,
    ) -> None:
        self.conn.execute(
            "UPDATE runs SET status = ?, stopped_reason = ?, finished_at = ?, duration_s = ?,"
            " evaluations = ?, cache_hits = ?, best_candidate_id = ?, best_score = ?"
            " WHERE run_id = ?",
            (
                status,
                stopped_reason,
                _now(),
                round(duration_s, 3),
                evaluations,
                cache_hits,
                best_candidate_id,
                _finite(best_score),
                run_id,
            ),
        )
        self.conn.commit()

    # -- reads ------------------------------------------------------------------------------

    def runs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT r.*, (SELECT COUNT(*) FROM candidates c WHERE c.run_id = r.run_id)"
            " AS candidates FROM runs r ORDER BY r.started_at DESC, r.rowid DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def resolve_run(self, run_id: str | None = None) -> str:
        """The given run id, or the most recent completed run (else the most recent run)."""
        if run_id:
            if self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone():
                return run_id
            raise LedgerError(f"no run {run_id!r} in {self.path}")
        row = self.conn.execute(
            "SELECT run_id FROM runs ORDER BY (status = 'completed') DESC, started_at DESC,"
            " rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise LedgerError(f"{self.path} has no runs yet")
        return row["run_id"]

    def run(self, run_id: str | None = None) -> dict[str, Any]:
        run_id = self.resolve_run(run_id)
        row = dict(self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone())
        row["config"] = json.loads(row.pop("config_json"))
        return row

    def _candidate(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["metrics"] = json.loads(out.pop("metrics_json"))
        out["ok"] = bool(out["ok"])
        out["cached"] = bool(out["cached"])
        return out

    def candidates(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM candidates WHERE run_id = ? ORDER BY generation, island, candidate_id",
            (run_id,),
        ).fetchall()
        return [self._candidate(row) for row in rows]

    def candidate(self, run_id: str, candidate_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM candidates WHERE run_id = ? AND candidate_id = ?",
            (run_id, candidate_id),
        ).fetchone()
        if row is None:
            raise LedgerError(f"no candidate {candidate_id!r} in run {run_id!r}")
        return self._candidate(row)

    def leaderboard(self, run_id: str, limit: int = 10, distinct: bool = True) -> list[dict]:
        """Best candidates first. ``distinct`` keeps one row per program (the earliest).

        Migrants are copies of candidates already listed, so they never appear; the order is
        the engine's own ranking, so the first row is the run's winner.
        """
        rows = self.conn.execute(
            "SELECT * FROM candidates WHERE run_id = ? AND ok = 1 AND operator != 'migrate'"
            " ORDER BY fitness DESC, generation ASC, candidate_id ASC",
            (run_id,),
        ).fetchall()
        out, seen = [], set()
        for row in rows:
            if distinct and row["content_hash"] in seen:
                continue
            seen.add(row["content_hash"])
            out.append(self._candidate(row))
            if len(out) >= limit:
                break
        return out

    def edges(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT parent_id, child_id, role FROM lineage WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def edge_diff(self, run_id: str, parent_id: str, child_id: str) -> str:
        row = self.conn.execute(
            "SELECT diff FROM lineage WHERE run_id = ? AND parent_id = ? AND child_id = ?",
            (run_id, parent_id, child_id),
        ).fetchone()
        return row["diff"] if row else ""

    def ancestry(self, run_id: str, candidate_id: str) -> list[dict[str, Any]]:
        """The chain from the seed to ``candidate_id`` along primary (and migrant) parents."""
        chain, current, seen = [], candidate_id, set()
        while current and current not in seen:
            seen.add(current)
            chain.append(self.candidate(run_id, current))
            row = self.conn.execute(
                "SELECT parent_id FROM lineage WHERE run_id = ? AND child_id = ?"
                " AND role IN ('primary', 'migrant')",
                (run_id, current),
            ).fetchone()
            current = row["parent_id"] if row else None
        chain.reverse()
        return chain

    def generations(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM generations WHERE run_id = ? ORDER BY generation, island", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def islands(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM islands WHERE run_id = ? ORDER BY island", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def best(self, run_id: str) -> dict[str, Any]:
        board = self.leaderboard(run_id, limit=1)
        if not board:
            raise LedgerError(f"run {run_id!r} has no successfully evaluated candidate")
        return board[0]

    def winner_patch(self, run_id: str | None = None) -> tuple[str, dict[str, Any]]:
        """A unified diff from the seed to the best candidate, and the best candidate."""
        run = self.run(run_id)
        best = self.best(run["run_id"])
        seed = self.candidate(run["run_id"], "seed")
        return unified_diff(seed["source"], best["source"], run["target"]), best
