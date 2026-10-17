# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""Island-model evolutionary search.

The engine owns the population and the loop. Evaluators score candidates; mutators propose new
ones. Every choice that shapes the search draws from seeded random streams (one per island, one
per child), so a run with the same seed, evaluator and deterministic mutator is reproducible even
when candidates are evaluated in parallel.

One generation, per island:

1. Plan children: tournament-select one parent (mutation) or two (crossover).
2. Realize and evaluate the children, in parallel when ``workers > 1``. Identical programs are
   evaluated once (the evaluation cache is keyed by content hash).
3. Select survivors: the island's elites plus the children (``generational``), or the best of
   parents and children together (``plus``).
4. Every ``migration_interval`` generations, each island sends its best candidates to the next
   island in a ring, replacing that island's weakest members.
"""

from __future__ import annotations

import difflib
import hashlib
import math
import random
import secrets
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:  # imported for annotations only, to keep the module graph acyclic
    from .evaluators import EvaluationResult, Evaluator
    from .ledger import Ledger
    from .mutators import Mutator

ENGINE_NAME = "omnievolve-python"
SURVIVOR_MODES = ("generational", "plus")
OBJECTIVES = ("maximize", "minimize")


class EvolutionError(RuntimeError):
    """The run cannot continue (for example, the seed program fails its evaluation)."""


def content_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def unified_diff(before: str, after: str, path: str) -> str:
    """A git-style unified diff of one file, ``a/path`` to ``b/path``."""
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    out = []
    for line in lines:
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out)


@dataclass
class SearchConfig:
    """Search parameters. ``validate()`` rejects impossible combinations."""

    generations: int = 8
    islands: int = 2
    population: int = 6
    elite: int = 1
    tournament_size: int = 3
    crossover_rate: float = 0.25
    migration_interval: int = 3
    migration_size: int = 1
    survivors: str = "generational"
    seed: int = 7
    objective: str = "maximize"
    workers: int = 1
    max_attempts: int = 4
    patience: int | None = None
    target_score: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SearchConfig:
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown search setting(s): {', '.join(unknown)}")
        return cls(**data).validate()

    def validate(self) -> SearchConfig:
        def positive(name: str, minimum: int = 1) -> None:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"search.{name} must be an integer >= {minimum}")

        positive("generations", 0)
        positive("islands")
        positive("population", 2)
        positive("elite", 0)
        positive("tournament_size")
        positive("migration_interval", 0)
        positive("migration_size", 0)
        positive("workers")
        positive("max_attempts")
        if self.elite >= self.population:
            raise ValueError("search.elite must be smaller than search.population")
        if not 0.0 <= float(self.crossover_rate) <= 1.0:
            raise ValueError("search.crossover_rate must be between 0 and 1")
        if self.survivors not in SURVIVOR_MODES:
            raise ValueError(f"search.survivors must be one of {', '.join(SURVIVOR_MODES)}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"search.objective must be one of {', '.join(OBJECTIVES)}")
        if self.patience is not None:
            positive("patience")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    """One version of the target file and its evaluation."""

    id: str
    source: str
    generation: int
    island: int
    parents: tuple[str, ...] = ()
    operator: str = "seed"
    score: float | None = None
    fitness: float = -math.inf
    metrics: dict[str, float] = field(default_factory=dict)
    ok: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    cached: bool = False
    note: str = ""
    parent_sources: tuple[str, ...] = field(default=(), repr=False, compare=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        self.digest = content_hash(self.source)

    def apply(self, result: EvaluationResult, objective: str, cached: bool = False) -> None:
        self.score = result.score
        self.metrics = dict(result.metrics)
        self.ok = bool(result.ok) and result.score is not None and math.isfinite(result.score)
        self.error = result.error
        self.latency_ms = 0.0 if cached else result.latency_ms
        self.cached = cached
        if self.ok:
            self.fitness = result.score if objective == "maximize" else -result.score
        else:
            self.fitness = -math.inf


def rank_key(candidate: Candidate) -> tuple[float, int, str]:
    """Best first; ties go to the older candidate, then the smaller id."""
    return (-candidate.fitness, candidate.generation, candidate.id)


@dataclass
class Island:
    """A sub-population that evolves on its own and exchanges migrants with its neighbour."""

    index: int
    members: list[Candidate] = field(default_factory=list)

    def ranked(self) -> list[Candidate]:
        return sorted(self.members, key=rank_key)

    def best(self) -> Candidate:
        return min(self.members, key=rank_key)

    def tournament(
        self, rng: random.Random, size: int, exclude: Candidate | None = None
    ) -> Candidate:
        pool = [m for m in self.members if exclude is None or m.digest != exclude.digest]
        pool = pool or self.members
        contenders = [pool[rng.randrange(len(pool))] for _ in range(max(1, size))]
        return min(contenders, key=rank_key)

    def mean_score(self) -> float | None:
        scores = [m.score for m in self.members if m.ok and m.score is not None]
        return sum(scores) / len(scores) if scores else None

    def distinct(self) -> int:
        return len({m.digest for m in self.members})


class Population:
    """All islands of one run."""

    def __init__(self, islands: Sequence[Island]):
        self.islands = list(islands)

    def members(self) -> list[Candidate]:
        seen: dict[str, Candidate] = {}
        for island in self.islands:
            for member in island.members:
                seen.setdefault(member.id, member)
        return list(seen.values())

    def best(self) -> Candidate:
        return min(self.members(), key=rank_key)


@dataclass
class RunResult:
    run_id: str
    name: str
    objective: str
    seed: Candidate
    best: Candidate
    history: list[dict[str, Any]]
    evaluations: int
    cache_hits: int
    candidates: int
    duration_s: float
    stopped_reason: str
    ledger_path: str | None = None

    @property
    def improvement(self) -> float:
        if self.seed.score is None or self.best.score is None:
            return 0.0
        return self.best.score - self.seed.score

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "objective": self.objective,
            "seed_score": self.seed.score,
            "best_score": self.best.score,
            "improvement": self.improvement,
            "improved": self.best.fitness > self.seed.fitness,
            "best_candidate_id": self.best.id,
            "best_operator": self.best.operator,
            "generations": len(self.history) - 1,
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "candidates": self.candidates,
            "duration_s": round(self.duration_s, 3),
            "stopped_reason": self.stopped_reason,
            "ledger": self.ledger_path,
        }


@dataclass
class _Plan:
    id: str
    island: int
    generation: int
    parents: tuple[Candidate, ...]
    crossover: bool
    rng_seed: int
    inspirations: tuple[Candidate, ...]


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run-{stamp}-{secrets.token_hex(3)}"


class EvolutionEngine:
    """Runs one evolutionary search over a single target file.

    ``seed_source`` is the starting version of the file and ``target_name`` its file name (the
    candidate is written under that name before evaluation, so the evaluator sees a real file).
    """

    def __init__(
        self,
        *,
        seed_source: str,
        evaluator: Evaluator,
        mutator: Mutator,
        config: SearchConfig | None = None,
        target_name: str = "program.py",
        ledger: Ledger | None = None,
        run_name: str = "omnievolve",
        run_config: dict[str, Any] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        workdir: Path | None = None,
    ):
        self.seed_source = seed_source
        self.evaluator = evaluator
        self.mutator = mutator
        self.config = (config or SearchConfig()).validate()
        self.target_name = Path(target_name).name
        self.target_label = target_name
        self.ledger = ledger
        self.run_name = run_name
        self.run_config = run_config or {}
        self.on_event = on_event
        self.workdir = workdir
        self._cache: dict[str, EvaluationResult] = {}
        self._lock = threading.Lock()
        self.evaluations = 0
        self.cache_hits = 0
        self._recorded = 0

    # -- public -----------------------------------------------------------------------------

    def run(self) -> RunResult:
        cfg = self.config
        run_id = new_run_id()
        started = time.perf_counter()
        holder = None
        if self.workdir is None:
            holder = tempfile.TemporaryDirectory(prefix="omnievolve-")
            scratch = Path(holder.name)
        else:
            scratch = Path(self.workdir)
            scratch.mkdir(parents=True, exist_ok=True)
        if self.ledger is not None:
            self.ledger.start_run(
                run_id=run_id,
                name=self.run_name,
                target=self.target_label,
                objective=cfg.objective,
                config={**self.run_config, "search": cfg.to_dict()},
                engine=ENGINE_NAME,
                engine_version=__version__,
            )
        status, stopped_reason = "failed", "error"
        try:
            result = self._run(run_id, scratch, started)
            status, stopped_reason = "completed", result.stopped_reason
            return result
        finally:
            if self.ledger is not None:
                best = getattr(self, "_best_ever", None)
                self.ledger.finish_run(
                    run_id,
                    status=status,
                    stopped_reason=stopped_reason,
                    best_candidate_id=best.id if best else None,
                    best_score=best.score if best else None,
                    evaluations=self.evaluations,
                    cache_hits=self.cache_hits,
                    duration_s=time.perf_counter() - started,
                )
            if holder is not None:
                holder.cleanup()

    # -- loop -------------------------------------------------------------------------------

    def _run(self, run_id: str, scratch: Path, started: float) -> RunResult:
        cfg = self.config
        seed = Candidate(id="seed", source=self.seed_source, generation=0, island=-1)
        self._evaluate(seed, scratch)
        self._record(run_id, [seed])
        if not seed.ok:
            raise EvolutionError(f"the seed program failed its evaluation: {seed.error}")
        self._best_ever = seed
        self._emit({"type": "seed", "run_id": run_id, "score": seed.score, "metrics": seed.metrics})

        islands = [Island(i, [seed]) for i in range(cfg.islands)]
        rngs = [random.Random(f"{cfg.seed}:island:{i}") for i in range(cfg.islands)]

        # Generation 0: each island starts from the seed plus its own mutants of it.
        plans = []
        for island in islands:
            for n in range(cfg.population - 1):
                plans.append(self._plan(island, rngs[island.index], 0, n, allow_crossover=False))
        children = self._execute(plans, scratch)
        self._record(run_id, children)
        for island in islands:
            island.members.extend(c for c in children if c.island == island.index)
        history = [self._snapshot(run_id, 0, islands, children)]

        stopped = "completed"
        since_improvement = 0
        for generation in range(1, cfg.generations + 1):
            before = self._best_ever.fitness
            plans = []
            for island in islands:
                keep = min(cfg.elite, len(island.members)) if cfg.survivors == "generational" else 0
                for n in range(cfg.population - keep):
                    plans.append(self._plan(island, rngs[island.index], generation, n))
            children = self._execute(plans, scratch)
            self._record(run_id, children)
            for island in islands:
                self._select(island, [c for c in children if c.island == island.index])
            migrants: list[Candidate] = []
            if (
                cfg.islands > 1
                and cfg.migration_interval
                and cfg.migration_size
                and generation % cfg.migration_interval == 0
            ):
                migrants = self._migrate(islands, generation)
                self._record(run_id, migrants)
            history.append(self._snapshot(run_id, generation, islands, children, migrants))

            if self._best_ever.fitness > before:
                since_improvement = 0
            else:
                since_improvement += 1
            if cfg.target_score is not None and self._reached_target():
                stopped = "target_score"
                break
            if cfg.patience is not None and since_improvement >= cfg.patience:
                stopped = "patience"
                break

        if self.ledger is not None:
            self.ledger.record_islands(
                run_id,
                [
                    {
                        "island": isl.index,
                        "size": len(isl.members),
                        "best_candidate_id": isl.best().id,
                        "best_score": isl.best().score,
                        "mean_score": isl.mean_score(),
                    }
                    for isl in islands
                ],
            )
        result = RunResult(
            run_id=run_id,
            name=self.run_name,
            objective=cfg.objective,
            seed=seed,
            best=self._best_ever,
            history=history,
            evaluations=self.evaluations,
            cache_hits=self.cache_hits,
            candidates=self._recorded,
            duration_s=time.perf_counter() - started,
            stopped_reason=stopped,
            ledger_path=str(self.ledger.path) if self.ledger is not None else None,
        )
        self._emit({"type": "done", **result.summary()})
        return result

    def _reached_target(self) -> bool:
        score = self._best_ever.score
        target = self.config.target_score
        if score is None or target is None:
            return False
        return score >= target if self.config.objective == "maximize" else score <= target

    # -- planning and realization -----------------------------------------------------------

    def _plan(
        self,
        island: Island,
        rng: random.Random,
        generation: int,
        n: int,
        allow_crossover: bool = True,
    ) -> _Plan:
        cfg = self.config
        crossover = allow_crossover and island.distinct() >= 2 and rng.random() < cfg.crossover_rate
        first = island.tournament(rng, cfg.tournament_size)
        parents: tuple[Candidate, ...] = (first,)
        if crossover:
            parents = (first, island.tournament(rng, cfg.tournament_size, exclude=first))
        ranked = [m for m in island.ranked() if m.digest != first.digest and m.ok]
        inspirations = tuple(ranked[:3])
        return _Plan(
            id=f"g{generation:03d}-i{island.index}-{n:02d}",
            island=island.index,
            generation=generation,
            parents=parents,
            crossover=crossover,
            rng_seed=rng.getrandbits(64),
            inspirations=inspirations,
        )

    def _execute(self, plans: list[_Plan], scratch: Path) -> list[Candidate]:
        with self._lock:
            known = frozenset(self._cache)
        if self.config.workers <= 1 or len(plans) <= 1:
            return [self._realize(plan, scratch, known) for plan in plans]
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            return list(pool.map(lambda plan: self._realize(plan, scratch, known), plans))

    def _realize(self, plan: _Plan, scratch: Path, known: frozenset[str]) -> Candidate:
        from .mutators import MutationContext

        rng = random.Random(plan.rng_seed)
        primary = plan.parents[0]
        ctx = MutationContext(
            target_name=self.target_name,
            objective=self.config.objective,
            generation=plan.generation,
            island=plan.island,
            inspirations=plan.inspirations,
        )
        mutation = None
        used: tuple[Candidate, ...] = (primary,)
        for attempt in range(self.config.max_attempts):
            used = (primary,)
            if plan.crossover and attempt == 0:
                mutation = self.mutator.crossover(primary, plan.parents[1], rng, ctx)
                used = plan.parents
                if mutation.source in (p.source for p in plan.parents):
                    mutation = self.mutator.mutate(primary, rng, ctx)
                    used = (primary,)
            else:
                mutation = self.mutator.mutate(primary, rng, ctx)
            if content_hash(mutation.source) not in known:
                break
        assert mutation is not None
        child = Candidate(
            id=plan.id,
            source=mutation.source,
            generation=plan.generation,
            island=plan.island,
            parents=tuple(p.id for p in used),
            operator=mutation.operator,
            note=mutation.note,
            parent_sources=tuple(p.source for p in used),
        )
        self._evaluate(child, scratch)
        return child

    def _evaluate(self, candidate: Candidate, scratch: Path) -> None:
        with self._lock:
            cached = self._cache.get(candidate.digest)
        if cached is not None:
            candidate.apply(cached, self.config.objective, cached=True)
            with self._lock:
                self.cache_hits += 1
        else:
            folder = scratch / candidate.id
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / self.target_name
            path.write_text(candidate.source, encoding="utf-8")
            try:
                result = self.evaluator.evaluate(path, candidate.source)
            finally:
                shutil.rmtree(folder, ignore_errors=True)
            candidate.apply(result, self.config.objective)
            with self._lock:
                self._cache.setdefault(candidate.digest, result)
                self.evaluations += 1
        best = getattr(self, "_best_ever", None)
        if candidate.ok and (best is None or rank_key(candidate) < rank_key(best)):
            with self._lock:
                best = getattr(self, "_best_ever", None)
                if best is None or rank_key(candidate) < rank_key(best):
                    self._best_ever = candidate

    # -- selection and migration ------------------------------------------------------------

    def _select(self, island: Island, children: list[Candidate]) -> None:
        cfg = self.config
        if cfg.survivors == "plus":
            unique: dict[str, Candidate] = {}
            for member in island.members + children:
                unique.setdefault(member.digest, member)
            island.members = sorted(unique.values(), key=rank_key)[: cfg.population]
        else:
            island.members = island.ranked()[: cfg.elite] + children

    def _migrate(self, islands: list[Island], generation: int) -> list[Candidate]:
        cfg = self.config
        outgoing = [isl.ranked()[: cfg.migration_size] for isl in islands]
        migrants = []
        for i, sources in enumerate(outgoing):
            dest = islands[(i + 1) % len(islands)]
            for k, src in enumerate(sources):
                if not src.ok or any(m.digest == src.digest for m in dest.members):
                    continue
                worst = dest.ranked()[-1]
                if len(dest.members) > 1 and rank_key(worst) <= rank_key(src):
                    continue  # the destination's weakest member is already as good
                copy = Candidate(
                    id=f"g{generation:03d}-i{dest.index}-m{k}",
                    source=src.source,
                    generation=generation,
                    island=dest.index,
                    parents=(src.id,),
                    operator="migrate",
                    parent_sources=(src.source,),
                )
                copy.score, copy.fitness, copy.ok = src.score, src.fitness, src.ok
                copy.metrics = dict(src.metrics)
                copy.cached = True
                dest.members.remove(worst)
                dest.members.append(copy)
                migrants.append(copy)
        return migrants

    # -- bookkeeping ------------------------------------------------------------------------

    def _snapshot(
        self,
        run_id: str,
        generation: int,
        islands: list[Island],
        children: list[Candidate],
        migrants: Sequence[Candidate] = (),
    ) -> dict[str, Any]:
        rows = []
        for island in islands:
            best = island.best()
            mine = [c for c in children if c.island == island.index]
            row = {
                "island": island.index,
                "best_score": best.score,
                "mean_score": island.mean_score(),
                "best_candidate_id": best.id,
                "evaluated": sum(1 for c in mine if not c.cached),
                "cache_hits": sum(1 for c in mine if c.cached),
                "failed": sum(1 for c in mine if not c.ok),
                "migrants_in": sum(1 for m in migrants if m.island == island.index),
            }
            rows.append(row)
            if self.ledger is not None:
                self.ledger.record_generation(run_id, generation, row)
        entry = {
            "type": "generation",
            "generation": generation,
            "best_score": self._best_ever.score,
            "best_candidate_id": self._best_ever.id,
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "islands": rows,
        }
        self._emit(entry)
        return entry

    def _record(self, run_id: str, candidates: Sequence[Candidate]) -> None:
        self._recorded += len(candidates)
        if self.ledger is None:
            return
        for candidate in candidates:
            edges = []
            pairs = zip(candidate.parents, candidate.parent_sources, strict=True)
            for index, (parent_id, parent_source) in enumerate(pairs):
                role = (
                    "migrant"
                    if candidate.operator == "migrate"
                    else ("primary" if index == 0 else "secondary")
                )
                diff = unified_diff(parent_source, candidate.source, self.target_label)
                edges.append((parent_id, role, diff))
            self.ledger.record_candidate(run_id, candidate, edges)
        self.ledger.commit()

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)
