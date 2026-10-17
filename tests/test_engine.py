# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
import math
import random
import unittest

from _support import CURVE_SEED, curve_evaluator

from omnievolve.engine import (
    Candidate,
    EvolutionEngine,
    EvolutionError,
    Island,
    Population,
    SearchConfig,
    unified_diff,
)
from omnievolve.evaluators import EvaluationResult, Evaluator
from omnievolve.mutators import DeterministicMutator


def engine(**search):
    config = SearchConfig(**{"generations": 6, "islands": 2, "population": 6, "seed": 3, **search})
    return EvolutionEngine(
        seed_source=CURVE_SEED,
        evaluator=curve_evaluator(),
        mutator=DeterministicMutator(sigma=0.4),
        config=config,
    )


class EngineTests(unittest.TestCase):
    def test_search_improves_on_the_seed(self):
        result = engine().run()
        self.assertTrue(result.seed.ok)
        self.assertGreater(result.best.score, result.seed.score)
        self.assertGreater(result.improvement, 0)
        self.assertEqual(len(result.history), 7)  # generation 0 plus six
        best_curve = [entry["best_score"] for entry in result.history]
        self.assertEqual(best_curve, sorted(best_curve))  # elitism: best so far never drops
        self.assertIn("EVOLVE-BLOCK-START", result.best.source)
        self.assertIn("# quadratic term", result.best.source)  # comments survive mutation

    def test_same_seed_same_result_even_in_parallel(self):
        first = engine().run()
        again = engine().run()
        parallel = engine(workers=4).run()
        self.assertEqual(first.best.source, again.best.source)
        self.assertEqual(first.best.source, parallel.best.source)
        self.assertEqual(first.best.score, parallel.best.score)
        other = engine(seed=99).run()
        self.assertNotEqual(first.best.source, other.best.source)

    def test_minimize_objective(self):
        class Mse(Evaluator):
            def _evaluate(self, path, source):
                return {"score": curve_evaluator().evaluate(path, source).metrics["mse"]}

        result = EvolutionEngine(
            seed_source=CURVE_SEED,
            evaluator=Mse(),
            mutator=DeterministicMutator(sigma=0.4),
            config=SearchConfig(generations=5, objective="minimize", seed=5),
        ).run()
        self.assertLess(result.best.score, result.seed.score)
        self.assertTrue(result.summary()["improved"])

    def test_plus_survivors_and_early_stops(self):
        result = engine(survivors="plus", generations=30, patience=3).run()
        self.assertEqual(result.stopped_reason, "patience")
        self.assertLess(len(result.history), 31)
        target = engine(generations=40, target_score=0.2).run()
        self.assertEqual(target.stopped_reason, "target_score")
        self.assertGreaterEqual(target.best.score, 0.2)

    def test_failing_seed_stops_the_run(self):
        broken = EvolutionEngine(
            seed_source="def predict(x):\n    raise RuntimeError('boom')\n",
            evaluator=curve_evaluator(),
            mutator=DeterministicMutator(),
            config=SearchConfig(generations=1),
        )
        with self.assertRaisesRegex(EvolutionError, "boom"):
            broken.run()

    def test_failed_candidates_do_not_stop_the_run(self):
        class Flaky(Evaluator):
            def _evaluate(self, path, source):
                if "B = -" in source:
                    raise ValueError("negative B rejected")
                return curve_evaluator().evaluate(path, source)

        result = EvolutionEngine(
            seed_source=CURVE_SEED,
            evaluator=Flaky(),
            mutator=DeterministicMutator(sigma=0.5),
            config=SearchConfig(generations=4, seed=2),
        ).run()
        self.assertTrue(result.best.ok)
        self.assertNotIn("B = -", result.best.source)

    def test_events_are_emitted(self):
        events = []
        run = engine(generations=2)
        run.on_event = events.append
        run.run()
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds, ["seed", "generation", "generation", "generation", "done"])

    def test_migration_moves_the_best_between_islands(self):
        result = engine(generations=6, migration_interval=2, islands=3).run()
        self.assertTrue(all(len(entry["islands"]) == 3 for entry in result.history))
        migrated = sum(row["migrants_in"] for entry in result.history for row in entry["islands"])
        self.assertGreater(migrated, 0)


class StructureTests(unittest.TestCase):
    def test_config_validation(self):
        for bad in (
            {"population": 1},
            {"elite": 6, "population": 6},
            {"crossover_rate": 1.5},
            {"survivors": "steady"},
            {"objective": "max"},
            {"generations": -1},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SearchConfig(**bad).validate()
        with self.assertRaisesRegex(ValueError, "unknown"):
            SearchConfig.from_dict({"generationz": 3})

    def test_island_tournament_and_population_best(self):
        members = []
        for i, fitness in enumerate((0.1, 0.9, 0.5)):
            c = Candidate(id=f"c{i}", source=f"x = {i}\n", generation=0, island=0)
            c.apply(EvaluationResult(score=fitness), "maximize")
            members.append(c)
        island = Island(0, members)
        self.assertEqual(island.best().id, "c1")
        self.assertEqual(island.tournament(random.Random(1), size=50).id, "c1")
        self.assertEqual(Population([island, Island(1, members[:1])]).best().id, "c1")
        failed = Candidate(id="f", source="", generation=0, island=0)
        failed.apply(EvaluationResult.failure("nope"), "maximize")
        self.assertEqual(failed.fitness, -math.inf)

    def test_unified_diff_headers(self):
        diff = unified_diff("a = 1\n", "a = 2\n", "pkg/mod.py")
        self.assertTrue(diff.startswith("--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"))
        self.assertIn("-a = 1\n+a = 2\n", diff)
        self.assertIn("No newline at end of file", unified_diff("a", "b", "f"))


if __name__ == "__main__":
    unittest.main()
