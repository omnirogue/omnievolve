# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
import os
import sys
import tempfile
import unittest
from pathlib import Path

from _support import ROOT

from omnievolve.evaluators import (
    CompositeEvaluator,
    EvaluationResult,
    LLMJudgeEvaluator,
    PythonCallableEvaluator,
    SubprocessEvaluator,
    build_evaluator,
    coerce_result,
    parse_score_output,
)
from omnievolve.llm import LLMClient, LLMConfig


class ParseTests(unittest.TestCase):
    def test_score_formats(self):
        cases = {
            '{"score": 0.5, "latency": 3}': (0.5, {"latency": 3.0}),
            'progress...\n{"score": 2, "ok": true}\n': (2.0, {}),
            "warming up\n0.75\n": (0.75, {}),
            "final score: 1.5e-1 (best)": (0.15, {}),
            "42": (42.0, {}),
        }
        for text, (score, metrics) in cases.items():
            with self.subTest(text=text):
                result = parse_score_output(text)
                self.assertTrue(result.ok, result.error)
                self.assertEqual(result.score, score)
                self.assertEqual(result.metrics, metrics)

    def test_unreadable_output_fails(self):
        for text in ("", "no numbers here", '{"accuracy": 0.9}', '{"score": "high"}'):
            with self.subTest(text=text):
                self.assertFalse(parse_score_output(text).ok)
        self.assertEqual(parse_score_output('{"error": "bad input"}').error, "bad input")

    def test_coerce(self):
        self.assertEqual(coerce_result(3).score, 3.0)
        self.assertFalse(coerce_result(True).ok)
        self.assertFalse(coerce_result(float("inf")).ok)
        same = EvaluationResult(score=1.0)
        self.assertIs(coerce_result(same), same)


class SubprocessTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="omnievolve-test-"))
        self.candidate = self.dir / "cand.py"
        self.candidate.write_text("VALUE = 3\n")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def script(self, body: str) -> Path:
        path = self.dir / "bench.py"
        path.write_text(body)
        return path

    def test_json_score_and_placeholders(self):
        bench = self.script(
            "import json, os, sys\n"
            "ns = {}\n"
            "exec(open(sys.argv[1]).read(), ns)\n"
            "assert os.environ['OMNIEVOLVE_CANDIDATE'] == sys.argv[1]\n"
            "cwd_ok = int(os.getcwd() == sys.argv[2])\n"
            "print(json.dumps({'score': ns['VALUE'] * 2, 'cwd_ok': cwd_ok}))\n"
        )
        evaluator = SubprocessEvaluator(
            ["{python}", str(bench), "{candidate}", "{project}"], cwd=self.dir
        )
        result = evaluator.evaluate(self.candidate, "VALUE = 3\n")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.score, 6.0)
        self.assertEqual(result.metrics["cwd_ok"], 1.0)
        self.assertGreater(result.latency_ms, 0)

    def test_string_command_and_nonzero_exit(self):
        bench = self.script("import sys\nprint('bad things', file=sys.stderr)\nsys.exit(3)\n")
        evaluator = SubprocessEvaluator(f'"{sys.executable}" "{bench}"', cwd=self.dir)
        result = evaluator.evaluate(self.candidate, "")
        self.assertFalse(result.ok)
        self.assertIn("exit status 3", result.error)
        self.assertIn("bad things", result.error)

    @unittest.skipUnless(os.name == "posix", "process groups are POSIX")
    def test_timeout_kills_the_process(self):
        bench = self.script("import time\ntime.sleep(30)\n")
        evaluator = SubprocessEvaluator([sys.executable, str(bench)], timeout_seconds=0.3)
        result = evaluator.evaluate(self.candidate, "")
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.error)
        self.assertLess(result.latency_ms, 5000)

    def test_missing_program(self):
        evaluator = SubprocessEvaluator(["omnievolve-no-such-program-xyz"])
        self.assertIn("cannot start", evaluator.evaluate(self.candidate, "").error)

    def test_factory_for_the_example(self):
        example = ROOT / "examples" / "heuristic_tuning"
        evaluator = build_evaluator(
            {"type": "subprocess", "command": ["{python}", "evaluate.py", "{candidate}"]}, example
        )
        result = evaluator.evaluate(example / "triage.py", "")
        self.assertTrue(result.ok, result.error)
        self.assertAlmostEqual(result.score, 0.553571, places=5)
        self.assertEqual(set(result.metrics), {"precision", "recall", "accuracy"})
        with self.assertRaisesRegex(ValueError, "unknown evaluator type"):
            build_evaluator({"type": "nope"}, example)


class CallableAndCompositeTests(unittest.TestCase):
    def test_callable_arity_and_errors(self):
        one = PythonCallableEvaluator(lambda path: 1.5)
        two = PythonCallableEvaluator(lambda path, source: {"score": len(source)})
        boom = PythonCallableEvaluator(lambda path: 1 / 0)
        self.assertEqual(one.evaluate(Path("x"), "abc").score, 1.5)
        self.assertEqual(two.evaluate(Path("x"), "abc").score, 3.0)
        failed = boom.evaluate(Path("x"), "")
        self.assertFalse(failed.ok)
        self.assertIn("ZeroDivisionError", failed.error)

    def test_composite(self):
        parts = [
            ("quality", PythonCallableEvaluator(lambda p: {"score": 1.0, "f1": 0.9}), 3),
            ("speed", PythonCallableEvaluator(lambda p: 0.0), 1),
        ]
        result = CompositeEvaluator(parts).evaluate(Path("x"), "")
        self.assertAlmostEqual(result.score, 0.75)
        self.assertEqual(result.metrics["quality.f1"], 0.9)
        broken = CompositeEvaluator(parts + [("bad", PythonCallableEvaluator(lambda p: 1 / 0), 1)])
        self.assertFalse(broken.evaluate(Path("x"), "").ok)
        lenient = CompositeEvaluator(broken.parts, required=False)
        self.assertAlmostEqual(lenient.evaluate(Path("x"), "").score, 0.75)


class JudgeTests(unittest.TestCase):
    def test_judge_averages_samples(self):
        replies = iter(['{"score": 8, "rationale": "good"}', "Score: 6"])

        def transport(url, headers, payload, timeout):
            return {"message": {"content": next(replies)}}

        client = LLMClient(LLMConfig(provider="ollama", model="local"), transport=transport)
        judge = LLMJudgeEvaluator(client, rubric="Is it clear?", scale=10, samples=2)
        result = judge.evaluate(Path("prompt.txt"), "Be clear.")
        self.assertTrue(result.ok, result.error)
        self.assertAlmostEqual(result.score, 0.7)
        self.assertAlmostEqual(result.metrics["judge_spread"], 1.0)
        self.assertIsNone(LLMJudgeEvaluator.parse_grade("I refuse"))


if __name__ == "__main__":
    unittest.main()
