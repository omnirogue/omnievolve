# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
import ast
import random
import unittest

from _support import CURVE_SEED

from omnievolve.engine import Candidate
from omnievolve.llm import LLMClient, LLMConfig
from omnievolve.mutators import (
    DeterministicMutator,
    LLMMutator,
    MutationContext,
    build_mutator,
    extract_source,
    format_float,
    python_numeric_slots,
    text_numeric_slots,
)

PY = MutationContext(target_name="program.py")
TXT = MutationContext(target_name="prompt.txt")


def cand(source, cid="p"):
    return Candidate(id=cid, source=source, generation=0, island=0)


class SlotTests(unittest.TestCase):
    def test_evolve_block_limits_the_slots(self):
        source = "K = 3\n# EVOLVE-BLOCK-START\nA = 1.5\nB = -2\n# EVOLVE-BLOCK-END\nZ = 9\n"
        slots = python_numeric_slots(source)
        self.assertEqual([(s.value, s.is_int) for s in slots], [(1.5, False), (-2.0, True)])
        self.assertEqual(source[slots[1].start : slots[1].end], "-2")

    def test_module_level_assignments_come_first(self):
        source = "WEIGHT = 0.5\nLIMIT: int = 10\n\ndef f(x):\n    return x[3] * 7\n"
        values = [s.value for s in python_numeric_slots(source)]
        self.assertEqual(values, [0.5, 10.0])
        self.assertEqual(len(python_numeric_slots("def f(x):\n    return x * 7\n")), 1)

    def test_non_ascii_lines_map_to_the_right_characters(self):
        source = 'LABEL = "über"; W = 2.5\n'
        (slot,) = [s for s in python_numeric_slots(source)]
        self.assertEqual(source[slot.start : slot.end], "2.5")

    def test_text_slots_and_float_format(self):
        slots = text_numeric_slots('{"threshold": 0.75, "max_items": 12, "name": "v2"}')
        self.assertEqual([s.value for s in slots], [0.75, 12.0])
        self.assertEqual(format_float(2.0), "2.0")
        self.assertEqual(format_float(float("nan")), "0.0")


class DeterministicMutatorTests(unittest.TestCase):
    def test_python_mutation_changes_numbers_only(self):
        mutator = DeterministicMutator(sigma=0.5, max_changes=3)
        rng = random.Random(0)
        for _ in range(25):
            mutation = mutator.mutate(cand(CURVE_SEED), rng, PY)
            ast.parse(mutation.source)
            self.assertNotEqual(mutation.source, CURVE_SEED)
            self.assertTrue(mutation.operator.startswith("mutate:params"))
            strip = lambda s: [ln.split("=")[0] for ln in s.splitlines()]  # noqa: E731
            self.assertEqual(strip(mutation.source), strip(CURVE_SEED))
            self.assertIn("# quadratic term", mutation.source)

    def test_non_negative_ints_stay_non_negative(self):
        mutator = DeterministicMutator(sigma=1.0)
        rng = random.Random(4)
        for _ in range(30):
            source = mutator.mutate(cand("SIZE = 1\n"), rng, PY).source
            self.assertGreaterEqual(int(source.split("=")[1]), 0)

    def test_same_rng_same_mutation(self):
        mutator = DeterministicMutator()
        a = mutator.mutate(cand(CURVE_SEED), random.Random(42), PY).source
        b = mutator.mutate(cand(CURVE_SEED), random.Random(42), PY).source
        self.assertEqual(a, b)

    def test_line_pool_operations(self):
        pool = ["Be concise.", "Cite sources.", "Ask when unsure."]
        mutator = DeterministicMutator(line_pool=pool, numeric_rate=0.0)
        rng = random.Random(1)
        seen = set()
        source = "You are helpful.\nAnswer the question.\n"
        for _ in range(40):
            mutation = mutator.mutate(cand(source), rng, TXT)
            seen.add(mutation.operator)
            self.assertTrue(mutation.source.endswith("\n"))
        self.assertTrue({"mutate:add-line", "mutate:remove-line"} <= seen)

    def test_parameter_crossover(self):
        a = cand("A = 1.0\nB = 2.0\nC = 3.0\n", "a")
        b = cand("A = 9.0\nB = 8.0\nC = 7.0\n", "b")
        mutation = DeterministicMutator().crossover(a, b, random.Random(3), PY)
        self.assertEqual(mutation.operator, "crossover:params")
        values = [line.split(" = ")[1] for line in mutation.source.splitlines()]
        expected = (("1.0", "9.0"), ("2.0", "8.0"), ("3.0", "7.0"))
        for value, options in zip(values, expected, strict=True):
            self.assertIn(value, options)

    def test_line_crossover(self):
        a = cand("one\ntwo\nthree\n", "a")
        b = cand("four\nfive\nsix\n", "b")
        mutation = DeterministicMutator().crossover(a, b, random.Random(2), TXT)
        self.assertEqual(mutation.operator, "crossover:lines")
        self.assertTrue(mutation.source.startswith("one\n"))

    def test_factory(self):
        from pathlib import Path

        self.assertIsInstance(build_mutator(None, Path(".")), DeterministicMutator)
        with self.assertRaisesRegex(ValueError, "unknown mutator setting"):
            build_mutator({"type": "deterministic", "sigmaa": 1}, Path("."))
        with self.assertRaisesRegex(ValueError, "unknown mutator type"):
            build_mutator({"type": "magic"}, Path("."))


def fake_client(replies):
    replies = list(replies)
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"content": [{"type": "text", "text": reply}]}

    client = LLMClient(
        LLMConfig(provider="anthropic", model="test-model", api_key_env="OMNIEVOLVE_TEST_KEY"),
        transport=transport,
    )
    return client, calls


class LLMMutatorTests(unittest.TestCase):
    def setUp(self):
        import os

        os.environ["OMNIEVOLVE_TEST_KEY"] = "test-value"
        self.addCleanup(os.environ.pop, "OMNIEVOLVE_TEST_KEY", None)

    def test_uses_the_model_reply(self):
        improved = CURVE_SEED.replace("A = 1.0", "A = 0.6")
        client, calls = fake_client([f"Here you go:\n```python\n{improved}```\n"])
        mutation = LLMMutator(client).mutate(cand(CURVE_SEED), random.Random(0), PY)
        self.assertEqual(mutation.operator, "mutate:llm")
        self.assertEqual(mutation.source, improved)
        prompt = calls[0]["messages"][0]["content"]
        self.assertIn("EVOLVE-BLOCK-START", prompt)
        self.assertIn("program.py", prompt)

    def test_falls_back_on_bad_replies(self):
        from omnievolve.llm import LLMError

        fallback = DeterministicMutator()
        client, _ = fake_client(
            [LLMError("HTTP 400 from api"), "```python\ndef broken(:\n```", "no code here```"]
        )
        mutator = LLMMutator(client, fallback=fallback)
        for _ in range(3):
            mutation = mutator.mutate(cand(CURVE_SEED), random.Random(0), PY)
            self.assertTrue(mutation.operator.endswith("+fallback"), mutation.operator)
            ast.parse(mutation.source)

    def test_marker_loss_is_rejected(self):
        client, _ = fake_client(["```python\ndef predict(x):\n    return x\n```"])
        mutation = LLMMutator(client).mutate(cand(CURVE_SEED), random.Random(0), PY)
        self.assertEqual(mutation.operator, "mutate:llm-failed")
        self.assertEqual(mutation.source, CURVE_SEED)

    def test_extract_source(self):
        self.assertEqual(
            extract_source("```\nshort\n```\n```py\nlonger body\n```"), "longer body\n"
        )
        self.assertEqual(extract_source("plain text"), "plain text\n")
        self.assertIsNone(extract_source("   "))


if __name__ == "__main__":
    unittest.main()
