# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""Shared fixtures for the test suite (standard library only)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnievolve.evaluators import PythonCallableEvaluator  # noqa: E402

CURVE_SEED = """\
# EVOLVE-BLOCK-START
A = 1.0  # quadratic term
B = 0.0
C = 0.5
# EVOLVE-BLOCK-END


def predict(x):
    return A * x * x + B * x + C
"""


def load_module(path: Path, name: str = "candidate"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def curve_score(path: Path) -> dict:
    module = load_module(path)
    xs = [i / 4 for i in range(-12, 13)]
    mse = sum((module.predict(x) - (0.6 * x * x - 1.3 * x + 2.0)) ** 2 for x in xs) / len(xs)
    return {"score": 1 / (1 + mse), "mse": mse}


def curve_evaluator() -> PythonCallableEvaluator:
    return PythonCallableEvaluator(curve_score)
