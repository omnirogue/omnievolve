# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""Files written by ``omnievolve init``: a tiny curve-fitting project that runs offline."""

from __future__ import annotations

PROGRAM = '''\
# SPDX-License-Identifier: Apache-2.0
"""The program OmniEvolve improves. Only the lines between the EVOLVE-BLOCK markers change."""

# EVOLVE-BLOCK-START
A = 1.0
B = 0.0
C = 0.5
# EVOLVE-BLOCK-END


def predict(x: float) -> float:
    return A * x * x + B * x + C
'''

EVALUATE = '''\
# SPDX-License-Identifier: Apache-2.0
"""Scores a program.py candidate: how closely predict() matches a hidden curve.

Prints {"score": ..., "mse": ...} as JSON. Higher score is better (1.0 is a perfect fit).
Replace this with your own benchmark: anything that prints a score works.
"""

import importlib.util
import json
import sys


def main() -> None:
    spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    xs = [i / 4 for i in range(-12, 13)]
    errors = [(module.predict(x) - (0.6 * x * x - 1.3 * x + 2.0)) ** 2 for x in xs]
    mse = sum(errors) / len(errors)
    print(json.dumps({"score": round(1 / (1 + mse), 6), "mse": round(mse, 6)}))


if __name__ == "__main__":
    main()
'''

CONFIG = """\
{
  "name": "starter",
  "target": "program.py",
  "evaluator": {
    "type": "subprocess",
    "command": ["{python}", "evaluate.py", "{candidate}"],
    "timeout_seconds": 10
  },
  "mutator": {"type": "deterministic", "sigma": 0.3, "max_changes": 2},
  "search": {"generations": 8, "islands": 2, "population": 6, "seed": 7, "workers": 4},
  "objective": "maximize",
  "ledger": ".omnievolve/ledger.db"
}
"""

GITIGNORE = """\
.omnievolve/
*.patch
evolution-report.html
"""

FILES = {
    "omnievolve.json": CONFIG,
    "program.py": PROGRAM,
    "evaluate.py": EVALUATE,
    ".gitignore": GITIGNORE,
}
