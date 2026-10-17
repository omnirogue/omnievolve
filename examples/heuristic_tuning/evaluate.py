# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""Scores a triage.py candidate on a fixed synthetic ticket set and prints JSON.

Deterministic and offline: the 600 tickets come from a seeded generator, and the hidden
"true" urgency rule (with noise) is unknown to the heuristic. The score is the F1 of the
urgent class; precision, recall and accuracy are reported as metrics.

    python3 evaluate.py triage.py
"""

import importlib.util
import json
import random
import sys


def tickets(n: int = 600, seed: int = 1234) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ticket = {
            "days_open": rng.uniform(0, 14),
            "tier": rng.choice((0, 0, 1, 1, 2)),
            "sentiment": rng.uniform(-1, 1),
            "outage": 1 if rng.random() < 0.15 else 0,
            "replies": rng.randint(0, 10),
        }
        hidden = (
            0.35 * ticket["days_open"]
            + 1.2 * ticket["tier"]
            - 2.0 * ticket["sentiment"]
            + 3.0 * ticket["outage"]
            + 0.15 * ticket["replies"]
            + rng.gauss(0, 0.8)
        )
        ticket["urgent"] = hidden > 6.0
        out.append(ticket)
    return out


def load(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    candidate = load(sys.argv[1] if len(sys.argv) > 1 else "triage.py")
    tp = fp = fn = tn = 0
    for ticket in tickets():
        predicted = bool(candidate.is_urgent(ticket))
        if predicted and ticket["urgent"]:
            tp += 1
        elif predicted:
            fp += 1
        elif ticket["urgent"]:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    print(
        json.dumps(
            {
                "score": round(f1, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "accuracy": round(accuracy, 6),
            }
        )
    )


if __name__ == "__main__":
    main()
