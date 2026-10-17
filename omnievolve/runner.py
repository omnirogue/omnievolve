# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""Runs a configured project end to end; shared by the CLI and the MCP server."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .engine import EvolutionEngine, RunResult
from .evaluators import build_evaluator
from .ledger import Ledger
from .mutators import build_mutator
from .report import write_report


def run_project(
    project: ProjectConfig,
    *,
    ledger_path: Path | None = None,
    html_path: Path | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[RunResult, Path | None]:
    """Evolves the project's target and returns the result and the report path (if any)."""
    evaluator = build_evaluator(project.evaluator, project.root)
    mutator = build_mutator(project.mutator, project.root)
    seed_source = project.target.read_text(encoding="utf-8")
    report = html_path or project.report
    with Ledger(ledger_path or project.ledger) as ledger:
        engine = EvolutionEngine(
            seed_source=seed_source,
            evaluator=evaluator,
            mutator=mutator,
            config=project.search,
            target_name=project.target_label,
            ledger=ledger,
            run_name=project.name,
            run_config={
                **project.run_config(),
                "evaluator_resolved": evaluator.describe(),
                "mutator_resolved": mutator.describe(),
            },
            on_event=on_event,
        )
        result = engine.run()
        written = write_report(ledger, report, result.run_id) if report else None
    return result, written
