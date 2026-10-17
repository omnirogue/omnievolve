# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""The ``omnievolve`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._version import __version__
from .config import DEFAULT_LEDGER, ConfigError, load_config
from .engine import EvolutionError
from .ledger import Ledger, LedgerError
from .llm import LLMError

EXIT_ERROR = 1
EXIT_USAGE = 2


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _out(text: str = "") -> None:
    print(text, flush=True)


def _display(path: Path | str) -> str:
    """A path relative to the working directory when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


# -- init ---------------------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    from .templates import FILES

    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    existing = [name for name in FILES if (root / name).exists()]
    if existing and not args.force:
        _err(f"{', '.join(existing)} already exist in {root} (use --force to overwrite)")
        return EXIT_ERROR
    for name, text in FILES.items():
        (root / name).write_text(text, encoding="utf-8")
    _out(f"Created an OmniEvolve project in {root}:")
    for name in FILES:
        _out(f"  {root / name}")
    _out("")
    _out("Next:")
    config, report = root / "omnievolve.json", root / "evolution-report.html"
    _out(f"  omnievolve run --config {config} --html {report}")
    return 0


# -- run ----------------------------------------------------------------------------------------


def _progress(stream):
    def on_event(event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "seed":
            print(f"seed        score {_fmt(event['score'])}", file=stream, flush=True)
        elif kind == "generation":
            islands = "  ".join(
                f"i{row['island']}={_fmt(row['best_score'])}" for row in event["islands"]
            )
            print(
                f"gen {event['generation']:>3}     best {_fmt(event['best_score'])}  "
                f"[{islands}]  evals {event['evaluations']}  cache {event['cache_hits']}",
                file=stream,
                flush=True,
            )

    return on_event


def cmd_run(args: argparse.Namespace) -> int:
    overrides = {
        "generations": args.generations,
        "islands": args.islands,
        "population": args.population,
        "seed": args.seed,
        "workers": args.workers,
    }
    project = load_config(args.config, overrides)
    if args.mutator == "llm":
        if not args.model:
            _err("--mutator llm needs --model (and --provider)")
            return EXIT_USAGE
        project.mutator = {
            "type": "llm",
            "provider": args.provider,
            "model": args.model,
            **(
                {"fallback": project.mutator}
                if project.mutator.get("type") == "deterministic"
                else {}
            ),
        }
    elif args.mutator == "deterministic":
        if project.mutator.get("type") != "deterministic":
            project.mutator = dict(project.mutator.get("fallback") or {"type": "deterministic"})
            project.mutator["type"] = "deterministic"

    if args.cloud:
        return _run_cloud(project, args)

    from .runner import run_project

    stream = sys.stderr if args.json else sys.stdout
    if not args.quiet:
        print(
            f"OmniEvolve {__version__} · {project.name} · target {project.target_label} · "
            f"{project.search.islands} islands x {project.search.population} · "
            f"{project.search.generations} generations",
            file=stream,
            flush=True,
        )
    ledger = Path(args.ledger) if args.ledger else None
    html = Path(args.html) if args.html else None
    result, report = run_project(
        project,
        ledger_path=ledger,
        html_path=html,
        on_event=None if args.quiet else _progress(stream),
    )
    summary = result.summary()
    summary["report"] = str(report) if report else None
    if args.json:
        _out(json.dumps(summary, indent=2))
        return 0
    if not args.quiet:
        _out("")
    _out(
        f"best {_fmt(result.best.score)} (seed {_fmt(result.seed.score)}, "
        f"{result.improvement:+.6g}) · candidate {result.best.id} · run {result.run_id}"
    )
    ledger_path = _display(result.ledger_path or "")
    _out(f"ledger  {ledger_path}")
    if report:
        _out(f"report  {_display(report)}")
    _out(f"next    omnievolve export-winner --ledger {ledger_path} --output best.patch")
    return 0


def _run_cloud(project, args: argparse.Namespace) -> int:
    from .cloud import CloudClient, CloudError, build_payload, summarize_payload

    try:
        body = build_payload(project)
    except CloudError as exc:
        _err(str(exc))
        return EXIT_ERROR
    if args.dry_run:
        _out(json.dumps(summarize_payload(body), indent=2))
        _err("dry run: nothing was sent")
        return 0
    try:
        client = CloudClient()
        created = client.submit(body)
        run = created.get("run", created) if isinstance(created, dict) else {}
        run_id = run.get("id") or run.get("run_id")
        _out(f"submitted to OmniSI Cloud: run {run_id} ({run.get('status', 'queued')})")
        if args.wait and run_id:
            final = client.wait(
                str(run_id),
                on_update=lambda r: _err(f"cloud run {run_id}: {r.get('status')}"),
            )
            _out(json.dumps(final, indent=2))
    except CloudError as exc:
        _err(str(exc))
        return EXIT_ERROR
    return 0


# -- ledger commands ----------------------------------------------------------------------------


def _ledger(args: argparse.Namespace) -> Ledger:
    return Ledger(args.ledger, create=False)


def cmd_runs(args: argparse.Namespace) -> int:
    with _ledger(args) as ledger:
        runs = ledger.runs()
    if args.json:
        _out(json.dumps(runs, indent=2, default=str))
        return 0
    if not runs:
        _out("no runs yet")
        return 0
    _out(f"{'run':<28} {'name':<28} {'status':<10} {'best':>10} {'cands':>6}  started")
    for run in runs:
        _out(
            f"{run['run_id']:<28} {run['name'][:28]:<28} {run['status']:<10} "
            f"{_fmt(run['best_score']):>10} {run['candidates']:>6}  {run['started_at']}"
        )
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    with _ledger(args) as ledger:
        run = ledger.run(args.run)
        board = ledger.leaderboard(run["run_id"], limit=args.limit)
        seed = ledger.candidate(run["run_id"], "seed")
    if args.json:
        rows = [{k: v for k, v in row.items() if k != "source"} for row in board]
        _out(
            json.dumps(
                {
                    "run": run["run_id"],
                    "objective": run["objective"],
                    "seed_score": seed["score"],
                    "leaderboard": rows,
                },
                indent=2,
            )
        )
        return 0
    _out(f"run {run['run_id']} · {run['name']} · {run['objective']} · seed {_fmt(seed['score'])}")
    header = f"{'#':>3}  {'candidate':<14} {'island':>6} {'gen':>4}  {'score':>10} {'delta':>10}"
    _out(header + "  operator")
    for rank, row in enumerate(board, 1):
        island = "seed" if row["island"] < 0 else str(row["island"])
        delta = row["score"] - seed["score"] if seed["score"] is not None else None
        _out(
            f"{rank:>3}  {row['candidate_id']:<14} {island:>6} {row['generation']:>4}  "
            f"{_fmt(row['score']):>10} {(f'{delta:+.4g}' if delta is not None else 'n/a'):>10}  "
            f"{row['operator']}"
        )
    return 0


def cmd_export_winner(args: argparse.Namespace) -> int:
    with _ledger(args) as ledger:
        patch, best = ledger.winner_patch(args.run)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patch, encoding="utf-8")
    changed = sum(
        1 for line in patch.splitlines() if line[:1] in "+-" and line[:3] not in ("+++", "---")
    )
    _out(
        f"wrote {output} ({changed} changed lines) · winner {best['candidate_id']} "
        f"score {_fmt(best['score'])}"
    )
    if args.source_output:
        source = Path(args.source_output)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(best["source"], encoding="utf-8")
        _out(f"wrote {source} (the winning file)")
    if not patch:
        _out("the winner is the seed itself: the patch is empty")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    with _ledger(args) as ledger:
        path = write_report(ledger, args.output, args.run)
    _out(f"wrote {path}")
    return 0


# -- parser -------------------------------------------------------------------------------------


def _err(message: str) -> None:
    print(f"omnievolve: {message}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnievolve",
        description="Evolve code and prompts locally: island-model search, pluggable "
        "evaluators, a SQLite lineage ledger and HTML reports.",
    )
    parser.add_argument("--version", action="version", version=f"omnievolve {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("init", help="scaffold a project (config, target program, evaluator)")
    p.add_argument("directory", help="where to create the project")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", help="evolve the target described by a config file")
    p.add_argument(
        "--config", "-c", default="omnievolve.json", help="project file (default: %(default)s)"
    )
    p.add_argument("--generations", "-g", type=int)
    p.add_argument("--islands", "-i", type=int)
    p.add_argument("--population", "-p", type=int, help="members per island")
    p.add_argument("--seed", type=int, help="random seed for reproducible runs")
    p.add_argument("--workers", "-w", type=int, help="parallel evaluations")
    p.add_argument(
        "--ledger", help="ledger database (default: the config's, .omnievolve/ledger.db)"
    )
    p.add_argument("--html", help="also write the HTML report here")
    p.add_argument(
        "--mutator",
        choices=("config", "deterministic", "llm"),
        default="config",
        help="override the config's mutator",
    )
    p.add_argument(
        "--provider",
        choices=("openai", "anthropic", "gemini", "ollama"),
        default="openai",
        help="LLM provider for --mutator llm",
    )
    p.add_argument("--model", help="model id for --mutator llm")
    p.add_argument(
        "--cloud",
        action="store_true",
        help="run on OmniSI Cloud instead of locally (needs OMNISI_CLOUD_URL, OMNISI_API_KEY)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="with --cloud: print the payload, send nothing"
    )
    p.add_argument("--wait", action="store_true", help="with --cloud: wait for the run to finish")
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    p.add_argument("--quiet", "-q", action="store_true", help="no progress lines")
    p.set_defaults(func=cmd_run)

    for name, func, text in (
        ("runs", cmd_runs, "list the runs in a ledger"),
        ("leaderboard", cmd_leaderboard, "show the best candidates of a run"),
        ("export-winner", cmd_export_winner, "write the seed-to-winner unified diff"),
        ("report", cmd_report, "render the HTML report of a run"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument(
            "--ledger", "-l", default=DEFAULT_LEDGER, help="ledger database (default: %(default)s)"
        )
        if name != "runs":
            p.add_argument("--run", help="run id (default: the latest completed run)")
        if name == "leaderboard":
            p.add_argument("--limit", "-n", type=int, default=10)
        if name in ("runs", "leaderboard"):
            p.add_argument("--json", action="store_true")
        if name == "export-winner":
            p.add_argument("--output", "-o", default="best.patch")
            p.add_argument("--source-output", help="also write the winning file itself here")
        if name == "report":
            p.add_argument("--output", "-o", default="evolution-report.html")
        p.set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ConfigError, LedgerError, EvolutionError, LLMError, ValueError, OSError) as exc:
        _err(str(exc))
        return EXIT_ERROR
    except KeyboardInterrupt:
        _err("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
