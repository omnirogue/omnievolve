# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""A self-contained, dark-mode HTML report for one run.

No scripts, fonts or images are fetched: the charts are inline SVG and hover details use SVG
``<title>`` tooltips. Sections: headline numbers, the fitness curve per generation (best so far
and each island's best), the lineage DAG (every candidate by generation and score, with the
winner's ancestry highlighted), the island and candidate leaderboards, the ancestor chain from
seed to winner with each step's diff, and the winning patch.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._version import __version__
from .ledger import Ledger

# Categorical slots stepped for a dark surface, in fixed order (never cycled past eight).
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

CSS = """
:root { color-scheme: dark; --bg: #111110; --surface: #1a1a19; --surface-2: #222220;
  --border: #2e2e2b; --grid: #2a2a28; --text: #f0efec; --text-2: #c3c2b7; --muted: #8f8e86;
  --accent: #3987e5; --add: #7fd1a4; --del: #f19a9a; --hunk: #9fb7e6; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 0 0 12px; }
.sub { color: var(--text-2); margin: 0 0 24px; }
.sub code { color: var(--text); }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 20px; margin: 0 0 20px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile { background: var(--surface-2); border-radius: 8px; padding: 12px 14px; }
.tile .k { color: var(--text-2); font-size: 12px; }
.tile .v { font-size: 22px; font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 0 0 8px; color: var(--text-2);
  font-size: 12px; }
.legend span::before { content: ""; display: inline-block; width: 14px; height: 3px;
  margin-right: 6px; vertical-align: middle; background: var(--c); border-radius: 2px; }
svg text { fill: var(--text-2); font-size: 11px; }
svg .label { fill: var(--text); font-size: 12px; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
th { color: var(--text-2); font-weight: 600; font-size: 12px; }
td.n, th.n { text-align: right; }
code, pre { font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px;
  overflow: auto; max-height: 520px; margin: 8px 0 0; }
pre .a { color: var(--add); } pre .d { color: var(--del); } pre .h { color: var(--hunk); }
pre .f { color: var(--muted); }
details { margin: 8px 0 0; } summary { cursor: pointer; color: var(--text-2); }
ol.chain { list-style: none; padding: 0; margin: 0; }
ol.chain li { border-left: 2px solid var(--border); padding: 4px 0 10px 16px; position: relative; }
ol.chain li::before { content: ""; position: absolute; left: -6px; top: 10px; width: 10px;
  height: 10px; border-radius: 50%; background: var(--surface); border: 2px solid var(--accent); }
.up { color: var(--add); } .down { color: var(--del); } .muted { color: var(--muted); }
footer { color: var(--muted); font-size: 12px; text-align: center; margin-top: 24px; }
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}g}" if abs(value) >= 1e-3 or value == 0 else f"{value:.3e}"


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, count)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    start = math.floor(lo / step) * step
    ticks, value = [], start
    while value <= hi + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


class _Frame:
    """Maps (generation, score) to chart coordinates."""

    def __init__(self, width, height, x_max, y_lo, y_hi, pad=(16, 24, 34, 52)):
        self.w, self.h = width, height
        self.top, self.right, self.bottom, self.left = pad
        self.x_max = max(1, x_max)
        ticks = _nice_ticks(y_lo, y_hi)
        self.y_ticks = ticks
        self.y_lo, self.y_hi = ticks[0], ticks[-1] if ticks[-1] > ticks[0] else ticks[0] + 1

    def x(self, generation: float) -> float:
        return self.left + (self.w - self.left - self.right) * generation / self.x_max

    def y(self, score: float) -> float:
        span = self.y_hi - self.y_lo
        return self.top + (self.h - self.top - self.bottom) * (1 - (score - self.y_lo) / span)

    def axes(self, x_label: str, y_label: str) -> list[str]:
        out = []
        for tick in self.y_ticks:
            y = self.y(tick)
            out.append(
                f'<line x1="{self.left}" x2="{self.w - self.right}" y1="{y:.1f}" y2="{y:.1f}" '
                'stroke="var(--grid)" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{self.left - 8}" y="{y + 4:.1f}" text-anchor="end">{_fmt(tick)}</text>'
            )
        step = max(1, math.ceil(self.x_max / 12))
        for generation in range(0, self.x_max + 1, step):
            x = self.x(generation)
            out.append(
                f'<text x="{x:.1f}" y="{self.h - self.bottom + 16}" text-anchor="middle">'
                f"{generation}</text>"
            )
        base = self.h - self.bottom
        out.append(
            f'<line x1="{self.left}" x2="{self.w - self.right}" y1="{base}" y2="{base}" '
            'stroke="var(--border)" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{(self.left + self.w - self.right) / 2:.1f}" y="{self.h - 4}" '
            f'text-anchor="middle">{_e(x_label)}</text>'
        )
        out.append(
            f'<text x="12" y="{(self.top + base) / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 12 {(self.top + base) / 2:.1f})">{_e(y_label)}</text>'
        )
        return out


def _svg(width: int, height: int, body: Sequence[str], label: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{_e(label)}" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(body)}</svg>'
    )


def _bounds(values: Sequence[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.08 or max(abs(hi) * 0.05, 0.05)
    return lo - pad, hi + pad


def fitness_chart(history: dict[int, dict[int, dict]], best_curve: list[tuple[int, float]]):
    """Best so far (all islands) plus each island's best, per generation."""
    islands = sorted({i for rows in history.values() for i in rows})
    values = [s for _, s in best_curve] + [
        r["best_score"]
        for rows in history.values()
        for r in rows.values()
        if r["best_score"] is not None
    ]
    if not values:
        return "<p class='muted'>No scored generations.</p>"
    width, height = 1040, 340
    frame = _Frame(width, height, max(history) if history else 1, *_bounds(values))
    body = frame.axes("generation", "score")
    series = [("best so far", SERIES[0], best_curve, 2.5)]
    for index, island in enumerate(islands[: len(SERIES) - 1], 1):
        points = [
            (g, rows[island]["best_score"])
            for g, rows in sorted(history.items())
            if island in rows and rows[island]["best_score"] is not None
        ]
        series.append((f"island {island}", SERIES[index], points, 1.5))
    for name, color, points, stroke in reversed(series):
        if not points:
            continue
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{frame.x(g):.1f},{frame.y(s):.1f}"
            for i, (g, s) in enumerate(points)
        )
        dash = "" if name == "best so far" else ' stroke-dasharray="5 4"'
        body.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke}"'
            f' stroke-linejoin="round" stroke-linecap="round"{dash}/>'
        )
        for g, s in points:
            x, y = frame.x(g), frame.y(s)
            body.append(
                f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" '
                f'stroke="var(--surface)" stroke-width="2"/><circle cx="{x:.1f}" cy="{y:.1f}" '
                f'r="9" fill="transparent"><title>{_e(name)} · generation {g}: '
                f"{_fmt(s, 6)}</title></circle></g>"
            )
    g_last, s_last = best_curve[-1]
    body.append(
        f'<text class="label" x="{frame.x(g_last) - 6:.1f}" y="{frame.y(s_last) - 10:.1f}" '
        f'text-anchor="end">best {_fmt(s_last, 6)}</text>'
    )
    legend = "".join(f'<span style="--c:{color}">{_e(name)}</span>' for name, color, _, _ in series)
    return f'<div class="legend">{legend}</div>' + _svg(
        width, height, body, "Fitness by generation"
    )


def lineage_chart(candidates: list[dict], edges: list[dict], ancestry_ids: set[str]) -> str:
    """Every candidate by generation (x) and score (y); edges are parent-to-child links."""
    scored = {c["candidate_id"]: c for c in candidates if c["ok"] and c["score"] is not None}
    if not scored:
        return "<p class='muted'>No scored candidates.</p>"
    width, height = 1040, 380
    x_max = max(c["generation"] for c in scored.values())
    frame = _Frame(width, height, x_max, *_bounds([c["score"] for c in scored.values()]))
    body = frame.axes("generation", "score")

    def point(c: dict) -> tuple[float, float]:
        # Spread islands slightly so same-generation candidates do not sit on one line.
        jitter = 0.0 if c["island"] < 0 else (c["island"] - 0.5) * 0.12
        return frame.x(c["generation"] + jitter), frame.y(c["score"])

    for edge in edges:
        parent, child = scored.get(edge["parent_id"]), scored.get(edge["child_id"])
        if parent is None or child is None:
            continue
        highlighted = edge["parent_id"] in ancestry_ids and edge["child_id"] in ancestry_ids
        if highlighted and edge["role"] == "secondary":
            highlighted = False
        (x1, y1), (x2, y2) = point(parent), point(child)
        color, w, opacity = ("var(--accent)", 2, 1) if highlighted else ("var(--muted)", 1, 0.25)
        body.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
            f'stroke-width="{w}" stroke-opacity="{opacity}"/>'
        )
    ordered = sorted(scored.values(), key=lambda c: c["candidate_id"] in ancestry_ids)
    for c in ordered:
        x, y = point(c)
        on_path = c["candidate_id"] in ancestry_ids
        fill = SERIES[0] if on_path else "var(--text-2)"
        radius = 5 if on_path else 3
        tip = (
            f"{c['candidate_id']} · island {c['island'] if c['island'] >= 0 else 'all'} · "
            f"{c['operator']} · score {_fmt(c['score'], 6)}"
        )
        body.append(
            f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" '
            f'fill-opacity="{1 if on_path else 0.55}" stroke="var(--surface)" stroke-width="2"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="transparent"><title>{_e(tip)}'
            "</title></circle></g>"
        )
    legend = (
        f'<div class="legend"><span style="--c:{SERIES[0]}">winner\'s ancestry</span>'
        '<span style="--c:var(--text-2)">other candidates</span></div>'
    )
    return legend + _svg(width, height, body, "Lineage of every candidate")


def _diff_html(diff: str) -> str:
    if not diff.strip():
        return "<pre class='muted'>(no change)</pre>"
    out = []
    for line in diff.splitlines():
        cls = ""
        if line.startswith(("+++", "---")):
            cls = "f"
        elif line.startswith("@@"):
            cls = "h"
        elif line.startswith("+"):
            cls = "a"
        elif line.startswith("-"):
            cls = "d"
        out.append(f'<span class="{cls}">{_e(line)}</span>' if cls else _e(line))
    return "<pre>" + "\n".join(out) + "</pre>"


def _delta(value: float | None, objective: str) -> str:
    if value is None or value == 0:
        return '<span class="muted">±0</span>'
    good = value > 0 if objective == "maximize" else value < 0
    return f'<span class="{"up" if good else "down"}">{value:+.4g}</span>'


def render_report(ledger: Ledger, run_id: str | None = None) -> str:
    run = ledger.run(run_id)
    rid = run["run_id"]
    objective = run["objective"]
    candidates = ledger.candidates(rid)
    edges = ledger.edges(rid)
    generations = ledger.generations(rid)
    seed = ledger.candidate(rid, "seed")
    best = ledger.best(rid)
    ancestry = ledger.ancestry(rid, best["candidate_id"])
    ancestry_ids = {c["candidate_id"] for c in ancestry}
    patch, _ = ledger.winner_patch(rid)

    history: dict[int, dict[int, dict]] = {}
    for row in generations:
        history.setdefault(row["generation"], {})[row["island"]] = row
    better = (lambda a, b: a > b) if objective == "maximize" else (lambda a, b: a < b)
    best_curve, running = [], seed["score"]
    for generation in sorted(history):
        for row in history[generation].values():
            if row["best_score"] is not None and better(row["best_score"], running):
                running = row["best_score"]
        best_curve.append((generation, running))

    improvement = best["score"] - seed["score"]
    relative = f"{improvement / abs(seed['score']) * 100:+.1f}%" if seed["score"] else "n/a"
    evaluated = sum(1 for c in candidates if not c["cached"])
    reason = run["stopped_reason"]
    status_text = run["status"]
    if reason and reason not in (run["status"], "completed"):
        status_text += f" · {reason.replace('_', ' ')}"
    tiles = [
        ("Best score", _fmt(best["score"], 6)),
        ("Seed score", _fmt(seed["score"], 6)),
        ("Improvement", f"{improvement:+.4g} ({relative})"),
        ("Generations", str(max(history) if history else 0)),
        ("Islands", str(len({r["island"] for r in generations}))),
        ("Candidates", f"{len(candidates)} ({evaluated} evaluated)"),
        ("Wall time", f"{run['duration_s'] or 0:.2f}s"),
        ("Status", status_text),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div></div>'
        for k, v in tiles
    )

    island_rows = ledger.islands(rid)
    if not island_rows and history:
        last = history[max(history)]
        island_rows = [
            {
                "island": i,
                "best_score": r["best_score"],
                "best_candidate_id": r["best_candidate_id"],
                "mean_score": r["mean_score"],
                "size": None,
            }
            for i, r in sorted(last.items())
        ]
    island_rows = sorted(
        island_rows,
        key=lambda r: (
            r["best_score"] is None,
            -(r["best_score"] or 0) if objective == "maximize" else (r["best_score"] or 0),
        ),
    )
    island_html = "".join(
        f"<tr><td>{rank}</td><td>island {_e(r['island'])}</td>"
        f"<td class='n'>{_fmt(r['best_score'], 6)}</td>"
        f"<td class='n'>{_fmt(r['mean_score'], 6)}</td>"
        f"<td><code>{_e(r['best_candidate_id'])}</code></td>"
        f"<td class='n'>{_e(r['size'] if r['size'] is not None else '')}</td></tr>"
        for rank, r in enumerate(island_rows, 1)
    )

    board = ledger.leaderboard(rid, limit=12)

    def metric_text(c: dict) -> str:
        return ", ".join(f"{k}={_fmt(v)}" for k, v in list(c["metrics"].items())[:4])

    board_html = "".join(
        f"<tr><td>{rank}</td><td><code>{_e(c['candidate_id'])}</code></td>"
        f"<td>{_e(c['island'] if c['island'] >= 0 else 'seed')}</td>"
        f"<td class='n'>{c['generation']}</td>"
        f"<td>{_e(c['operator'])}</td><td class='n'>{_fmt(c['score'], 6)}</td>"
        f"<td class='n'>{c['latency_ms']:.0f}</td>"
        f"<td class='muted'>{_e(metric_text(c))}</td></tr>"
        for rank, c in enumerate(board, 1)
    )

    chain_items = []
    for index, node in enumerate(ancestry):
        previous = ancestry[index - 1] if index else None
        delta = _delta(node["score"] - previous["score"], objective) if previous else ""
        diff = (
            ledger.edge_diff(rid, previous["candidate_id"], node["candidate_id"])
            if previous
            else ""
        )
        detail = f"<details><summary>diff</summary>{_diff_html(diff)}</details>" if previous else ""
        island = "all islands" if node["island"] < 0 else f"island {node['island']}"
        chain_items.append(
            f"<li><code>{_e(node['candidate_id'])}</code> · generation {node['generation']} · "
            f"{_e(island)} · {_e(node['operator'])} · score <b>{_fmt(node['score'], 6)}</b> "
            f"{delta}{detail}</li>"
        )

    table_rows = "".join(
        f"<tr><td class='n'>{g}</td><td class='n'>{_fmt(s, 6)}</td>"
        + "".join(
            f"<td class='n'>{_fmt(history[g][i]['best_score'], 6)}</td>" for i in sorted(history[g])
        )
        + "</tr>"
        for g, s in best_curve
    )
    island_ids = sorted({r["island"] for r in generations})
    table_head = "".join(f"<th class='n'>island {i}</th>" for i in island_ids)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmniEvolve report · {_e(run["name"])}</title>
<style>{CSS}</style></head>
<body><main>
<h1>OmniEvolve · {_e(run["name"])}</h1>
<p class="sub">Run <code>{_e(rid)}</code> · target <code>{_e(run["target"])}</code> ·
{_e(objective)} · started {_e(run["started_at"])} ·
{_e(run["engine"])} {_e(run["engine_version"])}</p>
<section><div class="tiles">{tiles_html}</div></section>
<section><h2>Fitness by generation</h2>{fitness_chart(history, best_curve)}
<details><summary>Table view</summary><table><thead><tr><th class='n'>generation</th>
<th class='n'>best so far</th>{table_head}</tr></thead><tbody>{table_rows}</tbody></table></details>
</section>
<section><h2>Lineage</h2>{lineage_chart(candidates, edges, ancestry_ids)}</section>
<section><h2>Island leaderboard</h2><table><thead><tr><th>#</th><th>island</th>
<th class='n'>best</th><th class='n'>mean</th><th>best candidate</th><th class='n'>size</th></tr>
</thead><tbody>{island_html}</tbody></table></section>
<section><h2>Top candidates</h2><table><thead><tr><th>#</th><th>candidate</th><th>island</th>
<th class='n'>gen</th><th>operator</th><th class='n'>score</th><th class='n'>ms</th><th>metrics</th>
</tr></thead><tbody>{board_html}</tbody></table></section>
<section><h2>Ancestry of the winner</h2><ol class="chain">{"".join(chain_items)}</ol></section>
<section><h2>Winning patch</h2><p class="sub">Seed to <code>{_e(best["candidate_id"])}</code>.
Apply with <code>git apply</code> or <code>patch -p1</code>.</p>{_diff_html(patch)}</section>
<footer>Generated by OmniEvolve {__version__} (Apache-2.0). Self-contained: no external resources.
</footer>
</main></body></html>
"""


def write_report(ledger: Ledger, output: Path | str, run_id: str | None = None) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(ledger, run_id), encoding="utf-8")
    return path
