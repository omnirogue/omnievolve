# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""Mutation and crossover operators.

``DeterministicMutator`` works offline with no API key. On Python targets it parses the file
with ``ast`` and perturbs numeric literals (weights, thresholds, sizes), rewriting only the
literal's characters so comments and formatting survive. On text targets (prompts, JSON or YAML
configs) it perturbs numbers and, given a line pool, adds, removes, replaces and swaps lines.

``LLMMutator`` asks a model you choose (OpenAI, Anthropic, Gemini or Ollama) to rewrite the file,
and falls back to the deterministic operators when the call fails or the reply is unusable.

If the target contains ``EVOLVE-BLOCK-START`` and ``EVOLVE-BLOCK-END`` marker lines, only the
lines between them change.
"""

from __future__ import annotations

import ast
import math
import re
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .llm import LLMClient, LLMConfig, LLMError

if TYPE_CHECKING:
    import random

    from .engine import Candidate

BLOCK_START = "EVOLVE-BLOCK-START"
BLOCK_END = "EVOLVE-BLOCK-END"
_TEXT_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


@dataclass
class Mutation:
    source: str
    operator: str
    note: str = ""


@dataclass
class MutationContext:
    target_name: str
    objective: str = "maximize"
    generation: int = 0
    island: int = 0
    inspirations: Sequence[Candidate] = ()

    @property
    def is_python(self) -> bool:
        return self.target_name.endswith(".py")


class Mutator:
    """Proposes new candidates. Subclasses implement ``mutate`` and may override ``crossover``."""

    name = "mutator"

    def mutate(self, parent: Candidate, rng: random.Random, ctx: MutationContext) -> Mutation:
        raise NotImplementedError

    def crossover(
        self, a: Candidate, b: Candidate, rng: random.Random, ctx: MutationContext
    ) -> Mutation:
        return Mutation(a.source, "crossover:none")

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}


# -- numeric slots ------------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericSlot:
    start: int  # character offsets into the source
    end: int
    value: float
    is_int: bool


def block_lines(source: str) -> list[tuple[int, int]]:
    """1-based inclusive line ranges strictly between block markers (empty when none)."""
    ranges, start = [], None
    for number, line in enumerate(source.splitlines(), 1):
        if BLOCK_START in line:
            start = number + 1
        elif BLOCK_END in line and start is not None:
            if number - 1 >= start:
                ranges.append((start, number - 1))
            start = None
    return ranges


def _in_blocks(line: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= line <= hi for lo, hi in ranges)


def _line_starts(source: str) -> list[int]:
    starts, total = [0], 0
    for line in source.splitlines(keepends=True):
        total += len(line)
        starts.append(total)
    return starts


def _is_number(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


def python_numeric_slots(source: str) -> list[NumericSlot]:
    """Numeric literals a parameter mutation may change.

    Inside EVOLVE blocks when the file has them; otherwise the literals in module-level
    assignments (``THRESHOLD = 0.5``); otherwise every numeric literal.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    starts = _line_starts(source)

    def offset(lineno: int, byte_col: int) -> int:
        text = lines[lineno - 1].encode("utf-8")[:byte_col].decode("utf-8", "ignore")
        return starts[lineno - 1] + len(text)

    def collect(root: ast.AST) -> list[tuple[ast.AST, float]]:
        found, negated = [], set()
        for node in ast.walk(root):
            if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub)
                and _is_number(node.operand)
            ):
                found.append((node, -node.operand.value))  # type: ignore[attr-defined]
                negated.add(id(node.operand))
        for node in ast.walk(root):
            if _is_number(node) and id(node) not in negated:
                found.append((node, node.value))  # type: ignore[attr-defined]
        return found

    ranges = block_lines(source)
    if ranges:
        nodes = [(n, v) for n, v in collect(tree) if _in_blocks(n.lineno, ranges)]
    else:
        nodes = []
        for statement in tree.body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value is not None:
                nodes.extend(collect(statement.value))
        nodes = nodes or collect(tree)
    slots = []
    for node, value in nodes:
        if node.lineno != node.end_lineno:
            continue
        start = offset(node.lineno, node.col_offset)
        end = offset(node.end_lineno, node.end_col_offset)
        slots.append(NumericSlot(start, end, float(value), isinstance(value, int)))
    return _non_overlapping(slots)


def text_numeric_slots(source: str) -> list[NumericSlot]:
    ranges = block_lines(source)
    starts = _line_starts(source)
    slots = []
    for match in _TEXT_NUMBER.finditer(source):
        if ranges:
            line = next(i for i, s in enumerate(starts[1:], 1) if match.start() < s)
            if not _in_blocks(line, ranges):
                continue
        token = match.group(0)
        is_int = not any(ch in token for ch in ".eE")
        slots.append(NumericSlot(match.start(), match.end(), float(token), is_int))
    return slots


def _non_overlapping(slots: list[NumericSlot]) -> list[NumericSlot]:
    out: list[NumericSlot] = []
    for slot in sorted(slots, key=lambda s: (s.start, -(s.end - s.start))):
        if out and slot.start < out[-1].end:
            continue
        out.append(slot)
    return out


def format_float(value: float) -> str:
    if not math.isfinite(value):
        value = 0.0
    text = f"{value:.4g}"
    if not any(ch in text for ch in ".e"):
        text += ".0"
    return text


def _replace(source: str, edits: dict[NumericSlot, str]) -> str:
    out = source
    for slot in sorted(edits, key=lambda s: s.start, reverse=True):
        out = out[: slot.start] + edits[slot] + out[slot.end :]
    return out


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


# -- deterministic mutator ----------------------------------------------------------------------


class DeterministicMutator(Mutator):
    """Offline operators: numeric perturbation, line edits and uniform parameter crossover.

    ``sigma`` scales a float's Gaussian step relative to its magnitude (with ``min_scale`` as a
    floor so zero can move); integers step by up to ``sigma`` of their size, at least 1, and
    non-negative integers stay non-negative. ``line_pool`` enables line operations on text
    targets; ``numeric_rate`` is the share of text mutations that touch numbers instead.
    """

    name = "deterministic"

    def __init__(
        self,
        sigma: float = 0.25,
        max_changes: int = 2,
        min_scale: float = 0.1,
        line_pool: Sequence[str] | None = None,
        numeric_rate: float = 0.2,
        kind: str = "auto",
    ):
        if kind not in ("auto", "python", "text"):
            raise ValueError("mutator.kind must be auto, python or text")
        self.sigma = float(sigma)
        self.max_changes = max(1, int(max_changes))
        self.min_scale = float(min_scale)
        self.line_pool = [line.rstrip("\n") for line in (line_pool or []) if line.strip()]
        self.numeric_rate = float(numeric_rate)
        self.kind = kind

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "sigma": self.sigma,
            "max_changes": self.max_changes,
            "min_scale": self.min_scale,
            "line_pool_size": len(self.line_pool),
            "numeric_rate": self.numeric_rate,
            "kind": self.kind,
        }

    def _python(self, ctx: MutationContext) -> bool:
        return self.kind == "python" or (self.kind == "auto" and ctx.is_python)

    def _slots(self, source: str, ctx: MutationContext) -> list[NumericSlot]:
        if self._python(ctx):
            try:
                return python_numeric_slots(source)
            except SyntaxError:
                return text_numeric_slots(source)
        return text_numeric_slots(source)

    def _perturb(self, slot: NumericSlot, rng: random.Random) -> str:
        if slot.is_int:
            value = int(slot.value)
            step = max(1, int(round(abs(value) * self.sigma)))
            new = value + rng.choice((-1, 1)) * rng.randint(1, step)
            if value >= 0:
                new = max(0, new)
            if new == value:
                new = value + 1
            return str(new)
        scale = max(abs(slot.value), self.min_scale)
        new_value = slot.value + rng.gauss(0.0, self.sigma) * scale
        text = format_float(new_value)
        if float(text) == slot.value:
            text = format_float(slot.value + math.copysign(scale * 0.01, new_value - slot.value))
        return text

    def mutate(self, parent: Candidate, rng: random.Random, ctx: MutationContext) -> Mutation:
        source = parent.source
        slots = self._slots(source, ctx)
        use_lines = bool(self.line_pool) and not self._python(ctx)
        if use_lines and (not slots or rng.random() >= self.numeric_rate):
            mutation = self._line_mutation(source, rng)
            if mutation is not None:
                return mutation
        if not slots:
            return Mutation(source, "mutate:noop", "no numeric literal or line pool to mutate")
        for _ in range(4):
            count = rng.randint(1, min(self.max_changes, len(slots)))
            chosen = rng.sample(slots, count)
            child = _replace(source, {slot: self._perturb(slot, rng) for slot in chosen})
            if not self._python(ctx) or _parses(child):
                return Mutation(child, f"mutate:params({count})")
        return Mutation(source, "mutate:noop", "parameter edits did not parse")

    # -- lines ------------------------------------------------------------------------------

    def _editable(self, lines: list[str]) -> range:
        ranges = block_lines("\n".join(lines))
        if ranges:
            lo, hi = ranges[0]
            return range(lo - 1, hi)
        return range(len(lines))

    def _line_mutation(self, source: str, rng: random.Random) -> Mutation | None:
        lines = source.splitlines()
        trailing = "\n" if source.endswith("\n") else ""
        window = self._editable(lines)
        content = [i for i in window if lines[i].strip()]
        present = {line.strip() for line in lines}
        fresh = [line for line in self.line_pool if line.strip() not in present]
        ops = []
        if fresh:
            ops.append("add")
        if len(content) > 1:
            ops += ["remove", "swap"]
        if content and fresh:
            ops.append("replace")
        if not ops:
            return None
        op = rng.choice(ops)
        if op == "add":
            position = rng.randint(window.start, window.stop) if len(window) else len(lines)
            lines.insert(position, rng.choice(fresh))
        elif op == "remove":
            del lines[rng.choice(content)]
        elif op == "replace":
            lines[rng.choice(content)] = rng.choice(fresh)
        else:
            k = rng.randrange(len(content) - 1)
            i, j = content[k], content[k + 1]
            lines[i], lines[j] = lines[j], lines[i]
        return Mutation("\n".join(lines) + trailing, f"mutate:{op}-line")

    # -- crossover --------------------------------------------------------------------------

    def crossover(
        self, a: Candidate, b: Candidate, rng: random.Random, ctx: MutationContext
    ) -> Mutation:
        slots_a, slots_b = self._slots(a.source, ctx), self._slots(b.source, ctx)
        if slots_a and len(slots_a) == len(slots_b):
            edits = {}
            for sa, sb in zip(slots_a, slots_b, strict=True):
                text_b = b.source[sb.start : sb.end]
                if a.source[sa.start : sa.end] != text_b and rng.random() < 0.5:
                    edits[sa] = text_b
            child = _replace(a.source, edits)
            if edits and (not self._python(ctx) or _parses(child)):
                return Mutation(child, "crossover:params")
        if not self._python(ctx):
            la = [line for line in a.source.splitlines() if line.strip()]
            lb = [line for line in b.source.splitlines() if line.strip()]
            if len(la) > 1 and len(lb) > 1 and not block_lines(a.source):
                cut_a, cut_b = rng.randint(1, len(la) - 1), rng.randint(1, len(lb) - 1)
                head = la[:cut_a]
                child_lines = head + [line for line in lb[cut_b:] if line not in head]
                trailing = "\n" if a.source.endswith("\n") else ""
                return Mutation("\n".join(child_lines) + trailing, "crossover:lines")
        return Mutation(a.source, "crossover:none")


# -- LLM mutator ---------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You improve one file inside an automated evolutionary search. An evaluator scores every "
    "version you return, and better-scoring versions survive. Make one focused, well-reasoned "
    "change per reply. Keep the file's public interface (names, signatures, output format) "
    "unchanged unless the instructions say otherwise. Reply with the complete updated file in a "
    "single fenced code block and nothing else."
)


def _fence(source: str, ctx: MutationContext) -> str:
    language = "python" if ctx.is_python else ""
    return f"```{language}\n{source.rstrip()}\n```"


def _score_line(candidate: Candidate) -> str:
    score = "n/a" if candidate.score is None else f"{candidate.score:.6g}"
    metrics = ", ".join(f"{k}={v:.4g}" for k, v in sorted(candidate.metrics.items())[:12])
    return f"score {score}" + (f" ({metrics})" if metrics else "")


def extract_source(reply: str) -> str | None:
    """The longest fenced code block in a reply, or the whole reply when it has no fence."""
    blocks = _FENCE.findall(reply)
    if blocks:
        return max(blocks, key=len).rstrip() + "\n"
    text = reply.strip()
    if text and "```" not in text:
        return text + "\n"
    return None


class LLMMutator(Mutator):
    """Rewrites candidates with a language model; deterministic operators are the fallback."""

    name = "llm"

    def __init__(
        self,
        client: LLMClient,
        fallback: Mutator | None = None,
        instructions: str = "",
        max_source_chars: int = 24_000,
    ):
        self.client = client
        self.fallback = fallback
        self.instructions = instructions.strip()
        self.max_source_chars = max_source_chars
        self._warned = False
        self._warn_lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.name,
            **self.client.config.describe(),
            "fallback": self.fallback.describe() if self.fallback else None,
            "instructions": bool(self.instructions),
        }

    def _clip(self, source: str) -> str:
        if len(source) <= self.max_source_chars:
            return source
        return source[: self.max_source_chars] + "\n... (truncated)\n"

    def _header(self, ctx: MutationContext) -> list[str]:
        goal = "higher" if ctx.objective == "maximize" else "lower"
        lines = [
            f"Target file: {ctx.target_name}",
            f"Objective: {ctx.objective} the evaluator's score ({goal} is better).",
        ]
        if self.instructions:
            lines += ["", "Task instructions:", self.instructions]
        return lines

    def mutate(self, parent: Candidate, rng: random.Random, ctx: MutationContext) -> Mutation:
        lines = self._header(ctx)
        lines += [
            "",
            f"Current version ({_score_line(parent)}):",
            _fence(self._clip(parent.source), ctx),
        ]
        if ctx.inspirations:
            lines += ["", "Other strong versions from this population (ideas, not to copy):"]
            for index, other in enumerate(ctx.inspirations[:2], 1):
                lines += [
                    f"Version {index} ({_score_line(other)}):",
                    _fence(self._clip(other.source), ctx),
                ]
        lines += ["", self._rules(parent.source)]
        return self._ask("\n".join(lines), parent, rng, ctx, "mutate:llm")

    def crossover(
        self, a: Candidate, b: Candidate, rng: random.Random, ctx: MutationContext
    ) -> Mutation:
        lines = self._header(ctx)
        lines += [
            "",
            f"Parent A ({_score_line(a)}):",
            _fence(self._clip(a.source), ctx),
            "",
            f"Parent B ({_score_line(b)}):",
            _fence(self._clip(b.source), ctx),
            "",
            "Combine the strongest ideas of both parents into one version that should score "
            "better than either.",
            self._rules(a.source),
        ]
        return self._ask("\n".join(lines), a, rng, ctx, "crossover:llm")

    @staticmethod
    def _rules(source: str) -> str:
        rules = "Return the complete updated file in one fenced code block."
        if block_lines(source):
            rules += (
                f" Change only the lines between {BLOCK_START} and {BLOCK_END} and keep both "
                "marker lines exactly as they are."
            )
        return rules

    def _ask(
        self, prompt: str, parent: Candidate, rng: random.Random, ctx: MutationContext, op: str
    ) -> Mutation:
        try:
            reply = self.client.complete(SYSTEM_PROMPT, prompt)
        except LLMError as exc:
            return self._fallback(parent, rng, ctx, f"LLM call failed: {exc}")
        source = extract_source(reply)
        if source is None:
            return self._fallback(parent, rng, ctx, "the reply had no code block")
        if ctx.is_python and not _parses(source):
            return self._fallback(parent, rng, ctx, "the reply was not valid Python")
        if block_lines(parent.source) and not block_lines(source):
            return self._fallback(parent, rng, ctx, "the reply dropped the EVOLVE-BLOCK markers")
        if not parent.source.endswith("\n"):
            source = source.rstrip("\n")
        return Mutation(source, op)

    def _fallback(
        self, parent: Candidate, rng: random.Random, ctx: MutationContext, reason: str
    ) -> Mutation:
        with self._warn_lock:
            if not self._warned:
                self._warned = True
                print(f"omnievolve: {reason}; using the fallback mutator", file=sys.stderr)
        if self.fallback is None:
            return Mutation(parent.source, "mutate:llm-failed", reason)
        mutation = self.fallback.mutate(parent, rng, ctx)
        return Mutation(mutation.source, mutation.operator + "+fallback", reason)


# -- factory ------------------------------------------------------------------------------------


def _deterministic_from(spec: dict[str, Any], root: Path) -> DeterministicMutator:
    pool = spec.get("line_pool")
    lines: list[str] | None = None
    if isinstance(pool, str):
        path = (root / pool).resolve()
        lines = path.read_text(encoding="utf-8").splitlines()
    elif isinstance(pool, list):
        lines = [str(line) for line in pool]
    return DeterministicMutator(
        sigma=spec.get("sigma", 0.25),
        max_changes=spec.get("max_changes", 2),
        min_scale=spec.get("min_scale", 0.1),
        line_pool=lines,
        numeric_rate=spec.get("numeric_rate", 0.2),
        kind=spec.get("kind", "auto"),
    )


DETERMINISTIC_KEYS = {
    "type",
    "sigma",
    "max_changes",
    "min_scale",
    "line_pool",
    "numeric_rate",
    "kind",
}
LLM_KEYS = set(LLMConfig.__dataclass_fields__) | {"type", "fallback", "instructions"}


def build_mutator(spec: dict[str, Any] | None, root: Path) -> Mutator:
    """Builds a mutator from the config's ``mutator`` section."""
    spec = dict(spec or {"type": "deterministic"})
    kind = spec.get("type", "deterministic")
    if kind == "deterministic":
        unknown = sorted(set(spec) - DETERMINISTIC_KEYS)
        if unknown:
            raise ValueError(f"unknown mutator setting(s): {', '.join(unknown)}")
        return _deterministic_from(spec, root)
    if kind == "llm":
        unknown = sorted(set(spec) - LLM_KEYS)
        if unknown:
            raise ValueError(f"unknown mutator setting(s): {', '.join(unknown)}")
        llm = {k: v for k, v in spec.items() if k in LLMConfig.__dataclass_fields__}
        client = LLMClient(LLMConfig.from_dict(llm))
        fallback_spec = spec.get("fallback", {"type": "deterministic"})
        fallback = None
        if fallback_spec:
            fallback_spec = {"type": "deterministic", **dict(fallback_spec)}
            fallback = _deterministic_from(fallback_spec, root)
        instructions = spec.get("instructions", "")
        if isinstance(instructions, str) and instructions.startswith("@"):
            instructions = (root / instructions[1:]).read_text(encoding="utf-8")
        return LLMMutator(client, fallback=fallback, instructions=instructions)
    raise ValueError(f"unknown mutator type {kind!r} (use deterministic or llm)")


__all__ = [
    "BLOCK_END",
    "BLOCK_START",
    "DeterministicMutator",
    "LLMMutator",
    "Mutation",
    "MutationContext",
    "Mutator",
    "NumericSlot",
    "build_mutator",
    "extract_source",
    "format_float",
    "python_numeric_slots",
    "text_numeric_slots",
]
