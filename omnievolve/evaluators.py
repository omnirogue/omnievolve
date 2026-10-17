# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""The evaluator plugin SDK.

An evaluator turns one candidate file into a score (plus optional named metrics). Subclass
``Evaluator`` and implement ``_evaluate``; return a number, a dict with a ``score`` key, or an
``EvaluationResult``. The base class times the call and turns exceptions into failed results,
so one broken candidate never stops a run.

Built in:

* ``SubprocessEvaluator`` runs your benchmark command with a timeout and reads the score from
  its standard output (a JSON object, or a bare number on the last line).
* ``PythonCallableEvaluator`` calls a Python function in-process (fast; not isolated).
* ``LLMJudgeEvaluator`` asks a model to grade the candidate against your rubric.
* ``CompositeEvaluator`` combines several evaluators into a weighted score.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import LLMClient, LLMConfig, LLMError

_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")
_SCORE_LINE = re.compile(r"\bscore\b\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", re.I)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class EvaluationResult:
    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    log: str = ""

    @classmethod
    def failure(cls, error: str, log: str = "") -> EvaluationResult:
        return cls(score=None, ok=False, error=error, log=log)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def coerce_result(value: Any, score_key: str = "score") -> EvaluationResult:
    """Accepts a number, a mapping with ``score_key`` or an ``EvaluationResult``."""
    if isinstance(value, EvaluationResult):
        return value
    number = _numeric(value)
    if number is not None:
        return EvaluationResult(score=number)
    if isinstance(value, dict):
        metrics = {
            str(k): n for k, v in value.items() if k != score_key and (n := _numeric(v)) is not None
        }
        score = _numeric(value.get(score_key))
        if score is None:
            error = value.get("error") or f"no numeric {score_key!r} in the evaluator output"
            return EvaluationResult(score=None, metrics=metrics, ok=False, error=str(error))
        return EvaluationResult(score=score, metrics=metrics)
    return EvaluationResult.failure(f"cannot read a score from {type(value).__name__}")


def parse_score_output(stdout: str, score_key: str = "score") -> EvaluationResult:
    """Reads a score from a benchmark's standard output.

    In order: the whole output as a JSON object; the last line that is a JSON object; the last
    line that is a bare number; the last ``score: <number>`` or ``score=<number>``.
    """
    text = stdout.strip()
    if not text:
        return EvaluationResult.failure("the evaluator printed nothing")
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, (dict, int, float)) and not isinstance(whole, bool):
        return coerce_result(whole, score_key)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                return coerce_result(json.loads(line), score_key)
            except json.JSONDecodeError:
                continue
    for line in reversed(lines):
        if _NUMBER.match(line):
            return coerce_result(float(line), score_key)
    matches = _SCORE_LINE.findall(text)
    if matches:
        return coerce_result(float(matches[-1]), score_key)
    return EvaluationResult.failure("no score found in the evaluator output", log=_tail(text))


def _tail(text: str, limit: int = 600) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


class Evaluator:
    """Base class: implement ``_evaluate(candidate_path, source)``."""

    name = "evaluator"

    def evaluate(self, candidate_path: Path, source: str) -> EvaluationResult:
        started = time.perf_counter()
        try:
            result = coerce_result(self._evaluate(Path(candidate_path), source))
        except Exception as exc:  # a broken candidate must not stop the run
            result = EvaluationResult.failure(f"{type(exc).__name__}: {exc}")
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    def _evaluate(self, candidate_path: Path, source: str) -> Any:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}


class SubprocessEvaluator(Evaluator):
    """Runs a benchmark command per candidate and parses the score it prints.

    ``command`` is an argument list (or a string split with POSIX shell rules; no shell runs).
    Placeholders: ``{candidate}`` (the candidate file), ``{candidate_dir}``, ``{project}`` (the
    working directory) and ``{python}`` (this interpreter). The candidate path is also in the
    ``OMNIEVOLVE_CANDIDATE`` environment variable, and ``{project}`` is prepended to
    ``PYTHONPATH`` so the candidate can import its sibling modules. On timeout the whole process
    group is killed.
    """

    name = "subprocess"

    def __init__(
        self,
        command: Sequence[str] | str,
        cwd: Path | str | None = None,
        timeout_seconds: float = 60.0,
        score_key: str = "score",
        env: dict[str, str] | None = None,
    ):
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        if not self.command:
            raise ValueError("evaluator.command is empty")
        self.cwd = Path(cwd).resolve() if cwd else Path.cwd()
        self.timeout_seconds = float(timeout_seconds)
        self.score_key = score_key
        self.env = {str(k): str(v) for k, v in (env or {}).items()}

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "command": self.command,
            "timeout_seconds": self.timeout_seconds,
            "score_key": self.score_key,
            "env": sorted(self.env),
        }

    def _argv(self, candidate_path: Path) -> list[str]:
        values = {
            "candidate": str(candidate_path),
            "candidate_dir": str(candidate_path.parent),
            "project": str(self.cwd),
            "python": sys.executable,
        }
        argv = []
        for part in self.command:
            for key, value in values.items():
                part = part.replace("{" + key + "}", value)
            argv.append(part)
        return argv

    def _evaluate(self, candidate_path: Path, source: str) -> EvaluationResult:
        env = dict(os.environ)
        env.update(self.env)
        env["OMNIEVOLVE_CANDIDATE"] = str(candidate_path)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(self.cwd), env.get("PYTHONPATH", "")) if p
        )
        posix = os.name == "posix"
        try:
            process = subprocess.Popen(
                self._argv(candidate_path),
                cwd=self.cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=posix,
            )
        except OSError as exc:
            return EvaluationResult.failure(f"cannot start the evaluator: {exc}")
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            if posix:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    process.kill()
            else:
                process.kill()
            stdout, stderr = process.communicate()
            return EvaluationResult.failure(
                f"timed out after {self.timeout_seconds:g}s", log=_tail(stderr or stdout or "")
            )
        if process.returncode != 0:
            detail = _tail(stderr or stdout or "", 400)
            return EvaluationResult.failure(f"exit status {process.returncode}: {detail}")
        result = parse_score_output(stdout, self.score_key)
        if not result.ok and stderr:
            result.log = _tail(stderr)
        return result


class PythonCallableEvaluator(Evaluator):
    """Calls ``fn(candidate_path)`` or ``fn(candidate_path, source)`` in this process.

    Fast, but the candidate runs with your interpreter's privileges and no timeout: use
    ``SubprocessEvaluator`` for untrusted or long-running code.
    """

    name = "python"

    def __init__(self, fn: Callable[..., Any], score_key: str = "score", label: str = ""):
        self.fn = fn
        self.score_key = score_key
        self.label = label or getattr(fn, "__qualname__", "callable")
        try:
            params = inspect.signature(fn).parameters.values()
            positional = [
                p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            self._arity = 2 if len(positional) >= 2 else 1
        except (TypeError, ValueError):
            self._arity = 1

    @classmethod
    def from_reference(cls, reference: str, root: Path, score_key: str = "score"):
        """``"evaluate.py:score"`` (a file relative to ``root``) or ``"package.module:score"``."""
        target, _, attr = reference.partition(":")
        if not attr:
            raise ValueError("evaluator.callable must look like 'evaluate.py:function'")
        if target.endswith(".py"):
            path = (root / target).resolve()
            spec = importlib.util.spec_from_file_location(f"omnievolve_eval_{path.stem}", path)
            if spec is None or spec.loader is None:
                raise ValueError(f"cannot load {target}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(target)
        fn = getattr(module, attr)
        return cls(fn, score_key=score_key, label=reference)

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, "callable": self.label, "score_key": self.score_key}

    def _evaluate(self, candidate_path: Path, source: str) -> Any:
        value = self.fn(candidate_path, source) if self._arity == 2 else self.fn(candidate_path)
        return coerce_result(value, self.score_key)


JUDGE_SYSTEM = (
    "You are a strict, consistent evaluator. Grade the candidate only against the rubric. "
    "Reply with a single JSON object and nothing else: "
    '{{"score": <number from 0 to {scale}>, "rationale": "<one short sentence>"}}'
)


class LLMJudgeEvaluator(Evaluator):
    """Grades a candidate with a language model against a rubric; the score is in [0, 1].

    ``samples`` > 1 asks several times and averages; ``judge_spread`` (the standard deviation
    across samples) is reported as a metric so noisy judgments are visible. Position-bias and
    swap-order calibration across judges are OmniSI Cloud features.
    """

    name = "llm_judge"

    def __init__(
        self,
        client: LLMClient,
        rubric: str,
        scale: float = 10.0,
        samples: int = 1,
        max_source_chars: int = 24_000,
    ):
        if not rubric.strip():
            raise ValueError("an LLM judge needs a rubric")
        self.client = client
        self.rubric = rubric.strip()
        self.scale = float(scale)
        self.samples = max(1, int(samples))
        self.max_source_chars = max_source_chars

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.name,
            **self.client.config.describe(),
            "scale": self.scale,
            "samples": self.samples,
            "rubric_chars": len(self.rubric),
        }

    def _evaluate(self, candidate_path: Path, source: str) -> EvaluationResult:
        system = JUDGE_SYSTEM.format(scale=f"{self.scale:g}")
        clipped = source[: self.max_source_chars]
        user = f"Rubric:\n{self.rubric}\n\nCandidate ({candidate_path.name}):\n```\n{clipped}\n```"
        grades = []
        for _ in range(self.samples):
            try:
                reply = self.client.complete(system, user)
            except LLMError as exc:
                return EvaluationResult.failure(f"judge call failed: {exc}")
            grade = self.parse_grade(reply)
            if grade is None:
                return EvaluationResult.failure("the judge reply had no numeric score")
            grades.append(min(max(grade, 0.0), self.scale))
        mean = sum(grades) / len(grades)
        metrics = {"judge_raw": mean}
        if len(grades) > 1:
            metrics["judge_spread"] = statistics.pstdev(grades)
        return EvaluationResult(score=mean / self.scale if self.scale else mean, metrics=metrics)

    @staticmethod
    def parse_grade(reply: str) -> float | None:
        match = _JSON_OBJECT.search(reply)
        if match:
            try:
                value = _numeric(json.loads(match.group(0)).get("score"))
                if value is not None:
                    return value
            except (json.JSONDecodeError, AttributeError):
                pass
        found = _SCORE_LINE.search(reply)
        return float(found.group(1)) if found else None


class CompositeEvaluator(Evaluator):
    """A weighted mean of several evaluators; each part's score and metrics become metrics.

    A failed required part fails the candidate. Local runs optimize this single weighted score;
    OmniSI Cloud keeps the objectives separate and selects on the Pareto frontier.
    """

    name = "composite"

    def __init__(self, parts: Sequence[tuple[str, Evaluator, float]], required: bool = True):
        if not parts:
            raise ValueError("a composite evaluator needs at least one part")
        self.parts = [(name, evaluator, float(weight)) for name, evaluator, weight in parts]
        if sum(weight for _, _, weight in self.parts) <= 0:
            raise ValueError("composite weights must add up to more than zero")
        self.required = required

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "parts": [
                {"name": name, "weight": weight, **evaluator.describe()}
                for name, evaluator, weight in self.parts
            ],
        }

    def _evaluate(self, candidate_path: Path, source: str) -> EvaluationResult:
        total, weights, metrics, errors = 0.0, 0.0, {}, []
        for name, evaluator, weight in self.parts:
            result = evaluator.evaluate(candidate_path, source)
            for key, value in result.metrics.items():
                metrics[f"{name}.{key}"] = value
            if not result.ok or result.score is None:
                errors.append(f"{name}: {result.error}")
                continue
            metrics[f"{name}.score"] = result.score
            total += weight * result.score
            weights += weight
        if errors and (self.required or not weights):
            return EvaluationResult(score=None, metrics=metrics, ok=False, error="; ".join(errors))
        return EvaluationResult(score=total / weights, metrics=metrics)


# -- factory ------------------------------------------------------------------------------------

SUBPROCESS_KEYS = {"type", "command", "timeout_seconds", "score_key", "env", "cwd"}


def build_evaluator(spec: dict[str, Any], root: Path) -> Evaluator:
    """Builds an evaluator from the config's ``evaluator`` section."""
    spec = dict(spec or {})
    kind = spec.get("type", "subprocess")
    if kind == "subprocess":
        unknown = sorted(set(spec) - SUBPROCESS_KEYS)
        if unknown:
            raise ValueError(f"unknown evaluator setting(s): {', '.join(unknown)}")
        if "command" not in spec:
            raise ValueError("a subprocess evaluator needs a command")
        return SubprocessEvaluator(
            spec["command"],
            cwd=(root / spec.get("cwd", ".")).resolve(),
            timeout_seconds=spec.get("timeout_seconds", 60),
            score_key=spec.get("score_key", "score"),
            env=spec.get("env"),
        )
    if kind == "python":
        if "callable" not in spec:
            raise ValueError("a python evaluator needs a callable such as 'evaluate.py:score'")
        return PythonCallableEvaluator.from_reference(
            spec["callable"], root, score_key=spec.get("score_key", "score")
        )
    if kind == "llm_judge":
        llm = {k: v for k, v in spec.items() if k in LLMConfig.__dataclass_fields__}
        rubric = spec.get("rubric", "")
        if isinstance(rubric, str) and rubric.startswith("@"):
            rubric = (root / rubric[1:]).read_text(encoding="utf-8")
        return LLMJudgeEvaluator(
            LLMClient(LLMConfig.from_dict(llm)),
            rubric=rubric,
            scale=spec.get("scale", 10),
            samples=spec.get("samples", 1),
        )
    if kind == "composite":
        parts = []
        for index, part in enumerate(spec.get("parts", [])):
            part = dict(part)
            name = part.pop("name", f"part{index}")
            weight = part.pop("weight", 1.0)
            parts.append((name, build_evaluator(part, root), weight))
        return CompositeEvaluator(parts, required=spec.get("required", True))
    raise ValueError(
        f"unknown evaluator type {kind!r} (use subprocess, python, llm_judge or composite)"
    )
