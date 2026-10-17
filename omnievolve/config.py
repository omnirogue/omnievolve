# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""Project configuration (``omnievolve.json``; YAML when PyYAML is installed).

Paths in the file are relative to the file's own directory. Example::

    {
      "name": "my-heuristic",
      "target": "heuristic.py",
      "evaluator": {"type": "subprocess", "command": ["{python}", "evaluate.py", "{candidate}"]},
      "mutator": {"type": "deterministic"},
      "search": {"generations": 8, "islands": 2, "population": 6, "seed": 7},
      "ledger": ".omnievolve/ledger.db"
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine import SearchConfig

DEFAULT_LEDGER = ".omnievolve/ledger.db"
TOP_LEVEL_KEYS = {
    "name",
    "target",
    "evaluator",
    "mutator",
    "search",
    "objective",
    "ledger",
    "report",
    "cloud",
    "description",
}


class ConfigError(ValueError):
    """The project configuration is missing or invalid."""


@dataclass
class ProjectConfig:
    path: Path
    root: Path
    name: str
    target: Path
    evaluator: dict[str, Any]
    mutator: dict[str, Any]
    search: SearchConfig
    ledger: Path
    report: Path | None = None
    cloud: dict[str, Any] = field(default_factory=dict)

    @property
    def target_label(self) -> str:
        """The target path relative to the project root, as patches and the ledger show it."""
        try:
            return self.target.relative_to(self.root).as_posix()
        except ValueError:
            return self.target.name

    def run_config(self) -> dict[str, Any]:
        """The settings stored with each run in the ledger (no secrets: keys are env names)."""
        return {
            "name": self.name,
            "target": self.target_label,
            "evaluator": self.evaluator,
            "mutator": self.mutator,
            "config_file": self.path.name,
        }


def _read(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ConfigError(
                "YAML configs need PyYAML: pip install 'omnievolve[yaml]', or use JSON"
            ) from None
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path.name}: invalid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: the top level must be an object")
    return data


def load_config(path: Path | str, overrides: dict[str, Any] | None = None) -> ProjectConfig:
    """Loads and validates a project file. ``overrides`` replace ``search`` settings."""
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "omnievolve.json"
    if not path.is_file():
        raise ConfigError(f"no config at {path} (create one with `omnievolve init`)")
    path = path.resolve()
    root = path.parent
    data = _read(path)
    unknown = sorted(set(data) - TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"unknown setting(s): {', '.join(unknown)}")
    if not data.get("target"):
        raise ConfigError("'target' (the file to evolve) is required")
    target = (root / data["target"]).resolve()
    if not target.is_file():
        raise ConfigError(f"target {data['target']!r} not found next to {path.name}")
    if not isinstance(data.get("evaluator"), dict):
        raise ConfigError('\'evaluator\' is required, for example {"type": "subprocess", ...}')
    search = dict(data.get("search") or {})
    if "objective" in data:
        search.setdefault("objective", data["objective"])
    for key, value in (overrides or {}).items():
        if value is not None:
            search[key] = value
    try:
        search_config = SearchConfig.from_dict(search)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from None
    ledger = root / data.get("ledger", DEFAULT_LEDGER)
    report = root / data["report"] if data.get("report") else None
    cloud = data.get("cloud") or {}
    if not isinstance(cloud, dict):
        raise ConfigError("'cloud' must be an object")
    return ProjectConfig(
        path=path,
        root=root,
        name=str(data.get("name") or root.name),
        target=target,
        evaluator=dict(data["evaluator"]),
        mutator=dict(data.get("mutator") or {"type": "deterministic"}),
        search=search_config,
        ledger=ledger,
        report=report,
        cloud=dict(cloud),
    )
