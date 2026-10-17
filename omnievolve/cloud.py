# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""Optional bridge to OmniSI Cloud (``omnievolve run --cloud``).

The local engine is complete on its own. OmniSI Cloud (powered by the proprietary
OmniRogue Rust Engine) runs the same project as a hosted multi-objective search: high-concurrency
Rust tournaments, Pareto-frontier selection across objectives, a novelty archive, swap-order
calibrated LLM judging, SHA-256 audit bundles, and evaluation on sandboxed fleet workers.
None of that engine ships in this package; this module only builds and submits the run.

Nothing leaves your machine unless you pass ``--cloud`` without ``--dry-run``. The payload is
the target file, the evaluator and search settings, and the project's text files (secret-bearing
names such as ``.env`` and ``*.pem`` are always skipped). ``--dry-run`` prints exactly what would
be sent. The endpoint and key come from ``OMNISI_CLOUD_URL`` and ``OMNISI_API_KEY``.
"""

from __future__ import annotations

import fnmatch
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._version import __version__
from .config import ProjectConfig

URL_ENV = "OMNISI_CLOUD_URL"
KEY_ENV = "OMNISI_API_KEY"
API_PREFIX = "/api/evolve"
MAX_SPEC_BYTES = 256 * 1024
MAX_FILE_BYTES = 64 * 1024
SKIP_DIRS = {".git", ".omnievolve", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
SECRET_NAMES = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "credentials*.json",
    ".npmrc",
    ".pypirc",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
)
CLOUD_FEATURES = {
    "selection": "pareto_frontier",
    "novelty_archive": True,
    "judge_calibration": "swap_order",
    "early_stopping": "sequential_dominance",
    "audit_bundle": "sha256",
    "workers": "sandboxed_fleet",
}

# transport(method, url, headers, payload or None, timeout) -> parsed JSON
Transport = Callable[[str, str, dict[str, str], Any, float], Any]


class CloudError(RuntimeError):
    """The cloud bridge cannot build or submit the run. Messages never contain the key."""


def _skipped(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_NAMES)


def project_files(root: Path, target: Path) -> tuple[dict[str, str], list[str]]:
    """Text files under ``root`` to send with the run, and the paths that were skipped."""
    files: dict[str, str] = {}
    skipped: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if path.resolve() == target.resolve():
                continue
            if _skipped(name) or path.is_symlink():
                skipped.append(rel)
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                skipped.append(rel)
                continue
            try:
                files[rel] = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                skipped.append(rel)
    return files, skipped


def build_payload(project: ProjectConfig, local_run_id: str | None = None) -> dict[str, Any]:
    """The request body for ``POST /api/evolve/runs``."""
    from .evaluators import build_evaluator
    from .mutators import build_mutator

    cfg = project.search
    files, skipped = project_files(project.root, project.target)
    objectives = project.cloud.get("objectives") or [{"name": "score", "direction": cfg.objective}]
    spec = {
        "client": {"name": "omnievolve", "version": __version__},
        "name": project.name,
        "target": {
            "path": project.target_label,
            "source": project.target.read_text(encoding="utf-8"),
        },
        "evaluator": build_evaluator(project.evaluator, project.root).describe(),
        "evaluator_config": project.evaluator,
        "mutator": build_mutator(project.mutator, project.root).describe(),
        "objectives": objectives,
        "search": cfg.to_dict(),
        "features": {**CLOUD_FEATURES, **project.cloud.get("features", {})},
        "files": files,
        "skipped_files": skipped,
    }
    body = {
        "spec": spec,
        "target": project.target_label,
        "generations": cfg.generations,
        "candidates_per_generation": cfg.islands * cfg.population,
        "usd_cap": project.cloud.get("usd_cap"),
        "execution_tier": project.cloud.get("execution_tier"),
        "run_id": local_run_id,
        "spec_path": project.path.name,
    }
    body = {k: v for k, v in body.items() if v is not None}
    size = len(json.dumps(spec).encode("utf-8"))
    if size > MAX_SPEC_BYTES:
        raise CloudError(
            f"the run spec is {size} bytes; the cloud accepts up to {MAX_SPEC_BYTES}. Move large "
            "files out of the project directory or into a skipped folder."
        )
    return body


def _http(method: str, url: str, headers: dict[str, str], payload: Any, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    host = urllib.parse.urlparse(url).netloc
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise CloudError(f"HTTP {exc.code} from {host}: {detail}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise CloudError(f"cannot reach {host}: {exc}") from None
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        raise CloudError(f"{host} returned a response that is not JSON") from None


class CloudClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: Transport | None = None,
        timeout_seconds: float = 30.0,
    ):
        base_url = base_url or os.environ.get(URL_ENV, "")
        api_key = api_key or os.environ.get(KEY_ENV, "")
        if not base_url:
            raise CloudError(f"set {URL_ENV} to your OmniSI Cloud endpoint")
        if not api_key:
            raise CloudError(f"set {KEY_ENV} to an OmniSI Cloud API key")
        parsed = urllib.parse.urlparse(base_url)
        local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise CloudError(f"{URL_ENV} must be an https URL")
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self.transport = transport or _http
        self.timeout_seconds = timeout_seconds

    def _call(self, method: str, path: str, payload: Any = None) -> Any:
        headers = {
            "authorization": f"Bearer {self._key}",
            "accept": "application/json",
            "user-agent": f"omnievolve/{__version__}",
        }
        if payload is not None:
            headers["content-type"] = "application/json"
        url = f"{self.base_url}{API_PREFIX}{path}"
        return self.transport(method, url, headers, payload, self.timeout_seconds)

    def submit(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/runs", body)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._call("GET", f"/runs/{urllib.parse.quote(run_id, safe='')}")

    def candidates(self, run_id: str) -> Any:
        return self._call("GET", f"/runs/{urllib.parse.quote(run_id, safe='')}/candidates")

    def wait(
        self,
        run_id: str,
        poll_seconds: float = 5.0,
        timeout_seconds: float = 3600.0,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            run = self.get_run(run_id)
            run = run.get("run", run) if isinstance(run, dict) else {}
            if on_update:
                on_update(run)
            if str(run.get("status", "")).lower() in {
                "completed",
                "succeeded",
                "failed",
                "cancelled",
                "canceled",
                "promoted",
                "rolled_back",
            }:
                return run
            if time.monotonic() >= deadline:
                raise CloudError(f"run {run_id} is still {run.get('status')!r} after waiting")
            sleep(poll_seconds)


def summarize_payload(body: dict[str, Any]) -> dict[str, Any]:
    """What ``--dry-run`` prints: the payload with file contents replaced by sizes."""
    spec = dict(body["spec"])
    spec["target"] = {
        "path": spec["target"]["path"],
        "bytes": len(spec["target"]["source"].encode("utf-8")),
    }
    spec["files"] = {name: len(text.encode("utf-8")) for name, text in spec["files"].items()}
    return {**body, "spec": spec, "endpoint": f"${URL_ENV}{API_PREFIX}/runs"}
