#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""Release hygiene for this repository; exits non-zero on any finding.

Checks: the LICENSE file is the canonical Apache-2.0 text; every source file carries an
``SPDX-License-Identifier: Apache-2.0`` header in its first five lines; the package version
agrees across files; no Rust sources or Cargo manifests (this package is Python only); no
run artifacts (ledgers, reports, patches); and no credential-shaped strings. Findings are
reported by file and line, never by value.

    python3 scripts/release_check.py
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
HEADER_SUFFIXES = {".py", ".toml", ".yml", ".yaml", ".sh"}
FORBIDDEN_NAMES = re.compile(r"(\.rs|Cargo\.toml|Cargo\.lock|\.db|\.sqlite3?|\.patch|\.pem|\.key)$")
FORBIDDEN_DIRS = {".omnievolve", "target"}
SECRET = re.compile(
    r"(^|[^A-Za-z0-9_-])sk-(ant-)?[A-Za-z0-9_-]{20,}"
    r"|(AKIA|ASIA)[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|AIza[0-9A-Za-z_-]{35}"
)


def tracked_files() -> list[Path]:
    """Files git tracks (or would add); every file when this is not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if out.strip():
            return sorted(ROOT / line for line in out.splitlines() if (ROOT / line).is_file())
    except (OSError, subprocess.CalledProcessError):
        pass
    skip = {".git", "__pycache__", ".venv", "build", "dist", ".pytest_cache", ".ruff_cache"}
    return sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file() and not skip.intersection(p.relative_to(ROOT).parts)
    )


def check() -> list[str]:
    findings = []
    license_file = ROOT / "LICENSE"
    if not license_file.is_file():
        findings.append("LICENSE is missing")
    elif hashlib.sha256(license_file.read_bytes()).hexdigest() != APACHE_SHA256:
        findings.append("LICENSE is not the canonical Apache-2.0 text")
    version = re.search(r'__version__ = "([^"]+)"', (ROOT / "omnievolve/_version.py").read_text())
    project = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)
    if not version or not project or version.group(1) != project.group(1):
        findings.append("omnievolve/_version.py and pyproject.toml disagree on the version")
    for path in tracked_files():
        rel = path.relative_to(ROOT).as_posix()
        if FORBIDDEN_NAMES.search(path.name) or FORBIDDEN_DIRS.intersection(Path(rel).parts):
            findings.append(f"{rel}: not allowed in the repository")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{rel}: binary file")
            continue
        if path.suffix in HEADER_SUFFIXES and path.name != "py.typed":
            head = "\n".join(text.splitlines()[:5])
            if "SPDX-License-Identifier: Apache-2.0" not in head:
                findings.append(f"{rel}: missing its Apache-2.0 SPDX license header")
        for number, line in enumerate(text.splitlines(), 1):
            if SECRET.search(line):
                findings.append(f"{rel}:{number}: credential-shaped string (value withheld)")
    return findings


def main() -> int:
    findings = check()
    for finding in findings:
        print(f"FAIL {finding}")
    if findings:
        print(f"release_check: {len(findings)} finding(s)")
        return 1
    print("release_check: OK (Apache-2.0 headers, Python only, no artifacts or credentials)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
