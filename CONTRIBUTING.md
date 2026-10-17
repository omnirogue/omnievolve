# Contributing to OmniEvolve

Thank you for helping. Bug reports, examples, evaluators and documentation fixes are all
welcome.

## Ground rules

- **Standard library only** in `omnievolve/`. Optional integrations go behind an extra in
  `pyproject.toml` and must import lazily.
- **Deterministic by default.** Anything that changes the search outcome must draw from the
  seeded random streams the engine passes in, so runs stay reproducible.
- **No secrets.** Read keys from environment variables by name; never log, store or echo them.
- **License.** Contributions are accepted under Apache-2.0. Every source file starts with
  `# SPDX-License-Identifier: Apache-2.0`.

## Workflow

```bash
git clone https://github.com/omnirogue/omnievolve && cd omnievolve
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'

python3 -m unittest discover -s tests     # or: pytest
python3 scripts/release_check.py
ruff check . && ruff format --check .
```

1. Open an issue first for larger changes so we can agree on the approach.
2. Keep pull requests focused, with tests for new behavior.
3. Sign off your commits (`git commit -s`) to certify the
   [Developer Certificate of Origin](https://developercertificate.org/), for example
   `Signed-off-by: Your Name <you@example.com>`.
4. CI must pass on every supported Python version.

## Adding an evaluator or mutator

Subclass `omnievolve.Evaluator` (implement `_evaluate`) or `omnievolve.Mutator` (implement
`mutate`, optionally `crossover`), add a `describe()` that returns only non-secret settings, and
register a config `type` in the matching `build_*` factory if it should be usable from
`omnievolve.json`.
