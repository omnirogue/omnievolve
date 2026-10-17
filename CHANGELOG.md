# Changelog

## 0.1.0 (2026-10-16)

First open-source release.

- Island-model evolutionary engine: tournament selection, elitism, generational or plus
  survivor selection, crossover, ring migration, evaluation cache, patience and target-score
  stopping, seeded and reproducible with parallel evaluation.
- Evaluator SDK: `SubprocessEvaluator`, `PythonCallableEvaluator`, `LLMJudgeEvaluator`,
  `CompositeEvaluator` and the `Evaluator` base class.
- Mutators: an offline deterministic mutator (AST numeric parameters, text lines from a pool,
  parameter and line crossover) and a bring-your-own-key LLM mutator for OpenAI, Anthropic,
  Gemini and Ollama with deterministic fallback.
- SQLite lineage ledger, self-contained dark-mode HTML report, winner patch export.
- CLI (`omnievolve init | run | runs | leaderboard | export-winner | report`), `python -m
  omnievolve`, and a stdio MCP server (`omnievolve-mcp`).
- Optional OmniSI Cloud bridge (`omnievolve run --cloud`, with `--dry-run`).
- Examples: heuristic tuning and prompt evolution.
