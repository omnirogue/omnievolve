# Heuristic tuning: a support-ticket triage rule

`triage.py` flags a ticket as urgent when a weighted sum of five features passes a threshold.
The six constants between the `EVOLVE-BLOCK` markers start out poorly chosen. OmniEvolve's
deterministic AST mutator perturbs them, and `evaluate.py` scores each candidate by F1 on 600
synthetic tickets whose true urgency follows a hidden rule plus noise.

Everything is offline and deterministic: no API key, no network, and the same seed gives the
same run.

```bash
# from the repository root
omnievolve run --config examples/heuristic_tuning/omnievolve.json --generations 5 --html report.html
omnievolve leaderboard --ledger examples/heuristic_tuning/.omnievolve/ledger.db
omnievolve export-winner --ledger examples/heuristic_tuning/.omnievolve/ledger.db --output best.patch
```

Score a file by hand:

```bash
cd examples/heuristic_tuning
python3 evaluate.py triage.py
# {"score": 0.553571, "precision": ..., "recall": ..., "accuracy": ...}
```

| File | Role |
| --- | --- |
| `triage.py` | The target. Only the constants between the markers change. |
| `evaluate.py` | The benchmark: prints `{"score": <F1>, ...}` as JSON. |
| `omnievolve.json` | Search settings: 2 islands of 6, 8 generations, 4 parallel workers. |

The best achievable F1 on this data is about 0.82 (the hidden rule itself, given the label
noise). To try a model instead of the offline mutator:

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, GEMINI_API_KEY; Ollama needs no key
omnievolve run --config examples/heuristic_tuning/omnievolve.json --mutator llm --provider anthropic --model claude-sonnet-5-5
```
