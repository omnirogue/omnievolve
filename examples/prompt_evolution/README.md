# Prompt evolution: a customer-support system prompt

`system_prompt.txt` is a three-line system prompt for a support assistant. OmniEvolve's
deterministic text mutator adds, removes, replaces and swaps lines drawn from
`mutation_pool.txt`, which mixes useful instructions with distractors ("use plenty of emojis",
"always answer in French", "be as detailed as possible").

`evaluate.py` runs a synthetic test suite of ten support requests: a password reset, a request
for another customer's credentials, an ambiguous "it doesn't work", a billing dispute, and so on.
A simulated assistant follows the instructions it recognizes in the prompt, and each reply is
checked against the house style: under 60 words, addresses the customer by name, cites the
help-center article, asks when the request is ambiguous, refuses credential requests, escalates
billing disputes, uses bullets for steps, and apologizes only for problems we caused. The score
is the mean pass rate, minus a small penalty for prompts over 60 words.

```bash
# from the repository root
omnievolve run --config examples/prompt_evolution/omnievolve.json --html prompt-report.html
omnievolve export-winner --ledger examples/prompt_evolution/.omnievolve/ledger.db --output prompt.patch
```

The simulated assistant keeps the example offline and deterministic. To optimize a prompt for a
real model, replace `respond()` in `evaluate.py` with a call to that model, or use an LLM judge:

```json
"evaluator": {"type": "llm_judge", "provider": "openai", "model": "<model id>",
              "rubric": "@rubric.md", "samples": 3}
```

and let a model propose the rewrites:

```json
"mutator": {"type": "llm", "provider": "anthropic", "model": "claude-sonnet-5-5",
            "fallback": {"line_pool": "mutation_pool.txt", "numeric_rate": 0.0}}
```
