# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""OmniEvolve: local-first evolutionary optimization for code and prompts.

The engine evolves one target file (a Python module, a prompt template, a JSON config) through
an island-model population. Evaluators score candidates, mutators propose new ones, and every
candidate, score and parent-to-child diff lands in a local SQLite lineage ledger.
"""

from ._version import __version__
from .engine import (
    Candidate,
    EvolutionEngine,
    EvolutionError,
    Island,
    Population,
    RunResult,
    SearchConfig,
)
from .evaluators import (
    CompositeEvaluator,
    EvaluationResult,
    Evaluator,
    LLMJudgeEvaluator,
    PythonCallableEvaluator,
    SubprocessEvaluator,
)
from .ledger import Ledger
from .llm import LLMClient, LLMConfig, LLMError
from .mutators import DeterministicMutator, LLMMutator, Mutation, MutationContext, Mutator

__all__ = [
    "__version__",
    "Candidate",
    "CompositeEvaluator",
    "DeterministicMutator",
    "EvaluationResult",
    "Evaluator",
    "EvolutionEngine",
    "EvolutionError",
    "Island",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "LLMJudgeEvaluator",
    "LLMMutator",
    "Ledger",
    "Mutation",
    "MutationContext",
    "Mutator",
    "Population",
    "PythonCallableEvaluator",
    "RunResult",
    "SearchConfig",
    "SubprocessEvaluator",
]
