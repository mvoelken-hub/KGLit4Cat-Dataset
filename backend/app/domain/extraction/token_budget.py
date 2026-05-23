from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


DEFAULT_MAX_CONTEXT_LENGTH = 8192
DEFAULT_INPUT_TARGET_RATIO = 0.75
FALLBACK_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class TokenBudget:
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH
    input_target_ratio: float = DEFAULT_INPUT_TARGET_RATIO
    chars_per_token: int = FALLBACK_CHARS_PER_TOKEN

    @property
    def input_token_budget(self) -> int:
        return max(1, int(self.max_context_length * self.input_target_ratio))

    def estimate_text_tokens(self, value: str) -> int:
        return estimate_text_tokens(value, chars_per_token=self.chars_per_token)

    def metadata(
        self,
        *,
        estimated_input_tokens: int,
        split_count: int = 0,
        compaction_count: int = 0,
    ) -> dict[str, int]:
        return {
            "estimated_input_tokens": max(0, int(estimated_input_tokens)),
            "input_token_budget": self.input_token_budget,
            "max_context_length": self.max_context_length,
            "split_count": max(0, int(split_count)),
            "compaction_count": max(0, int(compaction_count)),
        }


class BudgetedUsage:
    """Usage proxy that carries preflight budget metadata with model usage."""

    def __init__(self, usage: Any, budget_metadata: dict[str, int]):
        self._usage = usage
        self.budget_metadata = budget_metadata
        for key, value in budget_metadata.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._usage, name)


def estimate_text_tokens(
    value: str,
    *,
    chars_per_token: int = FALLBACK_CHARS_PER_TOKEN,
) -> int:
    if not value:
        return 0
    return max(1, math.ceil(len(value) / max(1, chars_per_token)))


def budget_from_context_length(max_context_length: int | None) -> TokenBudget:
    return TokenBudget(max_context_length=max_context_length or DEFAULT_MAX_CONTEXT_LENGTH)
