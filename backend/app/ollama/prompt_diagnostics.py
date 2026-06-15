"""Prompt component diagnostics for structured completions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.ollama.usage import RunUsage

PromptChannel = Literal["system", "prompt"]


@dataclass(frozen=True)
class PromptComponent:
    """Named prompt text supplied by a structured completion caller."""

    name: str
    text: str
    channel: PromptChannel = "prompt"


class PromptComponentDiagnostic(BaseModel):
    channel: PromptChannel
    name: str
    text: str
    chars: int
    tokens: int
    cumulative_tokens: int


class PromptAttemptDiagnostic(BaseModel):
    attempt_index: int
    attempt_kind: str
    model: str
    tokenizer_fallback: bool = False
    tokenizer_fallback_reason: str | None = None
    components: list[PromptComponentDiagnostic] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)


class PromptCompletionDiagnostics(BaseModel):
    operation_id: str = ""
    agent_name: str = ""
    model: str
    output_type: str
    status: str = "running"
    metadata: dict[str, Any] = Field(default_factory=dict)
    attempts: list[PromptAttemptDiagnostic] = Field(default_factory=list)


def run_usage_to_dict(usage: RunUsage | None) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "prompt_eval_duration_ms": usage.prompt_eval_duration_ms,
        "load_duration_ms": usage.load_duration_ms,
        "response_duration_ms": usage.response_duration_ms,
        "total_duration_ms": usage.total_duration_ms,
        **usage.details,
    }
