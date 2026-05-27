"""Token usage tracking for structured completions."""

from __future__ import annotations

from dataclasses import dataclass, field

import ollama


@dataclass
class RunUsage:
    """Aggregated token usage across one or more completion attempts."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    response_duration_ms: int = 0
    total_duration_ms: int = 0
    details: dict[str, int] = field(default_factory=dict)

    def merge(self, other: RunUsage) -> RunUsage:
        """Return a new RunUsage combining self + other."""
        merged_details: dict[str, int] = {}
        for d in (self.details, other.details):
            for key, value in d.items():
                merged_details[key] = merged_details.get(key, 0) + value
        return RunUsage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            response_duration_ms=self.response_duration_ms + other.response_duration_ms,
            total_duration_ms=self.total_duration_ms + other.total_duration_ms,
            details=merged_details,
        )

    def __add__(self, other: RunUsage) -> RunUsage:
        return self.merge(other)

    @classmethod
    def from_ollama_response(cls, response: ollama.GenerateResponse) -> RunUsage:
        """Create RunUsage from an ollama GenerateResponse object."""
        return cls(
            requests=1,
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            response_duration_ms=_nanoseconds_to_milliseconds(
                getattr(response, "eval_duration", 0) or 0
            ),
            total_duration_ms=_nanoseconds_to_milliseconds(
                getattr(response, "total_duration", 0) or 0
            ),
        )


def _nanoseconds_to_milliseconds(value: int | float | None) -> int:
    if not value:
        return 0
    try:
        return round(float(value) / 1_000_000)
    except (TypeError, ValueError):
        return 0
