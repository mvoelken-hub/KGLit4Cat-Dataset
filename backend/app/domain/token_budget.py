from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


ESTIMATED_CHARS_PER_TOKEN = 4
DEFAULT_TRUNCATION_MARKER = "\n[orientation truncated]"


@dataclass(frozen=True)
class PromptTokenBudgeter:
    tokenizer: Any | None = None
    fallback_reason: str | None = None
    truncation_marker: str = DEFAULT_TRUNCATION_MARKER

    @classmethod
    def from_tokenizer_source(
        cls,
        tokenizer_source: str | None,
        *,
        hf_token: str | None = None,
    ) -> PromptTokenBudgeter:
        source = (tokenizer_source or "").strip()
        if not source:
            return cls(
                tokenizer=None,
                fallback_reason="No Hugging Face tokenizer configured; using conservative token estimates.",
            )
        tokenizer, fallback_reason = _load_tokenizer(source, (hf_token or "").strip() or None)
        return cls(tokenizer=tokenizer, fallback_reason=fallback_reason)

    @property
    def uses_fallback(self) -> bool:
        return self.tokenizer is None

    def count(self, value: str) -> int:
        if not value:
            return 0
        if self.tokenizer is None:
            return estimated_tokens(value)
        return len(encoding_ids(self.tokenizer.encode(value, add_special_tokens=False)))

    def truncate(self, value: str, *, max_tokens: int) -> str:
        if not value or max_tokens <= 0:
            return ""
        if self.count(value) <= max_tokens:
            return value
        if self.tokenizer is None:
            return truncate_by_estimated_tokens(
                value,
                max_tokens=max_tokens,
                marker=self.truncation_marker,
            )
        return truncate_with_token_offsets(
            value,
            max_tokens=max_tokens,
            tokenizer=self.tokenizer,
            marker=self.truncation_marker,
        )


def estimated_tokens(value: str) -> int:
    return math.ceil(len(value) / ESTIMATED_CHARS_PER_TOKEN)


@lru_cache(maxsize=8)
def _load_tokenizer(source: str, hf_token: str | None) -> tuple[Any | None, str | None]:
    try:
        path = Path(source)
        if path.exists():
            tokenizer = Tokenizer.from_file(str(path))
        else:
            tokenizer = Tokenizer.from_pretrained(source, token=hf_token)
    except Exception as exc:  # pragma: no cover - exact exception types depend on tokenizers/hub internals.
        return (
            None,
            (
                f"Could not load Hugging Face tokenizer '{source}'; "
                f"using conservative token estimates ({type(exc).__name__}: {exc})."
            ),
        )
    return tokenizer, None


def truncate_by_estimated_tokens(
    value: str,
    *,
    max_tokens: int,
    marker: str = DEFAULT_TRUNCATION_MARKER,
) -> str:
    marker_tokens = estimated_tokens(marker)
    body_budget = max(0, max_tokens - marker_tokens)
    if body_budget <= 0:
        return marker.strip() if marker_tokens <= max_tokens else ""
    low = 0
    high = len(value)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        candidate = value[:mid].rstrip() + marker
        if estimated_tokens(candidate) <= max_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return value[:best].rstrip() + marker


def truncate_with_token_offsets(
    value: str,
    *,
    max_tokens: int,
    tokenizer: Any,
    marker: str = DEFAULT_TRUNCATION_MARKER,
) -> str:
    marker_tokens = len(encoding_ids(tokenizer.encode(marker, add_special_tokens=False)))
    body_budget = max(0, max_tokens - marker_tokens)
    if body_budget <= 0:
        return marker.strip() if marker_tokens <= max_tokens else ""
    encoding = tokenizer.encode(value, add_special_tokens=False)
    ids = encoding_ids(encoding)
    if len(ids) <= max_tokens:
        return value
    offsets = list(getattr(encoding, "offsets", []) or [])
    if len(offsets) >= body_budget:
        end_offset = offsets[body_budget - 1][1]
        if end_offset:
            return value[:end_offset].rstrip() + marker
    return truncate_by_estimated_tokens(value, max_tokens=max_tokens, marker=marker)


def encoding_ids(encoding: Any) -> list[Any]:
    ids = getattr(encoding, "ids", None)
    return list(ids or [])
