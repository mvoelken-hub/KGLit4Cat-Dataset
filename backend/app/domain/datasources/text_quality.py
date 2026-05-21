"""Text quality gate for filtering semantically meaningful lines.

This module is pure Python with no framework dependencies. It classifies
text lines as KEEP or DROP based on heuristic features.
"""

import math
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum


class DecisionKind(Enum):
    KEEP = "keep"
    DROP = "drop"


@dataclass
class TextQualityDecision:
    kind: DecisionKind
    score: float
    reason: str


@dataclass
class TextQualityConfig:
    """Tunable thresholds for the text-quality classifier.

    All defaults mirror the original hard-coded behaviour so that omitting
    the config does not change results.
    """
    symbol_ratio_threshold: float = 0.45
    digit_ratio_threshold: float = 0.45
    keep_score_threshold: float = 0.30
    structured_text_bonus: float = 0.70


WORD_RE = re.compile(r"[^\W\d_][^\W\d_'\-]{1,}", re.UNICODE)
LONG_DENSE_TOKEN_RE = re.compile(r"\S{80,}")
MOSTLY_HEX_RE = re.compile(r"^[0-9a-fA-F\s:,-]{40,}$")
MOSTLY_NUMERIC_RE = re.compile(
    r"^\s*[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
    r"(?:[\s,;]+[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?){5,}\s*$"
)
BASE64ISH_RE = re.compile(r"^[A-Za-z0-9+/=_-]{80,}$")
REPEATED_CHAR_RE = re.compile(r"(.)\1{20,}")


def classify_text_line(
    line: str,
    config: TextQualityConfig | None = None,
) -> TextQualityDecision:
    cfg = config or TextQualityConfig()
    s = line.strip()
    if not s:
        return TextQualityDecision(DecisionKind.DROP, 0.0, "empty")
    length = len(s)
    if length < 2:
        return TextQualityDecision(DecisionKind.DROP, 0.0, "too_short")
    if _has_many_control_chars(s):
        return TextQualityDecision(DecisionKind.DROP, 0.0, "many_control_chars")
    if REPEATED_CHAR_RE.search(s):
        return TextQualityDecision(DecisionKind.DROP, 0.05, "repeated_char_noise")
    if MOSTLY_NUMERIC_RE.match(s):
        return TextQualityDecision(DecisionKind.DROP, 0.05, "numeric_array")
    features = _features(s)
    score = 0.0
    reasons: list[str] = []
    # Positive Signals
    if features["word_count"] >= 3:
        score += 0.35
        reasons.append("has_words")
    if features["alpha_ratio"] >= 0.25:
        score += 0.20
        reasons.append("enough_letters")
    if features["space_ratio"] >= 0.08:
        score += 0.15
        reasons.append("has_spacing")
    if features["avg_word_len"] >= 3:
        score += 0.10
        reasons.append("reasonable_word_length")
    if _looks_like_structured_text(s):
        score += cfg.structured_text_bonus
        reasons.append("structured_text")
    # Negative Signals
    if features["digit_ratio"] >= cfg.digit_ratio_threshold:
        score -= 0.25
        reasons.append("too_many_digits")
    if features["symbol_ratio"] >= cfg.symbol_ratio_threshold:
        score -= 0.25
        reasons.append("too_many_symbols")
    if features["space_ratio"] < 0.02 and length > 60:
        score -= 0.25
        reasons.append("dense_no_spaces")
    if LONG_DENSE_TOKEN_RE.search(s):
        score -= 0.30
        reasons.append("long_dense_token")
    if MOSTLY_HEX_RE.match(s):
        score -= 0.35
        reasons.append("hex_like")
    if BASE64ISH_RE.match(s) and features["space_ratio"] == 0:
        score -= 0.30
        reasons.append("base64_like")
    if features["entropy"] >= 4.5 and features["space_ratio"] < 0.05:
        score -= 0.20
        reasons.append("high_entropy_dense_text")
    score = max(0.0, min(1.0, score))
    if score >= cfg.keep_score_threshold:
        return TextQualityDecision(DecisionKind.KEEP, score, ", ".join(reasons))
    return TextQualityDecision(DecisionKind.DROP, score, ", ".join(reasons))

def _features(s: str) -> dict[str, float]:
    n = len(s)
    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    symbols = sum(
        not ch.isalnum() and not ch.isspace()
        for ch in s
    )
    words = WORD_RE.findall(s)
    word_lengths = [len(w) for w in words]
    return {
        "alpha_ratio": letters / n,
        "digit_ratio": digits / n,
        "space_ratio": spaces / n,
        "symbol_ratio": symbols / n,
        "word_count": float(len(words)),
        "avg_word_len": (
            sum(word_lengths) / len(word_lengths)
            if word_lengths
            else 0.0
        ),
        "entropy": _entropy(s),
    }

def _entropy(s: str) -> float:
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum(
        (count / n) * math.log2(count / n)
        for count in counts.values()
    )

def _has_many_control_chars(s: str) -> bool:
    allowed = {"\t", "\n", "\r"}
    control_chars = sum(
        unicodedata.category(ch).startswith("C") and ch not in allowed
        for ch in s
    )
    return control_chars / len(s) > 0.02

def _looks_like_structured_text(s: str) -> bool:
    """
    Contains patterns like:
    - key: value
    - key = value
    - Markdown headings
    - Bullet points
    - short JSON/YAML-like metadata
    """
    if len(s) > 300:
        return False
    structured_patterns = [
        r"^\s*[-*•]\s+\S+",
        r"^\s*#{1,6}\s+\S+",
        r"^\s*[A-Za-z0-9_#.\-$ ]{2,80}\s*[:=]\s*\S+",
        r"^\s*\"?[A-Za-z0-9_#.\-$ ]+\"?\s*:\s*",
    ]
    return any(re.search(pattern, s) for pattern in structured_patterns)
