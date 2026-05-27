"""Custom structured completion errors."""

from typing import Any


class CompletionError(Exception):
    """Base error for structured completion failures."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class OutputParsingError(CompletionError):
    """JSON decode or schema validation failure."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details)


class EmptyResponseError(OutputParsingError):
    """Ollama returned no response text to parse or repair."""

    def __init__(self, message: str = "Empty response from model", details: dict[str, Any] | None = None):
        super().__init__(message, details)


class MaxRetriesExceeded(CompletionError):
    """Exhausted output_retries without valid output."""

    def __init__(
        self,
        message: str = "Max retries exceeded",
        details: dict[str, Any] | None = None,
        last_error: Exception | None = None,
    ):
        details = dict(details or {})
        if last_error is not None:
            details.setdefault("last_error_type", type(last_error).__name__)
            details.setdefault("last_error", str(last_error))
        super().__init__(message, details)
        self.last_error = last_error
