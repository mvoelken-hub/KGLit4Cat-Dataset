"""Custom structured completion errors."""

from typing import Any


class CompletionError(Exception):
    """Base error for structured completion failures."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelRetry(CompletionError):
    """Signal to re-prompt with error message appended.

    Same semantics as pydantic-ai ModelRetry: validation logic uses this
    to indicate output does not meet requirements and should be retried.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details)


class OutputParsingError(CompletionError):
    """JSON decode or schema validation failure after all retries."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details)


class MaxRetriesExceeded(CompletionError):
    """Exhausted output_retries without valid output."""

    def __init__(self, message: str = "Max retries exceeded", details: dict[str, Any] | None = None):
        super().__init__(message, details)
