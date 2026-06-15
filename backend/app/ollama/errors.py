"""Custom structured completion errors."""

from typing import Any

from app.ollama.usage import RunUsage


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
        failed_response: str | None = None,
        first_response: str | None = None,
        usage: RunUsage | None = None,
        prompt_diagnostics: Any | None = None,
    ):
        details = dict(details or {})
        if last_error is not None:
            details.setdefault("last_error_type", type(last_error).__name__)
            details.setdefault("last_error", str(last_error))
        if failed_response is not None:
            details.setdefault("failed_response", failed_response)
        if first_response is not None:
            details.setdefault("first_response", first_response)
        super().__init__(message, details)
        self.last_error = last_error
        self.failed_response = failed_response
        self.first_response = first_response
        self.usage = usage or RunUsage()
        self.prompt_diagnostics = prompt_diagnostics
