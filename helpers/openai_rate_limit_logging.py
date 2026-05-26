"""Utilities for making OpenAI rate-limit cooldowns visible in logs."""

from __future__ import annotations

import re
from typing import Any

RED = "\033[91m"
RESET = "\033[0m"


def create_rate_limit_logging_http_client(
    logger: Any,
    operation: str,
    model_id: str,
    item_count: int,
) -> Any:
    """Create an httpx client that logs only exhausted OpenAI rate-limit buckets."""
    import httpx

    def log_rate_limit_headers(response: httpx.Response) -> None:
        log_openai_rate_limit_response(
            response=response,
            logger=logger,
            operation=operation,
            model_id=model_id,
            item_count=item_count,
        )

    return httpx.Client(event_hooks={"response": [log_rate_limit_headers]})


def log_openai_rate_limit_response(
    response: Any,
    logger: Any,
    operation: str,
    model_id: str,
    item_count: int,
) -> None:
    """Log a red warning only when a response shows a hit rate-limit bucket."""
    headers = response.headers
    remaining_requests = headers.get("x-ratelimit-remaining-requests")
    remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
    status_code = getattr(response, "status_code", None)

    if status_code != 429 and not (
        _is_exhausted(remaining_requests) or _is_exhausted(remaining_tokens)
    ):
        return

    limit_requests = headers.get("x-ratelimit-limit-requests")
    reset_requests = headers.get("x-ratelimit-reset-requests")
    limit_tokens = headers.get("x-ratelimit-limit-tokens")
    reset_tokens = headers.get("x-ratelimit-reset-tokens")
    retry_after = headers.get("retry-after") or headers.get("retry-after-ms")
    cooldown = _cooldown_display(headers)
    logger.warning(
        f"{RED}OpenAI rate limit hit: operation={operation} model={model_id} "
        f"item_count={item_count} status={status_code} cooldown={cooldown} "
        f"remaining_requests={remaining_requests}/{limit_requests} "
        f"reset_requests={reset_requests} "
        f"remaining_tokens={remaining_tokens}/{limit_tokens} "
        f"reset_tokens={reset_tokens} retry_after={retry_after}{RESET}"
    )


def rate_limit_wait_seconds(
    error: Exception,
    attempt_number: int,
    default_wait_seconds: float,
    max_wait_seconds: float,
) -> float:
    """Choose a retry wait from OpenAI headers, falling back to simple backoff."""
    retry_after = retry_after_seconds_from_error(error)
    if retry_after is not None:
        return min(retry_after, max_wait_seconds)

    headers = _headers_from_error(error)
    if headers is not None:
        reset_seconds = _max_reset_seconds(headers)
        if reset_seconds is not None:
            return min(reset_seconds, max_wait_seconds)

    return min(default_wait_seconds * attempt_number, max_wait_seconds)


def retry_after_seconds_from_error(error: Exception) -> float | None:
    """Read retry-after headers from an OpenAI/API exception when present."""
    headers = _headers_from_error(error)
    if headers is None:
        return None

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            return float(retry_after_ms) / 1000.0
        except (TypeError, ValueError):
            return None

    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            return None

    return None


def is_openai_rate_limit_error(error: Exception) -> bool:
    """Return whether an exception looks like an OpenAI rate-limit failure."""
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True

    error_name = error.__class__.__name__.lower()
    return "ratelimit" in error_name or "rate_limit" in error_name


def format_rate_limit_retry_message(
    operation: str,
    model_id: str,
    item_count: int,
    attempt_number: int,
    max_attempts: int,
    wait_seconds: float,
    error: Exception,
) -> str:
    """Build a red retry log message with a clear cooldown."""
    headers = _headers_from_error(error)
    cooldown = _cooldown_display(headers) if headers is not None else f"{wait_seconds:.1f}s"
    return (
        f"{RED}OpenAI rate limit hit: operation={operation} model={model_id} "
        f"item_count={item_count} attempt={attempt_number}/{max_attempts} "
        f"cooldown={cooldown} retrying_in_seconds={wait_seconds:.1f} "
        f"error={error}{RESET}"
    )


def _headers_from_error(error: Exception) -> Any | None:
    response = getattr(error, "response", None)
    return getattr(response, "headers", None)


def _cooldown_display(headers: Any) -> str:
    retry_after = headers.get("retry-after")
    if retry_after:
        return f"{retry_after}s"

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            return f"{float(retry_after_ms) / 1000.0:.1f}s"
        except (TypeError, ValueError):
            return str(retry_after_ms)

    reset_requests = headers.get("x-ratelimit-reset-requests")
    reset_tokens = headers.get("x-ratelimit-reset-tokens")
    if reset_requests and reset_tokens:
        return f"requests={reset_requests}, tokens={reset_tokens}"
    if reset_requests:
        return f"requests={reset_requests}"
    if reset_tokens:
        return f"tokens={reset_tokens}"
    return "unknown"


def _max_reset_seconds(headers: Any) -> float | None:
    reset_values = [
        _parse_duration_seconds(headers.get("x-ratelimit-reset-requests")),
        _parse_duration_seconds(headers.get("x-ratelimit-reset-tokens")),
    ]
    numeric_values = [value for value in reset_values if value is not None]
    return max(numeric_values) if numeric_values else None


def _parse_duration_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass

    matches = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", value)
    if not matches:
        return None

    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(amount) * multipliers[unit] for amount, unit in matches)


def _is_exhausted(value: str | None) -> bool:
    if value is None:
        return False
    try:
        return float(value) <= 0
    except ValueError:
        return False
