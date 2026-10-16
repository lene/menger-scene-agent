"""What every model adapter shares (review 2026-10-14, msa#37: each adapter re-implemented
it, msa#43: and copied the provider's exception text verbatim into `ModelError.message`,
from where it reaches the history, `model-calls.jsonl` and `agent.log`).

- `MAX_OUTPUT_TOKENS`, `check_timeout`: the same limits and argument check everywhere.
- `call_error`: a failed call as a `ModelError`, with the exception text redacted: anything
  shaped like a credential, and the current value of any `*_KEY` / `*_TOKEN` / `*_SECRET`
  environment variable. An SDK error can echo a request header or a URL with a key in it.
- `truncated`, `unparseable`, `empty_response`: the shared response failures.
"""

from __future__ import annotations

import math
import os
import re
from typing import Optional

from core.ports import ModelError, ModelErrorKind

MAX_OUTPUT_TOKENS = 16000

REDACTED = "[redacted]"
# Key-shaped strings: OpenAI/Anthropic/DeepSeek/Moonshot-style `sk-...`, Google `AIza...`, a
# bearer token, `key=`/`api_key=`/`token=` query or header values, and credentials in a URL.
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    # also the JSON form, `"api_key": "..."`
    re.compile(r"(?i)\b((?:api[_-]?)?key|token|secret|password)([\"']?\s*[=:]\s*[\"']?)([^\s&\"',;]{6,})"),
    re.compile(r"(?i)(://)[^/\s:@]+:[^/\s@]+@"),
]
_SECRET_ENV_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET")
_MIN_SECRET_LENGTH = 8


def redact(text: str) -> str:
    for name, value in os.environ.items():
        if name.upper().endswith(_SECRET_ENV_SUFFIXES) and len(value) >= _MIN_SECRET_LENGTH:
            text = text.replace(value, REDACTED)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups == 3:
            text = pattern.sub(lambda m: m.group(1) + m.group(2) + REDACTED, text)
        elif pattern.groups == 1:
            text = pattern.sub(lambda m: m.group(1) + REDACTED + "@", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def check_timeout(timeout: Optional[float]) -> None:
    """A programmer-error precondition: a zero, negative, NaN or infinite timeout would
    otherwise reach the SDK client unchecked. `None` leaves the SDK's own default alone."""
    if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
        raise ValueError(f"timeout must be a finite, positive number, got {timeout!r}")


def call_error(kind: ModelErrorKind, what: str, error: BaseException) -> ModelError:
    """`what` names the failure ("Anthropic API call timed out"); the exception's own text
    follows, redacted."""
    return ModelError(kind=kind, message=f"{what}: {redact(str(error))}", cause=error)


def truncated() -> ModelError:
    return ModelError(
        kind="invalid_output",
        message=f"Model response was truncated at {MAX_OUTPUT_TOKENS} tokens "
        "before completing -- not usable scene text",
    )


def unparseable(error: BaseException) -> ModelError:
    return call_error("invalid_output", "Could not parse the model response", error)


def empty_response() -> ModelError:
    return ModelError(kind="invalid_output", message="Model returned an empty response")
