"""Provider exception text is redacted before it becomes a ModelError message (review
2026-10-14, msa#43): the message reaches history.jsonl, model-calls.jsonl and agent.log, and an
SDK error can echo a header, a query string or a URL with a credential in it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapters.model import ModelError, ModelRequest
from adapters.model_common import REDACTED, call_error, redact
from adapters.openai_compatible_model import OpenAICompatibleModelAdapter

_REQUEST = ModelRequest(system_prompt="s", user_prompt="u")


@pytest.mark.parametrize("text, secret", [
    ("Incorrect API key provided: sk-proj-AbCdEf0123456789xyz", "sk-proj-AbCdEf0123456789xyz"),
    ("401 for url https://x.googleapis.com/v1?key=AIzaSyA1234567890abcdefghijkLMNOP",
     "AIzaSyA1234567890abcdefghijkLMNOP"),
    ("request headers: Authorization: Bearer abc.def-ghi_jkl012345", "abc.def-ghi_jkl012345"),
    ("proxy at https://user:hunter2pass@proxy.example/v1 refused", "hunter2pass"),
    ('{"api_key": "x9y8z7w6v5u4"}', "x9y8z7w6v5u4"),
])
def test_key_shaped_strings_are_redacted(text, secret):
    out = redact(text)
    assert secret not in out
    assert REDACTED in out


def test_the_values_of_secret_environment_variables_are_redacted(monkeypatch):
    # Whatever its shape: a proxy key need not look like any known provider's.
    monkeypatch.setenv("MENGER_AGENT_LITELLM_API_KEY", "plain-looking-proxy-credential")
    assert "plain-looking-proxy-credential" not in redact(
        "auth failed for plain-looking-proxy-credential"
    )


def test_ordinary_text_is_left_alone():
    text = "Connection error: timed out after 30s reading from api.deepseek.com:443"
    assert redact(text) == text


def test_call_error_keeps_the_kind_and_the_description():
    error = call_error("call_failed", "DeepSeek API returned an error", RuntimeError("sk-AAAAAAAAAAAA"))
    assert error.kind == "call_failed"
    assert error.message == f"DeepSeek API returned an error: {REDACTED}"


@pytest.mark.parametrize("provider", ["deepseek", "openai", "kimi"])
def test_an_adapter_never_returns_the_key_in_its_error(provider):
    key = "sk-live-0123456789abcdefghij"
    adapter = OpenAICompatibleModelAdapter(provider=provider, api_key=key)

    def _raise(**_kwargs):
        raise RuntimeError(f"Error code: 401 - invalid api key {key} for {provider}")

    adapter._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_raise)))
    result = adapter.complete(_REQUEST)
    assert isinstance(result, ModelError)
    assert key not in result.message
