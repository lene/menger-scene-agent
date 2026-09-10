"""Unit tests for adapters/model_factory.py -- provider resolution precedence, error
behavior for a bad/unrecognized selection, and the lazy-import property that keeps importing
this module alone from requiring any provider's SDK to be installed. No real network call
anywhere: every adapter constructed here uses an explicit fake API key, so construction
succeeds without needing a real credential or touching the network."""

from __future__ import annotations

import subprocess
import sys

import pytest

from adapters.model import AnthropicModelAdapter, MissingAPIKeyError, UnknownProviderError
from adapters.gemini_model import GeminiModelAdapter
from adapters.openai_compatible_model import OpenAICompatibleModelAdapter
from adapters.model_factory import get_model_adapter, DEFAULT_PROVIDER, _ENV_VAR, _FACTORIES


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Every test starts from a clean slate: no provider selected by environment, and a fake
    key present for whichever provider the test actually exercises (set per-test below)."""
    monkeypatch.delenv(_ENV_VAR, raising=False)


def test_default_provider_is_anthropic():
    assert DEFAULT_PROVIDER == "anthropic"


def test_no_argument_and_no_env_var_defaults_to_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-a-real-credential")

    adapter = get_model_adapter()

    assert isinstance(adapter, AnthropicModelAdapter)


def test_env_var_alone_selects_the_right_provider(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")

    adapter = get_model_adapter()

    assert isinstance(adapter, GeminiModelAdapter)


def test_env_var_is_case_insensitive_and_stripped(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "  GEMINI  ")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")

    adapter = get_model_adapter()

    assert isinstance(adapter, GeminiModelAdapter)


def test_explicit_argument_wins_over_the_environment_variable(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-a-real-credential")

    adapter = get_model_adapter(provider="deepseek")

    assert isinstance(adapter, OpenAICompatibleModelAdapter)
    assert adapter._provider_name == "DeepSeek"


@pytest.mark.parametrize("provider_key,expected_type", [
    ("anthropic", AnthropicModelAdapter),
    ("gemini", GeminiModelAdapter),
    ("deepseek", OpenAICompatibleModelAdapter),
    ("openai", OpenAICompatibleModelAdapter),
    ("kimi", OpenAICompatibleModelAdapter),
])
def test_every_registered_provider_constructs_the_expected_adapter_type(
    provider_key, expected_type, monkeypatch
):
    for env_var in (
        "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
        "MOONSHOT_API_KEY",
    ):
        monkeypatch.setenv(env_var, "test-key-not-a-real-credential")

    adapter = get_model_adapter(provider=provider_key)

    assert isinstance(adapter, expected_type)


def test_unknown_provider_argument_raises_a_typed_error_naming_valid_choices():
    with pytest.raises(UnknownProviderError) as exc_info:
        get_model_adapter(provider="not-a-real-provider")

    message = str(exc_info.value)
    assert "not-a-real-provider" in message
    for valid in _FACTORIES:
        assert valid in message


def test_unknown_provider_via_env_var_raises_the_same_typed_error(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "not-a-real-provider")

    with pytest.raises(UnknownProviderError):
        get_model_adapter()


def test_missing_key_for_the_selected_provider_propagates_unchanged(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError):
        get_model_adapter(provider="deepseek")


def test_importing_the_factory_alone_does_not_import_any_provider_sdk():
    # A regression test for the load-bearing property described in model_factory.py's own
    # module docstring: importing this module must not force google-genai/openai to be
    # installed, or the whole point of them being optional extras breaks. Run in a fresh
    # subprocess rather than relying on this test process's own sys.modules, since pytest's
    # collection may have already imported sibling test modules (and therefore their SDKs)
    # before this test runs.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import adapters.model_factory, sys; "
            "assert 'openai' not in sys.modules, 'openai leaked'; "
            "assert 'google.genai' not in sys.modules, 'google.genai leaked'; "
            "print('OK')",
        ],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
