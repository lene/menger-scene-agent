"""Real API round-trip tests for the OpenAI-compatible adapter family (DeepSeek, OpenAI,
Kimi/Moonshot), one independently-skipped case per provider -- deliberately NOT a single
module-level `pytestmark`, since that would skip all three together the moment any one key
is missing, breaking the "degrades gracefully with 0..N keys present" property every other
integration file in this repo already has. Each provider's own env var gates only its own
case. Mirrors tests/test_integration.py's Turn-1 prompt and assertions; the revise/readback
checks aren't repeated per provider here since `complete()`'s parsing logic is already proven
identical across the three in tests/test_openai_compatible_model_adapter.py -- this file is
about proving each vendor's real API actually round-trips through the shared adapter, not
re-proving CAP-2/CAP-6 a third and fourth time."""

from __future__ import annotations

import os

import pytest

from adapters.artifacts import load_corpus, load_manifest
from adapters.openai_compatible_model import OpenAICompatibleModelAdapter
from core.generation import generate
from core.types import GenerationError

TURN_1_PROMPT = (
    "a tesseract sponge that gets more intricate as it turns, about ten seconds, "
    "glass, dark background"
)

_PROVIDER_ENV_VARS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
}


def _skip_reason(env_var: str) -> str:
    return f"{env_var} not set -- skipping the real API integration test"


@pytest.mark.parametrize(
    "provider",
    [
        pytest.param(
            provider,
            marks=pytest.mark.skipif(
                not os.environ.get(env_var), reason=_skip_reason(env_var)
            ),
        )
        for provider, env_var in _PROVIDER_ENV_VARS.items()
    ],
)
def test_generate_turn_1_prompt_against_the_real_api(provider):
    manifest = load_manifest("reference/dsl-manifest.json")
    corpus = load_corpus("reference/dsl-corpus.json")
    adapter = OpenAICompatibleModelAdapter(provider=provider)

    result = generate(TURN_1_PROMPT, manifest, corpus, adapter)

    assert not isinstance(result, GenerationError), (
        result.message if isinstance(result, GenerationError) else ""
    )
    assert isinstance(result, str)
    assert result.strip() != ""

    first_60_lines = result.splitlines()[:60]
    assert any(line.strip().startswith("object ") for line in first_60_lines), (
        f"[{provider}] Expected a top-level 'object' declaration within the first 60 lines "
        "(SceneLoader.detectObjectName's heuristic); got:\n" + result
    )
