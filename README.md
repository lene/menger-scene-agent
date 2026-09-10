# menger-scene-agent

An AI coding agent that turns plain-language scene descriptions into valid
[Menger](../menger) DSL scene files, validates them, renders them in
menger's existing interactive window, and refines them by editing.

GitHub: `lene/menger-scene-agent`.

## The contract

This repo implements the spec at
`../_bmad-output/specs/spec-ai-scene-agent/SPEC.md`, governed by the
architecture spine at
`../_bmad-output/planning-artifacts/architecture/architecture-ai-scene-agent-2026-08-30/ARCHITECTURE-SPINE.md`.
Story breakdown: `../_bmad-output/specs/spec-ai-scene-agent/stories.yaml`.

Its only coupling to `menger` is the CLI contract (`--scene`,
`--texture-dir`, `--optix`) — it never depends on `menger` as a library and
never modifies renderer code.

## Status

All 9 stories are implemented (see each story's frontmatter under
`../_bmad-output/specs/spec-ai-scene-agent/stories/`). The renderer-side
stories (1, 3's renderer half, 5, 8) shipped in `menger` 0.9.0; the
agent-side stories (2, 3's agent half, 4, 6, 7, 9) are on this repo's
`feat/sprint-37` branch, PR #1.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # core dependencies: anthropic, pytest
```

To use a model backend other than the default (Anthropic), install that
provider's extra too:

```bash
.venv/bin/pip install -e ".[gemini]"              # Gemini only
.venv/bin/pip install -e ".[openai-compatible]"   # DeepSeek, OpenAI, Kimi/Moonshot
.venv/bin/pip install -e ".[gemini,openai-compatible]"  # everything
```

(`requirements.txt` mirrors only the hard `dependencies` in `pyproject.toml` — the
per-provider extras are installed separately so a default-Anthropic setup never pulls in
SDKs it won't use.)

## Choosing a model backend

`adapters/model_factory.py`'s `get_model_adapter()` is the entry point — not any
individual adapter class directly. It picks a provider by, in order: an explicit
`provider=` argument, else the `MENGER_AGENT_MODEL_PROVIDER` environment variable
(case-insensitive), else `anthropic` (the default, so existing zero-config usage is
unaffected).

| Provider | `MENGER_AGENT_MODEL_PROVIDER` value | Required env var | Extra needed |
|---|---|---|---|
| Anthropic Claude (default) | `anthropic` | `ANTHROPIC_API_KEY` | none (base install) |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | `gemini` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `openai-compatible` |
| OpenAI / ChatGPT | `openai` | `OPENAI_API_KEY` | `openai-compatible` |
| Kimi (Moonshot AI) | `kimi` | `MOONSHOT_API_KEY` | `openai-compatible` |

Only the selected provider's key needs to be set. An unrecognized provider name raises
`UnknownProviderError`; a selected provider with no key set raises `MissingAPIKeyError` —
both from `adapters/model.py`, both before any network call.

Adding a further OpenAI-compatible vendor (same request/response shape as OpenAI's own chat
completions API) costs one `_ProviderSpec` table entry in
`adapters/openai_compatible_model.py` plus one factory-table entry in
`adapters/model_factory.py` — no new adapter class. A genuinely different wire protocol
needs a new adapter file mirroring `adapters/gemini_model.py`'s shape.

## Running the tests

The unit test suite constructs every provider's adapter (each with a monkey-patched client,
never a real network call — see below), so it needs every provider's SDK installed, even if
you only ever plan to *use* one:

```bash
.venv/bin/pip install -e ".[dev,gemini,openai-compatible]"
.venv/bin/python -m pytest
```

Installing only the extra for the one provider you actually use (the "Setup" section above)
is correct for *using* the agent, but running `pytest` without every provider's SDK present
gets `ModuleNotFoundError` failures from the other providers' adapter-construction tests, not
clean skips — those tests are unconditional by design, proving each adapter's own
construction/error-mapping logic, not gated on whether you personally plan to use that
provider.

Unit tests need no API key and no network regardless — they run against fakes
(`tests/fakes.py`) or a real adapter with its private SDK client monkey-patched
(`tests/test_model_adapter.py` and its siblings). Integration tests
(`tests/test_integration.py`, `tests/test_gemini_integration.py`,
`tests/test_openai_compatible_integration.py`) make real API calls and skip cleanly,
per-provider, when that provider's key isn't set — set whichever key(s) you have to
exercise them for real.
