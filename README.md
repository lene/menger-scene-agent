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

The CLI-agent epic (stories 10-22) is complete (see each story's frontmatter
under `../_bmad-output/specs/spec-ai-scene-agent/stories/`) -- a persistent
REPL, hand-edit fallback, live status line, timeout/clarification/consult
turn handling, and `/retry` confirmation are all implemented. Stories 3, 4, 8
predate this epic and track separately.

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

**The default is Anthropic regardless of which keys you have set.** Having
`GEMINI_API_KEY` set does *not* select Gemini — `cli.py` still tries to
construct the Anthropic adapter unless you explicitly set
`MENGER_AGENT_MODEL_PROVIDER=gemini` (or `deepseek`/`openai`/`kimi`). This is
the single most common setup mistake: `ANTHROPIC_API_KEY is not set` while a
different provider's key sits right there in the environment.

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

## Running the agent

Set `MENGER_AGENT_MODEL_PROVIDER` if you're not using the Anthropic default
(see table above), and that provider's API key. Two more env vars are
required regardless of provider, both pointing into the sibling `menger`
repo -- unset either and `cli.py` exits with a clear error before any model
call or session is created:

| Env var | Points to |
|---|---|
| `MENGER_SCENE_VALIDATOR_SCRIPT` | the renderer-side scene validator script, e.g. `../menger/docker/scene-validator/run-sandboxed.sh` |
| `MENGER_RENDER_LAUNCHER` | the staged `menger-app` render window launcher binary, e.g. `../menger/menger-app/target/universal/stage/bin/menger-app` (build it first with `sbt stage` in `menger/` if it doesn't exist yet) |

Optional: `MENGER_AGENT_SESSIONS_DIR` (default `./sessions`).

Anthropic (default):

```bash
export MENGER_SCENE_VALIDATOR_SCRIPT=../menger/docker/scene-validator/run-sandboxed.sh
export MENGER_RENDER_LAUNCHER=../menger/menger-app/target/universal/stage/bin/menger-app
export ANTHROPIC_API_KEY=...

.venv/bin/python3 cli.py
```

Gemini (or any other provider — same shape, swap the two lines):

```bash
export MENGER_SCENE_VALIDATOR_SCRIPT=../menger/docker/scene-validator/run-sandboxed.sh
export MENGER_RENDER_LAUNCHER=../menger/menger-app/target/universal/stage/bin/menger-app
export MENGER_AGENT_MODEL_PROVIDER=gemini
export GEMINI_API_KEY=...   # already installed: pip install -e ".[gemini]"

.venv/bin/python3 cli.py
```

```bash
.venv/bin/python3 cli.py --session <id>      # resume an existing session (id = its directory name under the sessions dir)
```

In the REPL:
- Plain text -> a generate/revise turn: prints `Turn N: <outcome>`, refreshes the render window on acceptance.
- `/ask <question>` or `/question <question>` -> a consult turn: prose answer grounded in the DSL manifest/corpus and current scene; never edits the scene, never consumes an ordinal.
- `/retry` -> resends the prompt that most recently failed with a `generation_timeout`. Nothing pending -> reports nothing to retry. Scene changed since the failure (hand edit or another accepted turn) -> resends immediately. Unchanged -> first `/retry` explains nothing has changed and arms a confirmation gate; a second `/retry` (still unchanged) resends. Never auto-retried (PRD FR7).
- The render window's own output (including render errors) goes to `render.stdout.log` / `render.stderr.log` in the session directory, overwritten on each refresh -- check there if the window shows nothing or stops responding.
- Hand-editing: edit the current scene file on disk directly between turns. Each loop iteration checks for such an edit and either promotes it to a new ordinal, reports the lint violation that blocked it, or reports a storage failure.

`cli.py --help` prints this same reference.

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
