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

In the workspace architecture docs this is arc42 decision AD-36
(`../docs/arc42/09-architectural-decisions.md`); the spine's own decisions are cited there as
SA-AD-n, while this repo's code and docstrings call them plain "AD-n".

Its only coupling to `menger` (0.9.0 or later) is through processes and files — it never
depends on `menger` as a library and never modifies renderer code:

- `menger-app --scene <file.scala> --display <:N> --render-lock-path <path>` — the render window
- `docker/scene-validator/run-sandboxed.sh` — sandboxed restricted-classpath compile +
  geometric checks, verdict JSON on stdout
- `reference/dsl-manifest.json`, `reference/dsl-corpus.json` — committed copies of the output
  of menger's `ManifestGenerator` and `CorpusExporter` (see "Refreshing the reference
  artifacts" below)

## Status

All 22 stories in `stories.yaml` are implemented (see each story's frontmatter under
`../_bmad-output/specs/spec-ai-scene-agent/stories/`): the original slate 1-9 (manifest,
harness spike, generate/revise, both gauntlets, semantic readback, history, render window,
ticket escalation) and the CLI-agent epic 10-22 (persistent REPL, hand-edit fallback, live
status line, timeout/clarification/consult turn handling, `/retry` confirmation). Sprint 37
remains open for the MVP acceptance run. All of this lives on `feat/sprint-37`; `main` is not
yet merged.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync   # core dependencies: anthropic, pytest
```

To use a model backend other than the default (Anthropic), sync that
provider's extra too:

```bash
uv sync --extra gemini              # Gemini only
uv sync --extra openai-compatible   # DeepSeek, OpenAI, Kimi/Moonshot, a LiteLLM proxy
uv sync --all-extras                # everything
```

(Per-provider extras are separate so a default-Anthropic setup never pulls in SDKs it won't
use.)

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
| A LiteLLM proxy | `litellm` | `LITELLM_API_KEY` and `LITELLM_BASE_URL` | `openai-compatible` |

Only the selected provider's key needs to be set. An unrecognized provider name raises
`UnknownProviderError`; a selected provider with no key set raises `MissingAPIKeyError` —
both from `adapters/model.py`, both before any network call.

**Choosing the model.** `MENGER_AGENT_MODEL` names the model to use with the selected
provider (any provider, e.g. `deepseek-reasoner`); unset, each provider uses its default
(Anthropic `claude-opus-5`, Gemini `gemini-3.1-pro-preview`, DeepSeek `deepseek-chat`, OpenAI
`gpt-5`, Kimi `kimi-k2`, LiteLLM `claude-sonnet-5.5`). The agent prints `Model: <provider>
<model>` at startup, next to the `Session:` line.

**LiteLLM.** Any OpenAI-compatible LiteLLM proxy works; its model names are the proxy's own
(`MENGER_AGENT_MODEL`). In fish:

```fish
set -Ux LITELLM_BASE_URL https://<your-litellm-host>/v1
set -Ux LITELLM_API_KEY <your key>
set -x MENGER_AGENT_MODEL_PROVIDER litellm
set -x MENGER_AGENT_MODEL claude-opus-5.5   # optional: one of the proxy's models
```

**Tests that call a real model API** run only with `MENGER_AGENT_LIVE_TESTS=1` and the
provider's key set (`tests/live_api.py`), so a key in the environment alone does not spend
credit on every test run.

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
| `MENGER_RENDER_LAUNCHER` | the contained render window, `../menger/docker/scene-validator/run-render-window.sh` (build its image together with the validator's: `../menger/docker/scene-validator/build.sh`). It needs Docker with the NVIDIA Container Toolkit (CDI) and an X display. |

Optional: `MENGER_AGENT_SESSIONS_DIR` (default `./sessions`).

The render window compiles and runs every accepted scene, so it runs in the same kind of
container as the validator: no network, no capabilities, read-only root, only the session's
`window/` directory mounted. Pointing `MENGER_RENDER_LAUNCHER` at the bare
`menger-app` launcher instead is refused unless you also set `MENGER_AGENT_HOST_RENDER=1` --
then generated scene code runs on your host, as you, with network access.

Anthropic (default):

```bash
export MENGER_SCENE_VALIDATOR_SCRIPT=../menger/docker/scene-validator/run-sandboxed.sh
export MENGER_RENDER_LAUNCHER=../menger/docker/scene-validator/run-render-window.sh
export ANTHROPIC_API_KEY=...

uv run menger-scene-agent
```

Gemini (or any other provider — same shape, swap the two lines):

```bash
export MENGER_SCENE_VALIDATOR_SCRIPT=../menger/docker/scene-validator/run-sandboxed.sh
export MENGER_RENDER_LAUNCHER=../menger/docker/scene-validator/run-render-window.sh
export MENGER_AGENT_MODEL_PROVIDER=gemini
export GEMINI_API_KEY=...   # already synced: uv sync --extra gemini

uv run menger-scene-agent
```

```bash
uv run menger-scene-agent --session <id>      # resume an existing session (id = its directory name under the sessions dir)
```

In the REPL:
- Plain text -> a generate/revise turn: prints `Turn N: <outcome>`. On acceptance the render window launches (first turn) or reloads in place (later turns, via menger's own file watcher -- the window never flickers or relaunches after the first accepted turn).
- `/ask <question>` or `/question <question>` -> a consult turn: prose answer grounded in the DSL manifest/corpus and current scene; never edits the scene, never consumes an ordinal.
- `/retry` -> resends the prompt that most recently failed with a `generation_timeout`. Nothing pending -> reports nothing to retry. Scene changed since the failure (hand edit or another accepted turn) -> resends immediately. Unchanged -> first `/retry` explains nothing has changed and arms a confirmation gate; a second `/retry` (still unchanged) resends. Never auto-retried (PRD FR7).
- `/export image [t]` / `/export video [seconds]` -> renders the accepted scene with menger, in the same container as the window, to `window/export/NNN.png` (at `t`, default 0) or `NNN.mp4` (t from 0 to `seconds`, default the scene's `durationSeconds`, 24 fps). No model call; the log of each export sits next to it as `NNN.log`.
- The render window's own output (including render errors) goes to `render.stdout.log` / `render.stderr.log` in the session directory; the previous five launches are kept as `render.stderr.1.log` ... `.5.log` -- check there if the window shows nothing or stops responding.
- Hand-editing: edit the current scene file on disk directly between turns. Each loop iteration checks for such an edit and either promotes it to a new ordinal, reports the lint violation that blocked it, or reports a storage failure.

`uv run menger-scene-agent --help` prints this same reference.

### What a session keeps

Everything needed to reconstruct a session -- for a bug report, say -- is in its directory:

| File | What |
|---|---|
| `session.json` | what the session was started with: agent and Python version, provider and model, manifest version, DSL revision and sha256, validator image id and DSL label, launcher paths |
| `history.jsonl` | one entry per turn: prompt, outcome, file, readback; each with a timestamp (`ts`) and the run's agent version, provider, model and DSL revision |
| `model-calls.jsonl` | every model call: the request, the raw response (before the scene is extracted) or the error, and its duration -- including each automatic repair round |
| `agent.log` | the agent's log: validator runs with their full stderr, window launches, turn outcomes |
| `NNN.scala`, `rejected-NNN.scala` | accepted scenes, and the candidates of rejected attempts |

These files hold your prompts and the model's answers. Look through them before you attach
them to a public issue.

## Install

The agent is paired with menger: it validates and renders through menger's two Docker images,
so both must be built from a menger whose DSL matches the agent's manifest. A released agent
names the menger release it was made with in its release notes.

```bash
# 1. menger, at the release the agent's notes name (0.9.0 or later)
git clone https://github.com/lene/menger.git; cd menger; git checkout <menger-version>
docker/scene-validator/build.sh          # builds menger-scene-validator and menger-render-window
cd ..

# 2. the agent, from its release tag
git clone https://github.com/lene/menger-scene-agent.git; cd menger-scene-agent
git checkout <agent-version>
uv sync --extra openai-compatible        # or the extra for your provider, see "Setup"

# 3. check that the two fit
export MENGER_SCENE_VALIDATOR_SCRIPT=../menger/docker/scene-validator/run-sandboxed.sh
export MENGER_RENDER_LAUNCHER=../menger/docker/scene-validator/run-render-window.sh
uv run menger-scene-agent --check
```

`--check` reports each part and what to do when one does not fit. It checks that both launcher
scripts exist and are executable, and that both images exist, were built from menger 0.9.0 or
later, and were built from the DSL in this agent's `reference/dsl-manifest.json`. It exits 0
when everything fits. The images need the NVIDIA Container Toolkit and a licensed OptiX SDK at
build time (see `../menger/docker/scene-validator/build.sh`).

Releases: a `<version>` tag (bare, like menger's) whose version matches `pyproject.toml` and a dated
`CHANGELOG.md` section publishes a GitHub release with that section as its notes
(`.github/workflows/release.yml`). The agent runs from a checkout of the tag; there is no
wheel or package-index release (a wheel would lack `reference/`).

## Refreshing the reference artifacts

`reference/dsl-manifest.json` and `reference/dsl-corpus.json` describe the DSL the model may
use. At startup `cli.py` checks their schema version and exits with a message naming the stale
artifact — before any model call — if they don't match what this agent expects. After a menger
DSL change, regenerate them from the sibling `menger` checkout and commit the result:

```bash
cd ../menger
sbt "mengerApp/runMain menger.tools.ManifestGenerator ../menger-scene-agent/reference/dsl-manifest.json"
sbt "mengerApp/runMain menger.tools.CorpusExporter ../menger-scene-agent/reference/dsl-corpus.json"
cp contracts/process-result.json ../menger-scene-agent/reference/process-result.json
```

`reference/process-result.json` is menger's process contract: the result the validator and
the render window print when they end at once, the exit code of each tag, and the window's
event and state versions. `tests/test_process_contract.py` holds the agent's parsers to it.

## Running the tests

The unit test suite constructs every provider's adapter (each with a monkey-patched client,
never a real network call — see below), so it needs every provider's SDK installed, even if
you only ever plan to *use* one:

```bash
uv sync --all-extras --dev
uv run pytest
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
