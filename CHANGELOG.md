# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A release is a `<version>` tag
(`.github/workflows/release.yml`); its section here names the menger release it pairs with.

## [0.1.0] - 2026-10-31

First release. Pairs with menger 0.9.1 (DSL manifest 2.0.0, result contract 1.3.0); menger 0.9.0
was released prematurely and is refused by `--check` (the minimum is 0.9.1).

### Changed

- **Breaking: reads DSL manifest 2.0.0** (menger#84, #95). `compileSurface.types` is gone, so a
  scene may import a menger package (`menger.dsl`, `menger.objects`, `menger.video`,
  `menger.common`, and now `menger.spec`) but no single root-package type: `import
  menger.Projection4DSpec` is `import menger.spec.Projection4DSpec`, and the vocabulary's
  `IMPORT_EXACT` is removed. `EXPECTED_MANIFEST_SCHEMA_VERSION` is `2.0.0`; a 1.x manifest is
  refused. The reference manifest and corpus are regenerated from menger: fields may carry a
  `unit` (degrees or radians), `Projection4DSpec` is a scene-composition type, and
  `Caustics.alpha` is `radiusReduction`.
- The drift check of the recorded cases (`python -m evals.recorded --validator ...`, the nightly
  `agent-contract.yml` step) no longer counts the validator result's `schemaVersion` as a changed
  answer: menger's additive bump to 1.3.0 (menger#113) made all 8 cases drift, and the bump
  carries no information about the scene. Compatibility of the version is still checked on
  every real run (`result_problem`). Found by running that step locally before its first
  nightly run: 23 validator calls differed, only in `schemaVersion`.
- The recorded eval case `tesseract-sponge-animation` has the import of `Projection4DSpec` in
  its two stored responses migrated to `menger.spec.Projection4DSpec` (decided by the
  maintainer 2026-10-30 over a live re-record; nothing else in the responses changed). The
  nightly run against menger's real validator is the check that its answers still match.

- Reads validator result schema 1.3.0 (menger#113): `loadError` says why a scene did not load,
  and a compile failure's repair feedback is one `line N: message` per compiler error instead
  of one multi-line message. An older validator image (no `loadError`) behaves as before.

### Security

- The agent no longer follows what the generated scene leaves in `<session>/window/`, the
  directory the window and export containers can write (menger#88). Opening the window logs
  for writing followed a symlink there, so a scene could create or overwrite a host file, for
  example one the user's shell sources, and fill it with the next window's output, which the
  scene controls. Logs are now created fresh, without following symlinks. Window state,
  events, logs and exports are read as regular files only, size-capped; a symlinked `export/`
  gets nothing. The window state must be finite, in-range numbers or it is ignored, and window
  events keep only the contract's kinds.
- Everything the REPL prints goes through `safe_text`: no terminal control or escape
  sequences, and no bidirectional overrides, from model replies, replayed history or window
  events (msa#42).
- The allowlist check now sees references in an interpolated string's `${...}` splices, with
  whitespace or a comment around their dots, and `getClass`/`getClassLoader` called on any
  value (menger#78). It stays advisory; the containers are the barrier.

### Added

- The REPL agent: generate, revise, validate (sandboxed) and render menger DSL scenes, with
  self-repair, semantic readback, consult turns, hand-edit promotion and ticket drafts.
- Model backends: Anthropic, Gemini, DeepSeek, OpenAI, Kimi/Moonshot, any LiteLLM proxy.
- Session provenance: `session.json`, `model-calls.jsonl`, `agent.log`, and a timestamp and
  run identity on every history entry (#31).
- `--check`: does this agent fit the installed menger's launchers and images (#32).
- A replay eval of recorded usability turns in the unit suite (`evals/`, #30). Each case
  holds the validator's answers too, so the replay runs the snapshot and budget checks, and a
  nightly job compares the answers with menger's real validator. `--record` writes them
  (menger#78).
- `scripts/time_validator.py`: the cold-start wall time of the sandboxed validator over the
  reference corpus (p50, p90, max over several passes), checked against the turn-latency
  budget of arc42 §10 P10 (p50 ≤ 5 s, p90 ≤ 10 s per run). Advisory; `--enforce` makes an
  overrun exit 1, for a CI gate once the nightly job's runner is measured (menger#91).
- `/export image [t]` and `/export video [seconds]`; an export asked for in words ("save the
  animation as mp4") gets that hint instead of a model call (#56).
