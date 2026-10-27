# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A release is a `<version>` tag
(`.github/workflows/release.yml`); its section here names the menger release it pairs with.

## [Unreleased]

Pairs with menger 0.9.0 (DSL manifest 1.11.0); first release not yet tagged.

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
- `/export image [t]` and `/export video [seconds]`; an export asked for in words ("save the
  animation as mp4") gets that hint instead of a model call (#56).
