# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A release is a `<version>` tag
(`.github/workflows/release.yml`); its section here names the menger release it pairs with.

## [Unreleased]

Pairs with menger 0.9.0 (DSL manifest 1.11.0); first release not yet tagged.

### Added

- The REPL agent: generate, revise, validate (sandboxed) and render menger DSL scenes, with
  self-repair, semantic readback, consult turns, hand-edit promotion and ticket drafts.
- Model backends: Anthropic, Gemini, DeepSeek, OpenAI, Kimi/Moonshot, any LiteLLM proxy.
- Session provenance: `session.json`, `model-calls.jsonl`, `agent.log`, and a timestamp and
  run identity on every history entry (#31).
- `--check`: does this agent fit the installed menger's launchers and images (#32).
- A replay eval of recorded usability turns in the unit suite (`evals/`, #30).
- `/export image [t]` and `/export video [seconds]`; an export asked for in words ("save the
  animation as mp4") gets that hint instead of a model call (#56).
