# menger-scene-agent

An AI coding agent that turns plain-language scene descriptions into valid
[Menger](../menger) DSL scene files, validates them, renders them in
menger's existing interactive window, and refines them by editing.

Local, unpublished repo — no remote configured yet.

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

Story 1 (capability manifest generator) not yet started. Language and
runtime are deliberately undecided — see the spec's open questions and the
spine's Deferred section (harness spike, story 2).
