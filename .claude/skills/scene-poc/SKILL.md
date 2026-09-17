---
name: scene-poc
description: 'Spike PoC for the AI scene agent harness question (open-decisions.md #1). Runs mvp-acceptance.md''s cold-start Turn 1 (generation) + Turn 2 (refinement) worked example end to end: writes a Menger DSL .scala scene file, then edits it in place, invoking menger --scene <path> after each turn for a compile/load verdict. Use when asked to run the scene-agent PoC, the harness spike, or reproduce mvp-acceptance.md''s worked example.'
---

# scene-poc

Disposable, heavily-observed PoC harness for `open-decisions.md` §1 (agent harness:
write-our-own vs. adopt). This skill is **not** the production agent — it is the front half
of a spike that runs the MVP worked example inside a harness (a Claude Code skill) that was
neither written nor chosen, to surface real tool/turn/history requirements before a harness
gets picked. Story: `_bmad-output/specs/spec-ai-scene-agent/stories/2-harness-poc-spike.md`.

Do not deviate from the prompts below, do not regenerate the file on turn 2, and do not
elaborate beyond what each prompt asks for — all three are exactly what this spike is
measuring.

## Context this skill reads (never modifies)

- `reference/dsl-manifest.json` — CAP-7 manifest snapshot (a one-time copy of `menger`'s
  `ManifestGenerator` output; regenerate manually from `menger` if the DSL surface changes,
  this skill never re-derives it).
- `<menger-checkout>/menger-app/src/main/scala/examples/dsl/SierpinskiHDRRotation.scala` —
  the closest few-shot example: 4D object + `def scene(t: Float): Scene` animation. No corpus
  example combines 4D + animation + glass + level-over-time — compose from the manifest plus
  this partial example, do not fail for lack of an exact match.

`<menger-checkout>` is the sibling `menger` repo (e.g. `../menger` relative to this repo's
root, or wherever the user's workspace has it checked out) — ask if it isn't obvious.

## Preconditions (one-time setup, not a runtime dependency)

`reference/dsl-manifest.json` must exist in this repo. If it doesn't:

```bash
# in the menger checkout, on the branch this spike targets (e.g. feat/sprint-37)
sbt "mengerApp/runMain menger.tools.ManifestGenerator target/dsl-manifest.json"
# then copy target/dsl-manifest.json to <this repo>/reference/dsl-manifest.json
```

Note which `menger` branch/commit the snapshot was generated from (e.g. in the copy step's
commit message or a comment in `spike-report.md`) — the manifest isn't checked for drift
against the checkout at read time. This is a one-time snapshot copy, not something this skill
re-runs per invocation.

Requires a working `menger` toolchain (JDK, sbt, and a CUDA-capable native build for
`menger-geometry`) already set up per `menger`'s own README/AGENTS.md. Each compile/load
verdict below pays a full sbt boot plus a native rebuild the first time (tens of seconds);
this is expected, not a hang.

## Turn 1 — generation

**Exact prompt (verbatim from `mvp-acceptance.md`, do not paraphrase or simplify):**

> "a tesseract sponge that gets more intricate as it turns, about ten seconds, glass, dark background"

Steps:

1. Read `reference/dsl-manifest.json` for the current DSL surface: object types and their
   fields/defaults, material presets, light/camera/plane constructors. Read
   `SierpinskiHDRRotation.scala` for the shape of an animated 4D scene object (package,
   imports, `def scene(t: Float): Scene`, `Projection4DSpec` usage, registration pattern).
2. Compose a **new** `.scala` file (pick a scene object name, e.g. `TesseractSpongeGrowth`)
   satisfying the prompt:
   - "tesseract sponge" -> a 4D sponge object (`TesseractSponge` in the manifest's `objects`
     list), not a plain `Tesseract`.
   - "gets more intricate as it turns" -> `level` increases as a function of `t` (intricacy),
     combined with a rotation that is also a function of `t` (turning) — either a 3D
     `rotation: Vec3` term or a 4D `Projection4DSpec` rotation term, driven by the same `t`.
   - "about ten seconds" -> a timing comment/mapping from the animation's `t in [0,1]` domain
     to real seconds is advisory only (the DSL's `t` is unitless 0..1 per-loop); note the
     ~10s intent in a comment near the animation expression, since the CLI's frame/duration
     flags are outside this skill's scope.
   - "glass" -> `material = Some(Material.Glass)`.
   - "dark background" -> `background = Some(Color(...))` with a low-luminance color, and/or
     omit `envMap`/`planes` that would introduce a bright ground.
   - This combination does not exist anywhere in the 28-scene corpus (confirmed in the story's
     Code Map) — compose it from the manifest fields, do not retrieve a near-match and rename it.
3. `mkdir -p poc-run` if it doesn't already exist, then write the file into it (e.g.
   `poc-run/turn1-generated.scala`) — never into the `menger` checkout. (This spike's own
   evidence files are the story's tracked deliverable and are expected to be committed to
   *this* repo as part of the story's normal implementation diff, same as story 1's generated
   test fixtures were; the "never persisted" constraint this references, SPEC.md AD-7/AD-10, is
   about the *production* agent never using git as its scene-version store — it is not a ban on
   this spike keeping its own evidence.)
4. Get a compile/load verdict:
   ```bash
   cd <menger-checkout>
   sbt "mengerApp/runMain Main --scene <absolute-path-to-poc-run/turn1-generated.scala> --headless --save-name <output.png> --t <t-value-in-0..1>"
   ```
   (adjust the sbt target/flags to whatever this checkout's `menger` launcher actually expects
   if it differs — verify once against `MengerCLIOptions.scala` rather than assuming `sbt run
   --` works, which is not this project's invocation shape). This invokes the same
   `MengerCLIOptions --scene` contract at `MengerCLIOptions.scala:284-288`, which drives
   `SceneLoader.load` -> `SceneCompiler.compile` -> `loadFromFile` ->
   `loadByReflectionWithLoader`, `SceneLoader.scala:46,87,117`). Capture stdout/stderr
   verbatim. A `def scene(t: Float): Scene` top-level object should load into
   `LoadedScene.Animated` (`LoadedScene.scala:4-8`); a `val scene: Scene` would load into
   `Static` and fail the prompt's "gets more intricate as it turns" requirement.
5. If compile or load fails: **do not silently retry with a different prompt or a simplified
   scene.** Record the failure verbatim — this is itself spike evidence (criterion 1/6 in
   `open-decisions.md`). Only retry the *same* prompt with a corrected DSL usage if the error
   is a straightforward manifest-vs-usage mismatch (e.g. wrong field name); if it looks
   structural (the harness can't do something the spike needs to observe), stop and say so
   per the story's "Ask First" boundary rather than working around it.

## Turn 2 — refinement

**Exact prompt (verbatim, same session/context as Turn 1 — do not start fresh):**

> "slower, and start it already at level 1"

If Turn 1's compile/load verdict failed, stop here — do not refine a scene that isn't known to
compile and load; that would invalidate the refinement measurement.

Steps:

1. **Edit the Turn 1 file in place.** Do not regenerate it, do not create a new file
   (SPEC.md Constraints: "Turn one writes a file; every later turn edits one. Regeneration is
   prohibited as the refinement mechanism"). Copy `poc-run/turn1-generated.scala` to
   `poc-run/turn2-refined.scala` first if you want a byte-identical "before" artifact to diff
   against — either way, the edit itself must be a targeted change to the Turn 1 expression,
   never a fresh composition from the prompt.
2. "slower" -> reduce the rate at which the intricacy/rotation expression advances per `t`.
   **Caveat, confirmed by this spike's own run:** since `t` is fixed to `[0,1]` per rendered
   clip and real playback duration is a CLI concern (`--frames`/`--end-t`) outside this file,
   simply dividing the `t`-driven term (e.g. `progress = t * Speed` with `Speed < 1`) does not
   make the clip play slower at a given frame count — it truncates where the loop ends at
   `t=1` (less total rotation, less total level rise), which is a different effect than
   "slower." Apply the edit anyway if that's what a literal reading produces (do not silently
   invent a different mechanism), but record the mismatch as a spike finding rather than
   presenting it as a correct "slower."
   "start it already at level 1" -> change the level expression's floor/offset so that at
   `t = 0`, `level = 1` (not `0`).
3. The diff against Turn 1 must touch **only** the timing expression and the level floor.
   Camera, material, background and lights must be byte-identical. If the diff touches
   anything else, do not "clean it up" — record it as a spike finding (this is the concrete
   CAP-3 test).
4. Re-run the same compile/load verdict command from Turn 1, step 4, against the Turn 2 file.

## Evidence to capture (for `spike-report.md`, written separately, not by this skill)

After both turns:
- The Turn 1 file, the Turn 2 file, and `diff poc-run/turn1-generated.scala poc-run/turn2-refined.scala`.
- Both compile/load verdicts (exit code + captured output), verbatim on failure.
- How many tool calls / distinct actions this run took (read manifest, read example, write
  file, shell compile/load, edit file, shell compile/load again — or however many it actually
  took).
- Whether Turn 2's edit required re-reading Turn 1's file content, or whether it was retained
  from Turn 1's own generation without a re-read (multi-turn state retention, criterion 2/3).
- Anything this skill format could not express that the spike needed to observe (Boundaries &
  Constraints "Ask First" clause) — e.g. no native sandboxing/permission declaration narrower
  than the whole Claude Code tool surface, no built-in session/version history beyond the
  conversation transcript itself.

## Explicitly out of scope for this skill

- Lint, geometric checks, or semantic readback (CAP-4's full validation gauntlet) — later
  stories build that; this PoC only compiles/loads.
- Persisting generated scene files anywhere outside this repo, or treating them as anything
  other than this spike's own dev-artifact evidence (see the note on step 3 above).
- Opening the interactive render window and manually judging whether the result is
  "recognisably the thing that was asked for" — that half of `mvp-acceptance.md`'s bar is a
  manual/observed check per the story's Verification section. A headless, non-interactive
  compile/load/render-to-PNG verdict (step 4) is in scope and is not the same thing.
- Selecting the production harness. This skill's only job is to produce evidence for that
  later decision.
