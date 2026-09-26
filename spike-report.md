# Harness PoC spike report

Story: `_bmad-output/specs/spec-ai-scene-agent/stories/2-harness-poc-spike.md`
Criteria under test: `_bmad-output/specs/spec-ai-scene-agent/open-decisions.md` §1

This spike ran `mvp-acceptance.md`'s cold-start worked example (Turn 1 generation, Turn 2
refinement) once, end to end, as a Claude Code skill
(`.claude/skills/scene-poc/SKILL.md`) — a harness neither written nor chosen for this
project. The goal was not to build the agent; it was to observe what a real run needs, so
the six criteria below can be answered from evidence instead of speculation.

## What actually happened

- **Setup:** `sbt "mengerApp/runMain menger.tools.ManifestGenerator target/dsl-manifest.json"`
  ran once against `menger` on `feat/sprint-37`, exit 0, produced a 27,011-byte manifest,
  copied to `reference/dsl-manifest.json`.
- **Turn 1** ("a tesseract sponge that gets more intricate as it turns, about ten seconds,
  glass, dark background"): composed `poc-run/turn1-generated.scala` — a `TesseractSponge`
  (`VolumeRemoving`), `Material.Glass`, `level` rising 0→3 and a 4D `rotXW` sweep both driven
  by a single `Speed`-scaled `progress` value, `background = Some(Color(0.02, 0.02, 0.03))`,
  no floor plane. Verdict command:
  `sbt "mengerApp/runMain Main --scene <path> --headless --save-name /tmp/turn1-poc.png --t 0.5"`
  → **exit 0**. Log:
  `SceneLoader - Loading scene: .../turn1-generated.scala` then
  `SceneLoader - Successfully loaded compiled scene: poc.TesseractSpongeGrowth`, followed by
  an `InteractiveEngine` frame render (no exception). Because the object exposes
  `def scene(t: Float): Scene` (no `val scene`), this can only have gone through
  `SceneLoader`'s `tryLoadAnimatedScene` path (`LoadedScene.scala:4-8`) — the `Animated`
  branch the prompt requires. Render saved to `poc-run/turn1-render.png`; the verdict log is
  saved verbatim at `poc-run/turn1-verdict.log` (captured on re-verification after the
  post-review comment fix below — same command, same result).
- **Turn 2** ("slower, and start it already at level 1"), same session, no restart: copied
  `turn1-generated.scala` to `poc-run/turn2-refined.scala` as a fixed "before" artifact, then
  edited **that file** in place — `Speed = 1f → 0.5f` and `level = progress * 3f → 1f +
  progress * 3f`. Never regenerated from the prompt. Diff (`poc-run/turn1-vs-turn2.diff`):
  exactly one hunk, touching two value expressions (`Speed`, `level`'s floor), both inside
  `scene(t)`'s local `val`s. Camera, material, lights, background, projection wiring —
  byte-identical. Verdict command re-run against the Turn 2 file → **exit 0**, same
  `Loading scene` / `Successfully loaded compiled scene` log pair, saved verbatim at
  `poc-run/turn2-verdict.log`. Render saved to `poc-run/turn2-render.png`.
- **Evidence saved under `poc-run/`:** `turn1-generated.scala`, `turn2-refined.scala`,
  `turn1-vs-turn2.diff`, `turn1-render.png`, `turn2-render.png`, `turn1-verdict.log`,
  `turn2-verdict.log`. Nothing was committed.

**Honest limitations of this run, stated up front:**
- Both renders are single un-denoised frames from a 5-bounce path tracer (no `--denoise`, no
  `--accumulation-frames`) — they show a dark background and a fractal cube silhouette that
  gets visibly more subdivided and rotates between the two turns, but neither frame clearly
  reads as "glass" (no obvious refraction/transmission in a still, noisy frame). This is a
  gap in *this evidence*, not a claim that the material assignment is wrong — the interactive,
  denoised render window is the correct instrument for that judgment and this PoC did not
  open it (out of scope per the story's Design Notes: render correctness is a manual check).
  Neither render was judged against `mvp-acceptance.md`'s full "recognisably the thing that
  was asked for" bar; this story doesn't require reaching it.
- **"Slower" was implemented in a way that truncates the loop rather than slowing it, and this
  is left uncorrected as a spike finding, not silently fixed.** `SceneLoader` maps `t` to
  `[0,1]` per rendered clip; real-time duration is controlled externally (CLI `--frames`/
  `--end-t`), outside the scene file. Scaling `progress = t * Speed` with `Speed = 0.5` doesn't
  make the clip take longer to play at the same `t` resolution — it caps the maximum `progress`
  reached at `t=1` to `0.5`, so the refined scene now ends at `level = 2.5` (not `4.0`) and
  `rotXW = 180°` (not `360°`): a shorter, less-intricate loop, not a slower one. This is a real
  wrinkle in mapping a plain-language "slower" onto a `t`-parameterised DSL scene whose actual
  playback speed is a separate CLI concern — worth carrying into whichever harness is chosen
  next (see criterion 6). The in-file comments were corrected post-review to state this
  accurately (level 1.0 -> 2.5, half turn) rather than the originally-generated but
  arithmetically wrong "1.0 -> 4.0, full turn."
- Both turns' compile/load verdicts were re-run after the post-review comment fix (comment-only
  edit, same code) and both still exit 0 with the same `SceneLoader` log pair — captured
  verbatim at `poc-run/turn1-verdict.log` and `poc-run/turn2-verdict.log`.
- Whether Turn 2's edit was made from Turn 1's content already retained in-session, or required
  a re-read from disk, is likewise self-reported by the executing session — no separate
  tool-call transcript was captured as independent evidence of that either.

## Criterion-by-criterion verdict

### 1. Can it be driven by a fixed, small tool set, with permissions constrained declaratively?

**Partially.** In actual use, this run touched a small, fixed set of tool *categories*: read
a reference file, write a new file, edit an existing file in place, and shell out to `sbt
run`/the compile-load verdict. That is close to the "roughly three tools" standing
hypothesis and is not falsified by the tool count observed here.

What this run could **not** demonstrate is declarative, skill-scoped permission
constraints. A Claude Code skill is a markdown instruction file interpreted by whatever
agent session invokes it — it does not carry its own manifest that narrows the *invoking
session's* tool access to "exactly these three tools, nothing else." Anything narrower than
the full ambient tool surface (arbitrary Bash, arbitrary file read/write anywhere on disk)
has to come from the surrounding session's `settings.json` allow/deny rules, which are
scoped to the session, not the skill. So: tool-count-wise, encouraging; permission-
declaration-wise, this mechanism does not give you what the criterion asks for on its own.

### 2. Does it support the clarification turn — multi-turn dialogue with retained state?

**Partially confirmed, but not the specific case named.** Turn 2 ran in the same
conversation as Turn 1 with no restart, and the edit to `turn2-refined.scala` was made using
the Turn 1 file's content already held in the conversation's own context — no re-read of
`turn1-generated.scala` from disk was needed to know what to change. That is real, native
multi-turn state retention, supplied by the ambient conversation transcript, not by anything
this skill built itself.

What was **not** exercised: an actual *clarification* turn, where the agent asks the user a
question back before proceeding (e.g. "glass at what IOR — plain Glass or GlassDispersive?").
Both `mvp-acceptance.md` prompts are direct instructions, not ambiguous ones, so this PoC
never had to test whether the harness supports a genuine back-and-forth clarification
exchange, only whether it retains state across two sequential instruction turns. It does.

### 3. Does it support session history sufficient for surgical-edit-and-history (CAP-3, CAP-6)?

**CAP-3: yes, demonstrated.** The Turn 2 diff is exactly one hunk touching two value
expressions (`Speed`, `level`'s floor); camera/material/lights/background are byte-identical,
matching the diff-minimality bar precisely — though see "Honest limitations" above: the *value*
Turn 2 landed on doesn't actually implement "slower" correctly, which is a DSL-semantics finding
distinct from the diff-shape question this criterion asks about.

**CAP-6: no, not demonstrated, and nothing here would provide it.** This PoC never
exercised restoring an earlier version. The only reason a distinct "Turn 1" artifact exists
at all (`poc-run/turn1-generated.scala` alongside `turn2-refined.scala`) is that the story
asked for it to be preserved for diffing — the skill/harness itself has no built-in
versioned-history store. Claude Code's own conversation transcript is not a queryable,
restorable scene-version store from the *rendered application's* point of view; "return to
any earlier version and continue from it" (SPEC.md CAP-6) would need a purpose-built
append-only version mechanism regardless of which harness is chosen — this criterion is not
something adopting a chat-turn-based harness gets you for free.

### 4. Can the user bring their own model or API key?

**No, not as tested.** This PoC ran inside Claude Code itself, so the "model" in play was
whatever the invoking Claude Code session was already authenticated against — there is no
separate model-selection surface exposed by the skill mechanism. A production version of
this harness, if built as a Claude Code skill, ties the product to Claude Code's own
model/auth story rather than letting a user plug in an arbitrary third-party model (a local
model, a different vendor's API) the way a self-written minimal loop could. This is a real,
observed constraint of adopting this particular harness, not a gap in the PoC's execution.

### 5. What language does it force on the new repo, and what does that cost against the workspace standards machinery?

**This PoC does not answer that question, and that is itself a finding.** Because the
"harness" here was Claude Code acting directly on the user's behalf, no new repo-resident
program had to be written at all — no Python, no TypeScript, nothing checked into
`menger-scene-agent` executes the agent loop. `menger-scene-agent` currently contains only a
skill (markdown), a manifest snapshot (JSON), and this run's evidence — nothing that
`bootstrap.sh sync`, `core.hooksPath`, or an sbt-based pre-push gate would have any opinion
about, because there is no source language yet.

That means the open question ("write our own vs. adopt pi/an SDK, and what does that cost
against the Scala-monoculture tooling") is **still open after this spike** — this PoC tested
the front half (can the MVP worked example be executed by *some* harness at all) but
deliberately used the one harness that sidesteps the language question entirely. Whatever
production harness gets chosen next (a written-from-scratch Python/TS loop, or an adopted
SDK) will still have to answer criterion 5 for real; this run only proves the *mechanics*
(compose from manifest + example, write, edit, shell to compile/load) are sound in
principle.

### 6. Build versus adapt: what is left to write after adopting it?

Substantial, regardless of adopt-vs-write. Concretely, not built or exercised by this PoC:

- **Permission scoping** (criterion 1) — a real declarative, narrow tool/permission
  boundary, not just "the invoking session happened to only need a few tools this time."
- **Version history / rollback** (CAP-6, criterion 3) — an append-only scene-version store
  outside all menger repos (SPEC.md AD-7/AD-10) does not exist yet in any form.
- **Model/provider abstraction** (criterion 4) — none; this run is Claude-Code-specific.
- **The rest of the validation gauntlet** (CAP-4) — this PoC only checked compile
  (`SceneCompiler`) and load (`SceneLoader`); lint, geometric checks, and semantic readback
  (`validation-gauntlet.md`) are entirely unbuilt and untested here, by design (story
  Boundaries: "Never: Building CAP-4's validation gauntlet").
- **Ticket escalation** (CAP-8) — nothing here detects an infeasible request and files a
  ticket instead of attempting one; that path was never triggered because both prompts were
  satisfiable from the manifest.
- **A fast dev loop.** Each compile/load verdict in this run paid a full `sbt` boot plus a
  CMake/CUDA native rebuild of `menger-geometry` (tens of seconds) before reaching the actual
  `SceneCompiler`/`SceneLoader` call. A production harness issuing many turns per session
  would need a persistent compile/load service or warmed JVM, not a fresh `sbt run` per turn.
- **A standalone, deployable process.** Claude Code is an interactive tool a maintainer runs
  locally; SPEC.md's audience includes "the artist" as a general user, not necessarily
  someone with Claude Code installed. Whatever harness is chosen for production needs to be
  something that can run as a service/CLI on its own, independent of Claude Code being open.

What **did** get proven, and is not nothing: composing an off-corpus scene (4D + animation +
glass + level-over-time, confirmed absent from the 28-scene corpus) from the manifest plus
one partial few-shot example worked on the first attempt with no compile or load failures,
and a subsequent surgical, in-place, two-property-only edit worked on the first attempt too.
The generation/refinement *mechanics* CAP-1/CAP-3 depend on are sound; everything else listed
above is harness engineering that remains, whichever direction open question 1 is ultimately
decided.

## Bottom line

The "minimal harness, three tools" hypothesis is **not falsified** by tool count or by the
generation/refinement mechanics, both of which worked cleanly in a single pass. It also is
**not confirmed** as sufficient on its own: declarative permission scoping, version history,
model portability, and the rest of the validation gauntlet all remain to be built no matter
which harness direction is chosen, and this particular front-half PoC (Claude Code as the
harness) cannot answer the language-cost question (criterion 5) at all, because it never had
to introduce a language into this repo in the first place.
