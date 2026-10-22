# Usability-script cases (msa#57)

Every request of the usability script (`docs/reviews/usability-scene-agent-session-4-script.md`
in menger-toplevel, used in sessions 4 and 5) with conditions on the scene it must produce. No
recorded responses, so they run only with a model:

```bash
uv run python -m evals.recorded --live --repeat 3                     # paid
uv run python -m evals.recorded --live --validator ../menger/docker/scene-validator/run-sandboxed.sh
```

A case is one REPL session; its turns build on each other. What a file cannot show (the window,
how it looks, REPL commands) is listed under `manual` and stays for the usability session.

## Turn keys

| Key | Meaning |
|---|---|
| `prompt` | the line the user types |
| `answer` | what the user answers if the model asks; threaded as the REPL does, once |
| `answered` | the prompt already carries an answer (`run_turn(answered=True)`) |
| `responses` | recorded model responses (only in `evals/cases/`) |

## `expect` keys

| Key | Condition |
|---|---|
| `questions_at_most` | questions the model asked in this turn (default 1) |
| `tag` | the turn's outcome, or a list of allowed outcomes (default `accepted`) |
| `message` | regexes the rejection messages must match (`nearest` for a decline) |
| `objects` / `no_objects` | object types present / absent |
| `object_count` | `{type: [min, max]}` |
| `materials`, `lights` | material presets / light types present |
| `levels` | `{type: [min, max]}`: some object of the type has a literal level in range |
| `above` | `[[upper, lower]]`: the first type's y is above the second's |
| `contains` / `not_contains` | regexes on the scene text (comments included, so `NOT DONE:` lines count) |
| `min_count` | `{regex: n}`: at least n matches |
| `changed` | regexes whose first match differs from the previous scene's |
| `camera_farther` | the camera is farther from its look-at point than before |
| `dark_background` | every background channel at most 0.15 |
