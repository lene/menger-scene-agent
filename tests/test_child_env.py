"""Review 2026-10-14, menger-scene-agent#27: no child process inherits a model credential."""

from __future__ import annotations

import subprocess
from pathlib import Path

from adapters.child_env import child_environment

SECRETS = {
    "ANTHROPIC_API_KEY": "x", "LITELLM_API_KEY": "x", "LITELLM_BASE_URL": "x",
    "GEMINI_API_KEY": "x", "DEEPSEEK_API_KEY": "x", "GITHUB_TOKEN": "x",
    "AWS_SECRET_ACCESS_KEY": "x", "OPENAI_APIKEY": "x",
}
KEPT = {"PATH": "/usr/bin", "HOME": "/home/u", "DISPLAY": ":0", "LANG": "C.UTF-8"}


def test_credentials_are_dropped_and_the_rest_is_kept():
    assert child_environment({**SECRETS, **KEPT}) == KEPT


def _captured_env(monkeypatch, launch) -> dict:
    """Runs `launch` with `subprocess.run`/`Popen` patched to record `env=` and fail fast."""
    captured: dict = {}

    def record(*_args, **kwargs):
        captured.update(kwargs)
        raise OSError("not launched in this test")

    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.setattr(subprocess, "Popen", record)
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    launch()
    return captured


def test_the_validator_and_the_render_window_get_no_credentials(monkeypatch, tmp_path: Path):
    from adapters.render_window import refresh_render_window
    from adapters.scene_validator import validate_scene

    scene = tmp_path / "scene.scala"
    scene.write_text("object S")
    script = tmp_path / "tool"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)

    for launch in (lambda: validate_scene(scene, script_path=script),
                   lambda: refresh_render_window(scene, launcher_path=script)):
        env = _captured_env(monkeypatch, launch)
        assert "env" in env and env["env"] is not None
        assert not set(SECRETS) & set(env["env"])
