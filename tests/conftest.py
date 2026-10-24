"""Every test runs with the real validator adapter wired into core.turn, as cli.py's main does
(core never imports it, msa#34); a test that fakes the validator replaces it as before."""

from __future__ import annotations

import pytest

import core.turn
from adapters.scene_validator import validate_scene


@pytest.fixture(autouse=True)
def _real_validator_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core.turn, "validate_scene", validate_scene)
