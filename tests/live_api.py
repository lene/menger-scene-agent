"""Gate for tests that call a real model API: they run only with MENGER_AGENT_LIVE_TESTS=1 and
the provider's key set. A key alone used to be enough, so every test run with a key in the
environment spent that vendor's credit (2026-10-14)."""

from __future__ import annotations

import os

import pytest

LIVE_TESTS_ENV = "MENGER_AGENT_LIVE_TESTS"


def skip_unless_live(*required_env: str) -> pytest.MarkDecorator:
    """Skip unless live tests are switched on and every variable in `required_env` is set."""
    missing = [name for name in required_env if not os.environ.get(name)]
    if os.environ.get(LIVE_TESTS_ENV) != "1":
        reason = f"{LIVE_TESTS_ENV}=1 not set -- skipping a test that calls a real model API"
    else:
        reason = f"{', '.join(missing)} not set -- skipping the real API integration test"
    return pytest.mark.skipif(os.environ.get(LIVE_TESTS_ENV) != "1" or bool(missing), reason=reason)
