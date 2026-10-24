"""The environment every child process gets: the agent's own minus anything credential-shaped.

The model adapters read their keys from the environment, and a child launched without `env=`
inherits all of it -- including the render window's JVM, which runs generated scene code
(review 2026-10-14, menger-scene-agent#27). Every launch passes `child_environment()` instead.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

# Names that hold, or point at, a credential: `ANTHROPIC_API_KEY`, `LITELLM_BASE_URL`,
# `GITHUB_TOKEN`, `AWS_SECRET_ACCESS_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, ...
_CREDENTIAL_NAME_RE = re.compile(
    r"(API_?KEY|_KEY$|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|_BASE_URL$|_AUTH)", re.IGNORECASE
)


def child_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """`environ` (default: this process's) without credential-shaped variables."""
    source = os.environ if environ is None else environ
    return {name: value for name, value in source.items() if not _CREDENTIAL_NAME_RE.search(name)}
