"""Text from the model, a session file or the render window, made safe for the terminal (review
2026-10-14, msa#42). An escape sequence in a model reply or in a line the scene wrote can move the
cursor, rewrite earlier output, set the window title, or (OSC 52) write the clipboard. Control
characters other than newline and tab are removed, and so are the bidirectional overrides that
make text read differently than it is; ESC goes with them, so no sequence survives."""

import re
from typing import TextIO

_UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f‪-‮⁦-⁩]")


def safe_text(text: str) -> str:
    return _UNSAFE.sub("", text)


class SafeStream:
    """A text stream that writes `safe_text` of everything; the REPL prints through one."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        return self._stream.write(safe_text(text))

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)
