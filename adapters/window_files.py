"""Files in the render window's directory (`<session>/window/`), as the agent on the host touches
them. That directory is mounted read-write into the window and export containers, where the
generated scene runs, so the scene can put anything under any name there: a symlink to a host
file, a FIFO, a directory. A host-side `open(path, "w")` through such a symlink created or
truncated a file of the scene's choosing outside the container, filled with the window's output,
which the scene also controls (review 2026-10-14, menger#88). Here nothing follows a symlink:
writes remove whatever has the name and create a new file, and reads take regular files only,
capped in size."""

import os
import stat
from contextlib import suppress
from pathlib import Path
from typing import TextIO, Union

_NOFOLLOW = os.O_NOFOLLOW | os.O_CLOEXEC
# Logs and state files are read whole and printed or parsed: a scene must not be able to make
# the agent read gigabytes.
READ_LIMIT_BYTES = 1 << 20


def _parent(path: Path) -> int:
    """The directory holding `path`, opened without following a symlink in its last component
    (`export/` is created inside the window directory, so the scene can replace it too)."""
    return os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)


def create(path: Union[str, Path]) -> TextIO:
    """A new, empty file at `path`, open for writing. Whatever had the name is removed, not
    followed; if something reappears under it before the create, `OSError` (FileExistsError)."""
    path = Path(path)
    dir_fd = _parent(path)
    try:
        with suppress(FileNotFoundError):
            os.unlink(path.name, dir_fd=dir_fd)
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o644,
                     dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    return os.fdopen(fd, "w")


def read_bytes(path: Union[str, Path], offset: int = 0, limit: int = READ_LIMIT_BYTES) -> bytes:
    """At most `limit` bytes of the regular file at `path` from `offset`; `b""` when it is
    missing, a symlink, or not a regular file (a FIFO would otherwise block the read)."""
    path = Path(path)
    try:
        dir_fd = _parent(path)
    except OSError:
        return b""
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | _NOFOLLOW, dir_fd=dir_fd)
    except OSError:
        return b""
    finally:
        os.close(dir_fd)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return b""
        handle.seek(offset)
        return handle.read(limit)


def read_text(path: Union[str, Path], limit: int = READ_LIMIT_BYTES) -> str:
    return read_bytes(path, 0, limit).decode(errors="replace")


def size(path: Union[str, Path]) -> int:
    """The size of the regular file at `path`, 0 for anything else."""
    with suppress(OSError):
        st = os.lstat(path)
        if stat.S_ISREG(st.st_mode):
            return st.st_size
    return 0


def is_file(path: Union[str, Path]) -> bool:
    """A regular file, not a symlink to one."""
    with suppress(OSError):
        return stat.S_ISREG(os.lstat(path).st_mode)
    return False
