"""Writing to disks that fill up.

The servers' data disks are shared and regularly reach 100%. A run must sit that out: nothing it writes may kill it,
nothing already on disk may be damaged, and writing starts again by itself once there is room. Three pieces:

- `quiet_stdio()`: stdout and stderr are log files on the same disks; a failed print is dropped instead of raised.
- `write_atomic(path, data)`: the file appears complete or not at all (temporary file, fsync, rename).
- `append_all(path, data)`: appends all of `data` or nothing; a partly written tail is cut off again.

All failures surface as OSError (ENOSPC, EDQUOT, EIO, a vanished mount); callers decide what to keep in memory.
"""
import os
import sys


class _QuietStream:
    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        try:
            return self._stream.write(text)
        except OSError:
            return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except OSError:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def quiet_stdio():
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is not None and not isinstance(stream, _QuietStream):
            setattr(sys, name, _QuietStream(stream))


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def write_atomic(path, data):
    """Write `data` (bytes-like) to `path` through `path + '.tmp'`. On failure the temporary file is removed, an
    existing `path` is untouched, and OSError is raised."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())      # some filesystems report a full disk only here
        os.replace(tmp, path)
    except OSError:
        remove_quietly(tmp)
        raise


def append_all(path, data):
    """Append `data` (bytes) to `path`, creating it. On failure the file is cut back to its earlier length and OSError
    is raised, so a caller that retries never leaves half a line behind. No fsync: this runs for every log row, and the
    local filesystems here reserve the space in write()."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        size = os.fstat(fd).st_size
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
        except OSError:
            try:
                os.ftruncate(fd, size)
            except OSError:
                pass
            raise
    finally:
        os.close(fd)
