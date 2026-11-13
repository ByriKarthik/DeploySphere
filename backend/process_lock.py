"""Cross-platform single-process ownership for the DeploySphere backend."""

import errno
import os
from pathlib import Path


class BackendProcessLockError(RuntimeError):
    """The backend process lock could not be acquired or released."""


def default_process_lock_path() -> Path:
    """Return the per-user local path used to coordinate backend instances."""
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            raise BackendProcessLockError(
                "LOCALAPPDATA is not set; DeploySphere cannot determine its process-lock path."
            )
        base_directory = Path(local_app_data)
        if not base_directory.is_absolute():
            raise BackendProcessLockError(
                "LOCALAPPDATA must be an absolute path for the DeploySphere process lock."
            )
        return base_directory / "DeploySphere" / "backend.lock"

    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        base_directory = Path(state_home)
        if not base_directory.is_absolute():
            raise BackendProcessLockError(
                "XDG_STATE_HOME must be an absolute path for the DeploySphere process lock."
            )
    else:
        base_directory = Path.home() / ".local" / "state"
    return base_directory / "deploysphere" / "backend.lock"


def _is_lock_contention(error: OSError) -> bool:
    contention_errnos = {
        errno.EACCES,
        errno.EAGAIN,
        getattr(errno, "EDEADLK", errno.EACCES),
        getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
    }
    return error.errno in contention_errnos


class BackendProcessLock:
    """Hold a nonblocking OS file lock for the lifetime of a backend process."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_process_lock_path()
        self._file = None
        self._locked = False

    def acquire(self) -> None:
        if self._file is not None:
            raise BackendProcessLockError(
                "This DeploySphere process-lock instance already owns its lock."
            )

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_file = self.path.open("a+b")
        except OSError as error:
            raise BackendProcessLockError(
                f"Cannot open the DeploySphere process lock at {self.path}: {error}"
            ) from error

        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as error:
            try:
                lock_file.close()
            except OSError:
                pass
            if _is_lock_contention(error):
                raise BackendProcessLockError(
                    "Another DeploySphere backend process already owns the process "
                    f"lock at {self.path}. Stop it before launching another. All v1 "
                    "instances must use the same OS user and local lock path."
                ) from error
            raise BackendProcessLockError(
                f"Cannot acquire the DeploySphere process lock at {self.path}: {error}"
            ) from error
        except BaseException:
            lock_file.close()
            raise

        self._file = lock_file
        self._locked = True

    def release(self) -> None:
        lock_file = self._file
        if lock_file is None:
            return

        self._file = None
        was_locked = self._locked
        self._locked = False
        try:
            if was_locked:
                if os.name == "nt":
                    import msvcrt

                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def __enter__(self) -> "BackendProcessLock":
        self.acquire()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()
