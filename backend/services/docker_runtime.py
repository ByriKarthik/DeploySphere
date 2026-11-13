"""Safely manage DeploySphere-owned containers through the local Docker CLI.

This module contains no API or database code. Docker commands use a restricted
environment, a temporary credential-free CLI configuration, and only the
fixed options defined here. Timeouts after a process starts are reported as
indeterminate because stopping the CLI cannot guarantee daemon cancellation.
"""

from dataclasses import dataclass, field
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from _thread import LockType
from threading import Condition, Lock, Thread
from typing import Callable, Literal

from bson import ObjectId


IMAGE_TAG_PATTERN = re.compile(r"^deploysphere/build:[a-f0-9]{32}$")
CONTAINER_ID_PATTERN = re.compile(r"^[a-f0-9]{64}$")
CONTAINER_PORT_PATTERN = re.compile(r"^([1-9][0-9]{0,4})/tcp$")

MANAGED_LABEL = "com.deploysphere.managed"
PROJECT_LABEL = "com.deploysphere.project_id"
DEPLOYMENT_LABEL = "com.deploysphere.deployment_id"
EXPECTED_MANAGED_LABEL = "true"

RUNTIME_MEMORY = "512m"
RUNTIME_CPUS = "1.0"
RUNTIME_PIDS_LIMIT = "256"
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60
MAX_COMMAND_TIMEOUT_SECONDS = 300
TERMINATE_GRACE_SECONDS = 3
PIPE_READER_COMPLETION_TIMEOUT_SECONDS = 3
MAX_CAPTURE_BYTES = 64 * 1024
READ_CHUNK_BYTES = 16 * 1024
MAX_RESERVED_PIPE_CLEANUPS = 8

RuntimeOutcome = Literal["succeeded", "failed", "indeterminate"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeError:
    """Sanitized error information safe for a caller to store or display."""

    code: str
    message: str


@dataclass(frozen=True)
class RuntimeInspection:
    """A small, validated view of a container owned by DeploySphere."""

    container_id: str
    container_name: str
    state: str
    container_port: int | None = None
    allocated_port: int | None = None


@dataclass(frozen=True)
class StartupContainerObservation:
    """Raw read-only Docker fields needed for startup consistency checks."""

    container_id: str
    container_name: str
    managed_label: object
    project_id_label: object
    deployment_id_label: object
    state: object
    network_ports: object
    port_bindings: object
    identity_verified: bool


@dataclass(frozen=True)
class StartupContainerInspection:
    """Observation or a sanitized reason why one could not be obtained."""

    observation: StartupContainerObservation | None = None
    error: RuntimeError | None = None


@dataclass(frozen=True)
class RuntimeResult:
    """Outcome and safe metadata for one Docker operation."""

    operation: str
    outcome: RuntimeOutcome
    container_id: str | None = None
    container_name: str | None = None
    container_port: int | None = None
    allocated_port: int | None = None
    state: str | None = None
    owned: bool | None = None
    error: RuntimeError | None = None


@dataclass(frozen=True)
class _Identity:
    project_id: str
    deployment_id: str

    @property
    def container_name(self) -> str:
        return f"deploysphere-{self.deployment_id}"


@dataclass
class _DockerSession:
    executable: str
    endpoint: str
    config_directory: str
    environment: dict[str, str]
    temporary_directory: tempfile.TemporaryDirectory
    cleanup_deferred: bool = False

    def defer_cleanup_until_process_exit(
        self, process, readers, pipes, reservation: "_CleanupReservation"
    ) -> None:
        """Retain pipes until readers finish and config until the process exits."""
        transferred = _defer_pipe_cleanup(
            reservation,
            process,
            readers,
            pipes,
            on_complete=self.temporary_directory.cleanup,
        )
        self.cleanup_deferred = self.cleanup_deferred or transferred


@dataclass
class _Capture:
    data: bytearray
    total_bytes: int = 0
    truncated: bool = False
    read_error: bool = False
    _lock: LockType = field(default_factory=Lock, repr=False)

    def add(self, chunk: bytes) -> None:
        with self._lock:
            self.total_bytes += len(chunk)
            remaining = max(0, MAX_CAPTURE_BYTES - len(self.data))
            if remaining:
                self.data.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self.truncated = True

    def snapshot(self) -> tuple[bytes, bool, bool]:
        with self._lock:
            return bytes(self.data), self.truncated, self.read_error

    def mark_read_error(self) -> None:
        with self._lock:
            self.read_error = True


@dataclass
class _DeferredPipeCleanup:
    process: object
    readers: list[Thread]
    pipes: list
    reservation: "_CleanupReservation"
    on_process_exit: Callable[[], None] | None
    process_exited: bool = False
    retry_after: float = 0


@dataclass
class _CleanupReservation:
    state: Literal["reserved", "deferred", "released"] = "reserved"


class _DeferredPipeCleanupManager:
    """Bound deferred process resources without creating per-operation threads."""

    def __init__(self, maximum: int):
        self.maximum = maximum
        self._condition = Condition(Lock())
        self._reserved = 0
        self._pending: list[_DeferredPipeCleanup] = []
        self._worker: Thread | None = None

    def reserve(self) -> _CleanupReservation | None:
        with self._condition:
            if self._worker is None or not self._worker.is_alive():
                worker = Thread(target=self._run, daemon=True)
                try:
                    worker.start()
                except RuntimeError:
                    return None
                self._worker = worker
            if self._reserved >= self.maximum:
                return None
            self._reserved += 1
            return _CleanupReservation()

    def release(self, reservation: _CleanupReservation) -> None:
        with self._condition:
            if reservation.state == "reserved":
                reservation.state = "released"
                self._reserved -= 1
                self._condition.notify_all()

    def is_reserved(self, reservation: _CleanupReservation) -> bool:
        with self._condition:
            return reservation.state == "reserved"

    def defer(
        self,
        reservation: _CleanupReservation,
        process,
        readers: list[Thread],
        pipes: list,
        on_process_exit: Callable[[], None] | None = None,
    ) -> bool:
        with self._condition:
            if reservation.state == "deferred":
                return True
            if reservation.state != "reserved":
                return False
            reservation.state = "deferred"
            self._pending.append(
                _DeferredPipeCleanup(
                    process=process,
                    readers=readers,
                    pipes=pipes,
                    reservation=reservation,
                    on_process_exit=on_process_exit,
                )
            )
            self._condition.notify_all()
            return True

    @property
    def reserved_count(self) -> int:
        with self._condition:
            return self._reserved

    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._pending:
                    self._condition.wait()
                    continue
                pending = list(self._pending)

            completed = []
            for cleanup in pending:
                if time.monotonic() >= cleanup.retry_after:
                    try:
                        if not cleanup.process_exited:
                            cleanup.process_exited = cleanup.process.poll() is not None
                        if cleanup.process_exited and cleanup.on_process_exit is not None:
                            cleanup.on_process_exit()
                            cleanup.on_process_exit = None
                        if cleanup.process_exited and all(
                            reader.ident is None or not reader.is_alive()
                            for reader in cleanup.readers
                        ):
                            _close_pipes(cleanup.pipes)
                            completed.append(cleanup)
                    except Exception:
                        cleanup.retry_after = time.monotonic() + 1
                        logger.exception(
                            "Deferred Docker pipe cleanup failed; retaining its reservation."
                        )

            if completed:
                with self._condition:
                    for cleanup in completed:
                        if cleanup in self._pending:
                            self._pending.remove(cleanup)
                        if cleanup.reservation.state == "deferred":
                            cleanup.reservation.state = "released"
                            self._reserved -= 1
                    self._condition.notify_all()
            else:
                with self._condition:
                    self._condition.wait(timeout=0.05)


_DEFERRED_PIPE_CLEANUP_MANAGER = _DeferredPipeCleanupManager(
    MAX_RESERVED_PIPE_CLEANUPS
)


def _run_discovery(executable: str, command: list[str], environment: dict[str, str]) -> bytes:
    """Run read-only context discovery with bounded output and a fixed timeout."""
    reservation = _DEFERRED_PIPE_CLEANUP_MANAGER.reserve()
    if reservation is None:
        raise _SafeRuntimeFailure(
            "docker_unavailable", "Docker output cleanup capacity is exhausted."
        )

    stdout_capture = _Capture(bytearray())
    stderr_capture = _Capture(bytearray())
    process = None
    readers: list[Thread] = []

    def defer_cleanup() -> None:
        _defer_pipe_cleanup(reservation, process, readers, pipes)

    def settle_readers(process_exited: bool) -> bool:
        return _settle_pipe_readers(
            reservation,
            readers,
            pipes,
            process_exited=process_exited,
            defer=defer_cleanup,
        )

    pipes = []
    try:
        try:
            process = subprocess.Popen(
                command,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                bufsize=0,
            )
            pipes = [pipe for pipe in (process.stdout, process.stderr) if pipe is not None]
            if process.stdout is None or process.stderr is None:
                raise OSError("Docker context output was unavailable")
            readers = [
                Thread(target=_read_pipe, args=(process.stdout, stdout_capture), daemon=True),
                Thread(target=_read_pipe, args=(process.stderr, stderr_capture), daemon=True),
            ]
            for reader in readers:
                reader.start()
            return_code = process.wait(timeout=10)
        except KeyboardInterrupt:
            _cleanup_interrupted_process(process, settle_readers, defer_cleanup)
            raise
        except subprocess.TimeoutExpired as error:
            if process is not None:
                stopped = _terminate(process)
                settle_readers(stopped)
            raise _SafeRuntimeFailure(
                "docker_unavailable", "The selected local Docker context could not be read in time."
            ) from error
        except Exception as error:
            if process is not None:
                stopped = _terminate(process)
                settle_readers(stopped)
            raise _SafeRuntimeFailure(
                "docker_unavailable", "The selected local Docker context could not be read."
            ) from error

        if not settle_readers(True):
            raise _SafeRuntimeFailure(
                "docker_unavailable", "Docker context output could not be fully read."
            )

        _, stdout_truncated, stdout_read_error = stdout_capture.snapshot()
        _, stderr_truncated, stderr_read_error = stderr_capture.snapshot()
        if stdout_read_error or stderr_read_error:
            raise _SafeRuntimeFailure(
                "docker_unavailable", "Docker context output could not be read."
            )
        if stdout_truncated or stderr_truncated:
            raise _SafeRuntimeFailure(
                "docker_unavailable", "The selected Docker context returned too much data."
            )
        if return_code != 0:
            raise _SafeRuntimeFailure(
                "docker_unavailable", "The selected local Docker context is unavailable."
            )
        return bytes(stdout_capture.data)
    finally:
        if _DEFERRED_PIPE_CLEANUP_MANAGER.is_reserved(reservation):
            if process is None:
                _DEFERRED_PIPE_CLEANUP_MANAGER.release(reservation)
            else:
                defer_cleanup()


@dataclass(frozen=True)
class _CommandResult:
    outcome: RuntimeOutcome
    return_code: int | None
    stdout: bytes
    error: RuntimeError | None = None
    stderr: bytes = b""


def _normalize_object_id(value: str | ObjectId, field: str) -> str:
    """Return a canonical ObjectId string or reject the supplied identifier."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, str) and ObjectId.is_valid(value):
        return str(ObjectId(value))
    raise ValueError(f"{field} must be a valid ObjectId.")


def _identity(project_id: str | ObjectId, deployment_id: str | ObjectId) -> _Identity:
    return _Identity(
        project_id=_normalize_object_id(project_id, "project_id"),
        deployment_id=_normalize_object_id(deployment_id, "deployment_id"),
    )


def _validate_container_id(container_id: str) -> str:
    """Require the full ID format returned by Docker, not a name or prefix."""
    if not isinstance(container_id, str) or not CONTAINER_ID_PATTERN.fullmatch(container_id):
        raise ValueError("container_id must be a full Docker container ID.")
    return container_id


def _validate_port(container_port: int) -> int:
    """Accept only a TCP port represented by an integer in the valid range."""
    if isinstance(container_port, bool) or not isinstance(container_port, int):
        raise ValueError("container_port must be an integer TCP port.")
    if not 1 <= container_port <= 65535:
        raise ValueError("container_port must be from 1 to 65535.")
    return container_port


def _validate_timeout(timeout_seconds: int) -> int:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= MAX_COMMAND_TIMEOUT_SECONDS
    ):
        raise ValueError("timeout_seconds must be from 1 to 300.")
    return timeout_seconds


def _restricted_environment(config_directory: str | None = None) -> dict[str, str]:
    """Keep only process essentials and point Docker at an empty CLI config."""
    allowed = {
        "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "COMSPEC",
        "TEMP", "TMP",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    if config_directory:
        environment["DOCKER_CONFIG"] = config_directory
    return environment


def _discovery_environment() -> dict[str, str]:
    """Preserve selected local context hints only for endpoint discovery."""
    allowed = {
        "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "COMSPEC",
        "TEMP", "TMP", "DOCKER_CONFIG", "DOCKER_CONTEXT", "DOCKER_HOST",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def _resolve_local_endpoint(executable: str) -> str:
    """Resolve the selected context and reject remote or credentialed endpoints."""
    environment = _discovery_environment()
    endpoint_info: dict = {}
    selected_context = os.environ.get("DOCKER_CONTEXT")
    explicit_host = os.environ.get("DOCKER_HOST")

    if explicit_host and not selected_context:
        endpoint = explicit_host
        if os.environ.get("DOCKER_TLS_VERIFY") or os.environ.get("DOCKER_CERT_PATH"):
            raise _SafeRuntimeFailure("unsupported_docker_context", "TLS Docker endpoints are not supported.")
    else:
        if selected_context:
            context_name = selected_context
        else:
            context_output = _run_discovery(
                executable, [executable, "context", "show"], environment
            )
            context_name = context_output.decode("utf-8", errors="replace").strip()
        if not context_name or len(context_name) > 255 or any(character.isspace() for character in context_name) or context_name.startswith("-"):
            raise _SafeRuntimeFailure("docker_unavailable", "The selected Docker context is invalid.")
        output = _run_discovery(
            executable,
            [executable, "context", "inspect", context_name, "--format", "{{json .Endpoints.docker}}"],
            environment,
        )
        try:
            endpoint_info = json.loads(output.decode("utf-8", errors="replace"))
        except (TypeError, ValueError) as error:
            raise _SafeRuntimeFailure("docker_unavailable", "The selected Docker endpoint is invalid.") from error
        endpoint = endpoint_info.get("Host") if isinstance(endpoint_info, dict) else None

    if not isinstance(endpoint, str) or not endpoint:
        raise _SafeRuntimeFailure("docker_unavailable", "The selected Docker endpoint is invalid.")
    if endpoint.startswith(("ssh://", "tcp://")) or endpoint_info.get("SkipTLSVerify") is True:
        raise _SafeRuntimeFailure("unsupported_docker_context", "Only local Docker endpoints are supported.")
    if not endpoint.startswith(("npipe://", "unix://")):
        raise _SafeRuntimeFailure("unsupported_docker_context", "Only local Docker Desktop or Unix socket endpoints are supported.")
    return endpoint


class _SafeRuntimeFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class _SessionOwner:
    """Create one isolated config directory for a sequence of Docker calls."""

    def __enter__(self) -> _DockerSession:
        executable = shutil.which("docker")
        if executable is None:
            raise _SafeRuntimeFailure("docker_unavailable", "The Docker executable is not installed or not on PATH.")
        endpoint = _resolve_local_endpoint(executable)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="deploysphere-runtime-", ignore_cleanup_errors=True
        )
        try:
            config_directory = os.path.join(self._temporary.name, "docker-config")
            os.mkdir(config_directory)
            with open(os.path.join(config_directory, "config.json"), "w", encoding="utf-8") as config_file:
                config_file.write("{}")
            self._session = _DockerSession(
                executable=executable,
                endpoint=endpoint,
                config_directory=config_directory,
                environment=_restricted_environment(config_directory),
                temporary_directory=self._temporary,
            )
            return self._session
        except BaseException:
            self._temporary.cleanup()
            raise

    def __exit__(self, *_exc_info) -> None:
        if hasattr(self, "_session") and not self._session.cleanup_deferred:
            self._temporary.cleanup()
        elif not hasattr(self, "_session") and hasattr(self, "_temporary"):
            self._temporary.cleanup()


def _read_pipe(pipe, capture: _Capture) -> None:
    try:
        while True:
            chunk = pipe.read(READ_CHUNK_BYTES)
            if not chunk:
                return
            capture.add(chunk)
    except (OSError, ValueError):
        capture.mark_read_error()


def _close_pipes(pipes) -> None:
    for pipe in pipes:
        try:
            pipe.close()
        except (OSError, ValueError):
            pass


def _defer_pipe_cleanup(
    reservation: _CleanupReservation,
    process,
    readers,
    pipes,
    *,
    on_complete: Callable[[], None] | None = None,
) -> bool:
    """Transfer an already-reserved cleanup slot to the shared manager."""
    return _DEFERRED_PIPE_CLEANUP_MANAGER.defer(
        reservation,
        process,
        readers,
        pipes,
        on_process_exit=on_complete,
    )


def _settle_pipe_readers(
    reservation: _CleanupReservation,
    readers,
    pipes,
    *,
    process_exited: bool,
    defer: Callable[[], None],
) -> bool:
    """Close pipes after EOF, or transfer ownership to the bounded manager."""
    if process_exited:
        deadline = time.monotonic() + PIPE_READER_COMPLETION_TIMEOUT_SECONDS
        for reader in readers:
            if reader.ident is not None:
                reader.join(max(0, deadline - time.monotonic()))
        if all(not reader.is_alive() for reader in readers):
            _close_pipes(pipes)
            _DEFERRED_PIPE_CLEANUP_MANAGER.release(reservation)
            return True
    defer()
    return False


def _terminate(process) -> bool:
    """Stop and reap only the Docker CLI process, if the OS allows it."""
    try:
        process.terminate()
    except Exception:
        pass
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except Exception:
            return False
    except Exception:
        return False
    try:
        return process.poll() is not None
    except Exception:
        return False


def _cleanup_interrupted_process(process, settle_readers, defer_cleanup) -> None:
    """Make a bounded termination attempt before transferring interrupted work."""
    if process is None:
        return

    try:
        process_exited = _terminate(process)
    except BaseException:
        process_exited = False

    try:
        settle_readers(process_exited)
    except BaseException:
        try:
            defer_cleanup()
        except BaseException:
            logger.exception(
                "Could not transfer interrupted Docker process cleanup ownership."
            )


def _run_command(
    session: _DockerSession,
    arguments: list[str],
    *,
    timeout_seconds: int,
) -> _CommandResult:
    reservation = _DEFERRED_PIPE_CLEANUP_MANAGER.reserve()
    if reservation is None:
        return _CommandResult(
            "failed",
            None,
            b"",
            RuntimeError(
                "docker_cleanup_capacity",
                "Docker output cleanup capacity is exhausted.",
            ),
        )

    process = None
    readers: list[Thread] = []
    pipes = []

    try:
        command = [
            session.executable,
            "--config", session.config_directory,
            "--host", session.endpoint,
            *arguments,
        ]
        stdout_capture = _Capture(bytearray())
        stderr_capture = _Capture(bytearray())

        def defer_cleanup() -> None:
            session.defer_cleanup_until_process_exit(
                process, readers, pipes, reservation
            )

        def settle_readers(process_exited: bool) -> bool:
            return _settle_pipe_readers(
                reservation,
                readers,
                pipes,
                process_exited=process_exited,
                defer=defer_cleanup,
            )

        try:
            process = subprocess.Popen(
                command,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=session.environment,
                bufsize=0,
            )
            pipes = [pipe for pipe in (process.stdout, process.stderr) if pipe is not None]
            if process.stdout is None or process.stderr is None:
                raise OSError("Docker process output was unavailable")
            readers = [
                Thread(target=_read_pipe, args=(process.stdout, stdout_capture), daemon=True),
                Thread(target=_read_pipe, args=(process.stderr, stderr_capture), daemon=True),
            ]
            for reader in readers:
                reader.start()
            return_code = process.wait(timeout=timeout_seconds)
        except KeyboardInterrupt:
            _cleanup_interrupted_process(process, settle_readers, defer_cleanup)
            raise
        except subprocess.TimeoutExpired:
            stopped = _terminate(process) if process is not None else False
            if process is not None:
                settle_readers(stopped)
            return _CommandResult(
                "indeterminate",
                None,
                stdout_capture.snapshot()[0],
                RuntimeError("docker_outcome_indeterminate", "The Docker operation timed out; its daemon outcome is unknown."),
            )
        except FileNotFoundError:
            if process is not None:
                stopped = _terminate(process)
                settle_readers(stopped)
                return _CommandResult(
                    "indeterminate",
                    None,
                    stdout_capture.snapshot()[0],
                    RuntimeError("docker_outcome_indeterminate", "The Docker operation ended unexpectedly; its daemon outcome is unknown."),
                )
            return _CommandResult("failed", None, b"", RuntimeError("docker_unavailable", "The Docker executable could not be started."))
        except OSError:
            if process is not None:
                stopped = _terminate(process)
                settle_readers(stopped)
                return _CommandResult(
                    "indeterminate",
                    None,
                    stdout_capture.snapshot()[0],
                    RuntimeError("docker_outcome_indeterminate", "The Docker operation ended unexpectedly; its daemon outcome is unknown."),
                )
            return _CommandResult("failed", None, b"", RuntimeError("docker_unavailable", "The Docker command could not be started."))
        except Exception:
            if process is not None:
                stopped = _terminate(process)
                settle_readers(stopped)
                return _CommandResult(
                    "indeterminate",
                    None,
                    stdout_capture.snapshot()[0],
                    RuntimeError("docker_outcome_indeterminate", "The Docker operation ended unexpectedly; its daemon outcome is unknown."),
                )
            return _CommandResult("failed", None, b"", RuntimeError("docker_runtime_error", "The Docker operation could not be completed safely."))

        if not settle_readers(True):
            return _CommandResult(
                "indeterminate",
                return_code,
                stdout_capture.snapshot()[0],
                RuntimeError("docker_output_incomplete", "Docker exited before its output could be fully read."),
            )

        stdout, stdout_truncated, stdout_read_error = stdout_capture.snapshot()
        stderr, _, stderr_read_error = stderr_capture.snapshot()
        if return_code != 0:
            return _CommandResult(
                "failed",
                return_code,
                stdout,
                RuntimeError(
                    "docker_command_failed",
                    "Docker could not complete the requested operation.",
                ),
                stderr if not stderr_read_error else b"",
            )
        if stdout_read_error or stderr_read_error:
            return _CommandResult(
                "indeterminate",
                return_code,
                stdout,
                RuntimeError("docker_output_read_failed", "Docker output could not be read completely."),
            )
        if stdout_truncated and arguments[0] == "inspect":
            return _CommandResult("failed", return_code, stdout, RuntimeError("docker_output_too_large", "Docker returned more output than DeploySphere can safely process."))
        return _CommandResult("succeeded", return_code, stdout)
    finally:
        if _DEFERRED_PIPE_CLEANUP_MANAGER.is_reserved(reservation):
            if process is None:
                _DEFERRED_PIPE_CLEANUP_MANAGER.release(reservation)
            else:
                defer_cleanup()


def _result(
    operation: str,
    outcome: RuntimeOutcome,
    *,
    identity: _Identity | None = None,
    container_id: str | None = None,
    container_port: int | None = None,
    allocated_port: int | None = None,
    state: str | None = None,
    owned: bool | None = None,
    error: RuntimeError | None = None,
) -> RuntimeResult:
    return RuntimeResult(
        operation=operation,
        outcome=outcome,
        container_id=container_id,
        container_name=identity.container_name if identity else None,
        container_port=container_port,
        allocated_port=allocated_port,
        state=state,
        owned=owned,
        error=error,
    )


def _inspect_fields(session: _DockerSession, container_id: str, timeout_seconds: int) -> tuple[_CommandResult, dict | None]:
    """Read only the fields needed to verify ownership and loopback binding."""
    command_result = _run_command(
        session,
        [
            "inspect",
            "--type",
            "container",
            "--format",
            _CONTAINER_INSPECT_TEMPLATE,
            container_id,
        ],
        timeout_seconds=timeout_seconds,
    )
    if command_result.outcome != "succeeded":
        return command_result, None
    try:
        fields = command_result.stdout.decode("utf-8").splitlines()
        if len(fields) != 8:
            raise ValueError("unexpected inspection field count")
        parsed = [json.loads(field) for field in fields[:6]]
        ports = json.loads(fields[6])
        port_bindings = json.loads(fields[7])
        data = {
            "id": parsed[0],
            "name": parsed[1],
            "state": parsed[2],
            "managed": parsed[3],
            "project_id": parsed[4],
            "deployment_id": parsed[5],
            "ports": ports,
            "port_bindings": port_bindings,
        }
        return command_result, data
    except (UnicodeDecodeError, ValueError, TypeError):
        return _CommandResult("failed", 0, b"", RuntimeError("invalid_docker_response", "Docker returned invalid container inspection data.")), None


_CONTAINER_INSPECT_TEMPLATE = (
    '{{json .Id}}{{"\\n"}}{{json .Name}}{{"\\n"}}'
    '{{json .State.Status}}{{"\\n"}}{{json (index .Config.Labels "com.deploysphere.managed")}}{{"\\n"}}'
    '{{json (index .Config.Labels "com.deploysphere.project_id")}}{{"\\n"}}'
    '{{json (index .Config.Labels "com.deploysphere.deployment_id")}}{{"\\n"}}'
    '{{json .NetworkSettings.Ports}}{{"\\n"}}{{json .HostConfig.PortBindings}}'
)


def _inspect_failure(
    result: _CommandResult,
    reference: str,
) -> RuntimeError:
    """Classify only Docker inspection failures with unambiguous evidence."""
    error = result.error
    if error is None:
        return RuntimeError(
            "inspection_inconclusive",
            "Docker inspection did not return a usable result.",
        )
    if error.code == "docker_unavailable":
        return error
    if result.outcome == "indeterminate":
        return RuntimeError(
            "inspection_inconclusive",
            "Docker inspection timed out or its output could not be confirmed.",
        )
    if error.code == "docker_command_failed" and result.stderr:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        not_found_messages = {
            f"Error response from daemon: No such container: {reference}",
            f"Error: No such object: {reference}",
            f"Error: No such container: {reference}",
        }
        if message in not_found_messages:
            return RuntimeError(
                "container_not_found",
                "Docker did not find the requested container.",
            )
        lowered = message.lower()
        if (
            lowered.startswith("cannot connect to the docker daemon at ")
            and "is the docker daemon running?" in lowered
        ):
            return RuntimeError(
                "docker_unavailable",
                "The local Docker daemon is unavailable.",
            )
        if (
            lowered.startswith("error during connect:")
            and "docker_engine" in lowered
            and (
                "cannot find the file specified" in lowered
                or "the system cannot find" in lowered
            )
        ):
            return RuntimeError(
                "docker_unavailable",
                "The local Docker daemon is unavailable.",
            )
    return RuntimeError(
        "inspection_inconclusive",
        "Docker inspection failed without a reliably classifiable result.",
    )


def inspect_container_for_startup(
    *,
    container_id: str | None,
    container_name: str,
    project_id: str | ObjectId,
    deployment_id: str | ObjectId,
    timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> StartupContainerInspection:
    """Inspect one exact ID or deterministic name without changing Docker state."""
    identity = _identity(project_id, deployment_id)
    timeout = _validate_timeout(timeout_seconds)
    if container_name != identity.container_name:
        return StartupContainerInspection(
            error=RuntimeError(
                "identity_mismatch",
                "The stored container name does not match the deployment identity.",
            )
        )
    if container_id is not None:
        try:
            reference = _validate_container_id(container_id)
        except ValueError:
            return StartupContainerInspection(
                error=RuntimeError(
                    "missing_identity",
                    "The stored container ID is not a full Docker container ID.",
                )
            )
    else:
        reference = identity.container_name

    try:
        with _SessionOwner() as session:
            command_result = _run_command(
                session,
                [
                    "inspect",
                    "--type",
                    "container",
                    "--format",
                    _CONTAINER_INSPECT_TEMPLATE,
                    reference,
                ],
                timeout_seconds=timeout,
            )
            if command_result.outcome != "succeeded":
                return StartupContainerInspection(
                    error=_inspect_failure(command_result, reference)
                )

            try:
                fields = command_result.stdout.decode("utf-8").splitlines()
                if len(fields) > 8 and len(fields) % 8 == 0:
                    return StartupContainerInspection(
                        error=RuntimeError(
                            "ambiguous_lookup",
                            "Docker returned multiple containers for one inspection request.",
                        )
                    )
                if len(fields) != 8:
                    raise ValueError("unexpected inspection field count")
                parsed = [json.loads(field) for field in fields[:6]]
                observation_id = parsed[0]
                returned_name = parsed[1]
                if (
                    not isinstance(observation_id, str)
                    or not CONTAINER_ID_PATTERN.fullmatch(observation_id)
                    or not isinstance(returned_name, str)
                ):
                    raise ValueError("invalid container identity fields")
                if returned_name.startswith("/"):
                    returned_name = returned_name[1:]
                network_ports = json.loads(fields[6])
                port_bindings = json.loads(fields[7])
            except (UnicodeDecodeError, ValueError, TypeError):
                return StartupContainerInspection(
                    error=RuntimeError(
                        "inspection_inconclusive",
                        "Docker returned malformed container inspection data.",
                    )
                )

            expected_id = container_id if container_id is not None else observation_id
            data = {
                "id": observation_id,
                "name": returned_name,
                "managed": parsed[3],
                "project_id": parsed[4],
                "deployment_id": parsed[5],
            }
            return StartupContainerInspection(
                observation=StartupContainerObservation(
                    container_id=observation_id,
                    container_name=returned_name,
                    managed_label=parsed[3],
                    project_id_label=parsed[4],
                    deployment_id_label=parsed[5],
                    state=parsed[2],
                    network_ports=network_ports,
                    port_bindings=port_bindings,
                    identity_verified=_ownership_matches(
                        data, expected_id, identity
                    ),
                )
            )
    except _SafeRuntimeFailure as error:
        return StartupContainerInspection(
            error=RuntimeError(error.code, error.message)
        )
    except Exception:
        return StartupContainerInspection(
            error=RuntimeError(
                "inspection_inconclusive",
                "The container could not be inspected safely.",
            )
        )


def _ownership_matches(data: dict, container_id: str, identity: _Identity) -> bool:
    returned_name = data.get("name")
    if isinstance(returned_name, str) and returned_name.startswith("/"):
        returned_name = returned_name[1:]
    return (
        data.get("id") == container_id
        and returned_name == identity.container_name
        and data.get("managed") == EXPECTED_MANAGED_LABEL
        and data.get("project_id") == identity.project_id
        and data.get("deployment_id") == identity.deployment_id
    )


def _verified_inspection(
    session: _DockerSession,
    container_id: str,
    identity: _Identity,
    timeout_seconds: int,
    *,
    expected_container_port: int | None = None,
    require_dynamic_loopback_binding: bool = False,
    verify_network_ports: bool = True,
) -> tuple[RuntimeInspection | None, RuntimeError | None, RuntimeOutcome]:
    command_result, data = _inspect_fields(session, container_id, timeout_seconds)
    if command_result.outcome != "succeeded":
        return None, command_result.error, command_result.outcome
    if data is None or not _ownership_matches(data, container_id, identity):
        return None, RuntimeError("container_not_owned", "The container identity or DeploySphere ownership labels do not match."), "failed"
    state = data.get("state")
    if not isinstance(state, str) or not state:
        return None, RuntimeError("invalid_docker_response", "Docker returned invalid container state data."), "failed"
    if not verify_network_ports:
        return RuntimeInspection(
            container_id,
            identity.container_name,
            state,
        ), None, "succeeded"
    ports = data.get("ports")
    if not isinstance(ports, dict):
        return None, RuntimeError("invalid_docker_response", "Docker returned invalid port binding data."), "failed"
    if require_dynamic_loopback_binding:
        expected_key = f"{expected_container_port}/tcp"
        configured = data.get("port_bindings")
        if (
            not isinstance(configured, dict)
            or list(configured) != [expected_key]
        ):
            return None, RuntimeError("invalid_port_binding", "Docker did not configure exactly the requested TCP container port."), "failed"
        bindings = configured[expected_key]
        if not isinstance(bindings, list) or len(bindings) != 1:
            return None, RuntimeError("invalid_port_binding", "Docker did not configure exactly one loopback port binding."), "failed"
        binding = bindings[0]
        if (
            not isinstance(binding, dict)
            or binding.get("HostIp") != "127.0.0.1"
            or binding.get("HostPort") != ""
        ):
            return None, RuntimeError("invalid_port_binding", "Docker did not configure a dynamic loopback port binding."), "failed"
        return RuntimeInspection(
            container_id,
            identity.container_name,
            state,
            expected_container_port,
            None,
        ), None, "succeeded"

    if expected_container_port is not None and f"{expected_container_port}/tcp" not in ports:
        return None, RuntimeError("invalid_port_binding", "Docker did not retain the requested TCP container port."), "failed"
    if len(ports) > 1:
        return None, RuntimeError("invalid_port_binding", "Docker returned more than one published port mapping."), "failed"

    container_port: int | None = None
    allocated_port: int | None = None

    if ports:
        key, bindings = next(iter(ports.items()))
        match = CONTAINER_PORT_PATTERN.fullmatch(key)
        if match is None or not 1 <= int(match.group(1)) <= 65535:
            return None, RuntimeError("invalid_port_binding", "Docker returned an unexpected published port mapping."), "failed"
        container_port = int(match.group(1))

        if bindings is not None:
            if not isinstance(bindings, list) or len(bindings) != 1:
                return None, RuntimeError("invalid_port_binding", "Docker returned an unexpected published port mapping."), "failed"
            binding = bindings[0]
            host_port = binding.get("HostPort") if isinstance(binding, dict) else None
            if (
                not isinstance(binding, dict)
                or binding.get("HostIp") != "127.0.0.1"
                or not isinstance(host_port, str)
                or not host_port.isascii()
                or not host_port.isdigit()
                or not 1 <= int(host_port) <= 65535
            ):
                return None, RuntimeError("invalid_port_binding", "Docker returned a non-loopback or invalid port mapping."), "failed"
            allocated_port = int(host_port)

    return RuntimeInspection(container_id, identity.container_name, state, container_port, allocated_port), None, "succeeded"


def create_container(
    image_tag: str,
    project_id: str | ObjectId,
    deployment_id: str | ObjectId,
    container_port: int,
    *,
    timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> RuntimeResult:
    """Create, but do not start, one isolated container from a DeploySphere image."""
    if not isinstance(image_tag, str) or not IMAGE_TAG_PATTERN.fullmatch(image_tag):
        raise ValueError("image_tag must be a DeploySphere-generated build tag.")
    identity = _identity(project_id, deployment_id)
    port = _validate_port(container_port)
    timeout = _validate_timeout(timeout_seconds)
    create_accepted = False
    created_container_id = None
    try:
        with _SessionOwner() as session:
            labels = [
                "--label", f"{MANAGED_LABEL}={EXPECTED_MANAGED_LABEL}",
                "--label", f"{PROJECT_LABEL}={identity.project_id}",
                "--label", f"{DEPLOYMENT_LABEL}={identity.deployment_id}",
            ]
            command_result = _run_command(
                session,
                [
                    "create", "--name", identity.container_name,
                    *labels,
                    "--publish", f"127.0.0.1::{port}/tcp",
                    "--memory", RUNTIME_MEMORY,
                    "--cpus", RUNTIME_CPUS,
                    "--pids-limit", RUNTIME_PIDS_LIMIT,
                    "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges",
                    "--restart", "no",
                    image_tag,
                ],
                timeout_seconds=timeout,
            )
            if command_result.outcome != "succeeded":
                return _result("create", command_result.outcome, identity=identity, container_port=port, error=command_result.error)
            create_accepted = True
            returned_id = command_result.stdout.decode("ascii", errors="ignore").strip()
            created_container_id = returned_id if CONTAINER_ID_PATTERN.fullmatch(returned_id) else None
            if not CONTAINER_ID_PATTERN.fullmatch(returned_id):
                return _result(
                    "create", "indeterminate", identity=identity, container_port=port,
                    error=RuntimeError("invalid_docker_response", "Docker created a container but did not return a valid container ID."),
                )
            inspection, error, _outcome = _verified_inspection(
                session,
                returned_id,
                identity,
                timeout,
                expected_container_port=port,
                require_dynamic_loopback_binding=True,
            )
            if inspection is None:
                return _result(
                    # A successful create command means the container exists;
                    # failure to verify its mapping must not be reported as if
                    # no container was created.
                    "create", "indeterminate",
                    identity=identity, container_id=returned_id, container_port=port,
                    error=error or RuntimeError("docker_outcome_indeterminate", "The created container could not be verified."),
                )
            if inspection.state != "created" or inspection.container_port != port:
                return _result(
                    "create", "indeterminate", identity=identity,
                    container_id=returned_id, container_port=port,
                    error=RuntimeError("invalid_port_binding", "Docker did not confirm the requested created-container port mapping."),
                )
            return _result(
                "create", "succeeded", identity=identity,
                container_id=returned_id, container_port=port,
                allocated_port=inspection.allocated_port, state=inspection.state, owned=True,
            )
    except _SafeRuntimeFailure as error:
        return _result("create", "failed", identity=identity, container_port=port, error=RuntimeError(error.code, error.message))
    except (OSError, subprocess.SubprocessError):
        outcome = "indeterminate" if create_accepted else "failed"
        error_code = "docker_outcome_indeterminate" if create_accepted else "docker_unavailable"
        message = "The container may have been created, but its result could not be confirmed." if create_accepted else "The local Docker Engine is unavailable."
        return _result("create", outcome, identity=identity, container_id=created_container_id, container_port=port, error=RuntimeError(error_code, message))
    except Exception:
        outcome = "indeterminate" if create_accepted else "failed"
        error_code = "docker_outcome_indeterminate" if create_accepted else "docker_runtime_error"
        message = "The container may have been created, but its result could not be confirmed." if create_accepted else "The Docker container could not be created safely."
        return _result("create", outcome, identity=identity, container_id=created_container_id, container_port=port, error=RuntimeError(error_code, message))


def verify_deploysphere_ownership(
    container_id: str,
    project_id: str | ObjectId,
    deployment_id: str | ObjectId,
    *,
    timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> RuntimeResult:
    """Inspect and verify the exact ID, deterministic name, and ownership labels."""
    container_id = _validate_container_id(container_id)
    identity = _identity(project_id, deployment_id)
    timeout = _validate_timeout(timeout_seconds)
    try:
        with _SessionOwner() as session:
            inspection, error, outcome = _verified_inspection(
                session, container_id, identity, timeout
            )
            return _result(
                "inspect", outcome, identity=identity, container_id=container_id,
                container_port=inspection.container_port if inspection else None,
                allocated_port=inspection.allocated_port if inspection else None,
                state=inspection.state if inspection else None,
                owned=(False if error and error.code == "container_not_owned" else None) if error else True,
                error=error,
            )
    except _SafeRuntimeFailure as error:
        return _result("inspect", "failed", identity=identity, container_id=container_id, error=RuntimeError(error.code, error.message))
    except Exception:
        return _result("inspect", "failed", identity=identity, container_id=container_id, error=RuntimeError("docker_runtime_error", "The container could not be inspected safely."))


def inspect_container(
    container_id: str,
    project_id: str | ObjectId,
    deployment_id: str | ObjectId,
    *,
    timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> RuntimeResult:
    """Return current state only after full DeploySphere ownership verification."""
    return verify_deploysphere_ownership(
        container_id, project_id, deployment_id, timeout_seconds=timeout_seconds
    )


def _lifecycle_operation(
    operation: Literal["start", "stop", "restart"],
    container_id: str,
    project_id: str | ObjectId,
    deployment_id: str | ObjectId,
    *,
    timeout_seconds: int,
) -> RuntimeResult:
    container_id = _validate_container_id(container_id)
    identity = _identity(project_id, deployment_id)
    timeout = _validate_timeout(timeout_seconds)
    operation_accepted = False
    current: RuntimeInspection | None = None
    try:
        with _SessionOwner() as session:
            current, error, outcome = _verified_inspection(
                session, container_id, identity, timeout
            )
            if current is None:
                return _result(
                    operation, outcome, identity=identity, container_id=container_id,
                    owned=False if error and error.code == "container_not_owned" else None,
                    error=error,
                )
            if operation == "stop" and current.state != "running":
                return _result(
                    operation,
                    "failed",
                    identity=identity,
                    container_id=container_id,
                    container_port=current.container_port,
                    allocated_port=current.allocated_port,
                    state=current.state,
                    owned=True,
                    error=RuntimeError(
                        "unexpected_initial_state",
                        "Docker can stop this deployment only when its container is running.",
                    ),
                )
            command_result = _run_command(session, [operation, container_id], timeout_seconds=timeout)
            if command_result.outcome != "succeeded":
                return _result(
                    operation, command_result.outcome, identity=identity,
                    container_id=container_id, container_port=current.container_port,
                    allocated_port=current.allocated_port, state=current.state,
                    owned=True, error=command_result.error,
                )
            operation_accepted = True
            updated, inspect_error, _inspect_outcome = _verified_inspection(
                session, container_id, identity, timeout,
                expected_container_port=(
                    None if operation == "stop" else current.container_port
                ),
                verify_network_ports=operation != "stop",
            )
            if updated is None:
                return _result(
                    operation, "indeterminate", identity=identity, container_id=container_id,
                    container_port=current.container_port,
                    owned=None,
                    error=inspect_error or RuntimeError(
                        "docker_outcome_indeterminate",
                        "The Docker operation completed but its final state could not be verified.",
                    ),
                )
            expected_state = {
                "start": "running",
                "stop": "exited",
                "restart": "running",
            }[operation]
            if updated.state != expected_state:
                return _result(
                    operation, "indeterminate", identity=identity,
                    container_id=container_id,
                    container_port=updated.container_port or current.container_port,
                    allocated_port=updated.allocated_port, state=updated.state,
                    owned=True,
                    error=RuntimeError(
                        "unexpected_final_state",
                        "Docker accepted the operation but the final container state did not match the requested operation.",
                    ),
                )
            if operation in {"start", "restart"} and (
                updated.container_port is None or updated.allocated_port is None
            ):
                return _result(
                    operation, "indeterminate", identity=identity,
                    container_id=container_id,
                    container_port=updated.container_port or current.container_port,
                    state=updated.state, owned=True,
                    error=RuntimeError(
                        "port_binding_unavailable",
                        "The container is running but Docker has not confirmed its loopback TCP host port.",
                    ),
                )
            return _result(
                operation, "succeeded", identity=identity, container_id=container_id,
                container_port=updated.container_port or current.container_port,
                allocated_port=updated.allocated_port,
                state=updated.state, owned=True,
            )
    except _SafeRuntimeFailure as error:
        outcome = "indeterminate" if operation_accepted else "failed"
        failure = (
            RuntimeError("docker_outcome_indeterminate", "The container operation may have completed, but its result could not be confirmed.")
            if operation_accepted
            else RuntimeError(error.code, error.message)
        )
        return _result(
            operation, outcome, identity=identity, container_id=container_id,
            container_port=current.container_port if current else None,
            error=failure,
        )
    except Exception:
        outcome = "indeterminate" if operation_accepted else "failed"
        failure = (
            RuntimeError("docker_outcome_indeterminate", "The container operation may have completed, but its result could not be confirmed.")
            if operation_accepted
            else RuntimeError("docker_runtime_error", "The container operation could not be completed safely.")
        )
        return _result(
            operation, outcome, identity=identity, container_id=container_id,
            container_port=current.container_port if current else None,
            error=failure,
        )


def start_container(
    container_id: str,
    project_id: str | ObjectId,
    deployment_id: str | ObjectId,
    *,
    timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> RuntimeResult:
    """Start a verified DeploySphere container."""
    return _lifecycle_operation("start", container_id, project_id, deployment_id, timeout_seconds=timeout_seconds)


def stop_container(
    container_id: str,
    project_id: str | ObjectId,
    deployment_id: str | ObjectId,
    *,
    timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> RuntimeResult:
    """Stop a verified DeploySphere container without removing it."""
    return _lifecycle_operation("stop", container_id, project_id, deployment_id, timeout_seconds=timeout_seconds)


def restart_container(
    container_id: str,
    project_id: str | ObjectId,
    deployment_id: str | ObjectId,
    *,
    timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> RuntimeResult:
    """Restart a verified DeploySphere container without removing it."""
    return _lifecycle_operation("restart", container_id, project_id, deployment_id, timeout_seconds=timeout_seconds)
