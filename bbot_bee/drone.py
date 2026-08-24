"""Manages a single bbot scan running in a separate subprocess (cgroup-based failsafe termination)."""

from __future__ import annotations

from asyncio import (
    CancelledError,
    Task,
    TimerHandle,
    create_subprocess_exec,
    create_task,
    get_running_loop,
    shield,
    wait_for,
)
from asyncio.subprocess import PIPE, Process
from collections.abc import Awaitable, Callable
from contextlib import suppress
from logging import getLogger
from os import environ, getpgid, killpg
from re import compile as re_compile
from signal import SIGTERM, Signals
from sys import executable
from time import time
from typing import Any

from orjson import JSONDecodeError, dumps, loads
from swarm_common.models import ScanInfo, ScanStatus, scan_status_code

from bbot_bee.cgroup import ScanCgroup, reap_zombies

log = getLogger(__name__)
__all__ = ["Drone", "reap_zombies"]

_ANSI_ESCAPE_RE = re_compile(r"\x1b\[[0-9;]*m")
_BBOT_STATUS_MAP: dict[str, ScanStatus] = {s.value: s for s in ScanStatus}
_DEFAULT_GRACEFUL_STOP_TIMEOUT_S = 30.0
# Zombies still count as cgroup members until reaped.
_CGROUP_DRAIN_TIMEOUT_S = 10.0


class Drone:
    """Manages a single bbot scan running in a separate subprocess.

    Lifecycle: NOT_STARTED -> STARTING -> RUNNING -> FINISHING -> FINISHED.
    Stop path: ABORTING -> ABORTED. Force stop: SIGKILL to entire process
    group. Subprocess non-zero exit: FAILED.
    """

    def __init__(
        self,
        scan_id: str,
        preset: dict[str, Any],
        on_event: Callable[[str, dict[str, Any]], Awaitable[None]],
        on_status_change: Callable[[str, ScanStatus], Awaitable[None]],
        on_log_line: Callable[[str, str], Awaitable[None]],
        on_cmd_result: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        graceful_stop_timeout_s: float = _DEFAULT_GRACEFUL_STOP_TIMEOUT_S,
        _subprocess_script: str | None = None,
    ) -> None:
        """Initialize a Drone to manage a single scan subprocess.

        Args:
            scan_id: Unique identifier for this scan.
            preset: BBOT preset dict (target, modules, config, etc.).
            on_event: Async callback for scan events.
            on_status_change: Async callback for scan status transitions.
            on_log_line: Async callback for log lines from the subprocess.
            on_cmd_result: Async callback for command results from the subprocess.
            graceful_stop_timeout_s: Seconds to wait after SIGTERM before SIGKILL.
            _subprocess_script: Override subprocess command for testing (inline Python).
        """
        self._scan_id = scan_id
        self._preset = preset
        self._on_event = on_event
        self._on_status_change = on_status_change
        self._on_log_line = on_log_line
        self._on_cmd_result = on_cmd_result
        self._graceful_stop_timeout_s = graceful_stop_timeout_s
        self._subprocess_script = _subprocess_script
        self._status = ScanStatus.NOT_STARTED
        self._events_sent = 0
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._process: Process | None = None
        self._stdout_task: Task[None] | None = None
        self._stderr_task: Task[None] | None = None
        self._wait_task: Task[None] | None = None
        self._cgroup: ScanCgroup = ScanCgroup(scan_id)
        # Distinguishes an intentional kill (ABORTED) from a crash/OOM (FAILED) in _monitor_process.
        self._stopping: bool = False
        log.debug(f"Drone created: {scan_id=}, cgroup={self._cgroup.path}")

    @property
    def scan_id(self) -> str:
        """The scan's unique identifier."""
        return self._scan_id

    @property
    def status(self) -> ScanStatus:
        """Current scan status."""
        return self._status

    @property
    def events_sent(self) -> int:
        """Number of events collected from the scan."""
        return self._events_sent

    @property
    def started_at(self) -> float | None:
        """Timestamp when the scan was started."""
        return self._started_at

    @property
    def finished_at(self) -> float | None:
        """Timestamp when the scan reached a terminal state."""
        return self._finished_at

    @property
    def is_terminal(self) -> bool:
        """Whether the scan has reached a terminal state."""
        return self._status.is_terminal

    async def start(self) -> None:
        """Start the scan by spawning a subprocess running `scan_process.py`.

        Writes the preset JSON to stdin, then starts background tasks to read
        stdout (events/status), stderr (logs), and monitor the process for exit.
        """
        scan_id = self._scan_id
        log.info(f"{scan_id=} — spawning scan subprocess")

        payload = dumps({"scan_id": scan_id, "preset": self._preset})

        if self._subprocess_script is not None:
            cmd = [executable, "-c", self._subprocess_script]
            log.debug(f"{scan_id=} — using test subprocess script")
        else:
            cmd = [executable, "-m", "bbot_bee.scan_process"]

        # Cgroup must exist before spawn so the child can self-enroll via
        # $BBOT_BEE_SCAN_CGROUP before any forking.
        self._cgroup.create()
        spawn_env = {**environ, **self._cgroup.env}

        self._process = await create_subprocess_exec(
            *cmd,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            start_new_session=True,
            env=spawn_env,
        )
        pid = self._process.pid
        log.info(f"{scan_id=} — subprocess spawned with {pid=}")

        self._cgroup.populate(pid)

        # Write preset as first line to stdin (kept open for commands)
        assert self._process.stdin is not None
        self._process.stdin.write(payload + b"\n")
        await self._process.stdin.drain()
        log.log(5, f"{scan_id=} — stdin write: {len(payload)} bytes, first_100={payload[:100]!r}")
        log.debug(f"{scan_id=} — wrote {len(payload)} bytes to stdin (kept open for commands)")

        self._started_at = time()

        self._stdout_task = create_task(
            self._read_stdout(),
            name=f"drone-{scan_id}-stdout",
        )
        self._stderr_task = create_task(
            self._read_stderr(),
            name=f"drone-{scan_id}-stderr",
        )
        self._wait_task = create_task(
            self._monitor_process(),
            name=f"drone-{scan_id}-monitor",
        )
        log.info(f"{scan_id=} — background tasks started")

    async def wait(self) -> None:
        """Wait for the scan to complete. Returns when the subprocess exits."""
        if self._wait_task is not None:
            with suppress(CancelledError):
                await self._wait_task
            log.debug(f"{self._scan_id=} — subprocess completed")

    async def stop(self, force: bool = False) -> None:
        """Stop the scan by sending signals to the subprocess process group.

        Cooperative (default): SIGTERM -> wait `graceful_stop_timeout_s` ->
        SIGKILL. Force: SIGKILL via `cgroup.kill` immediately. No-op if the
        scan is already terminal.

        Args:
            force: If True, skip cooperative SIGTERM and go straight to the
                cgroup-level SIGKILL.
        """
        scan_id = self._scan_id
        log.info(f"{scan_id=}, {force=}, current_status={self._status.value}")
        self._stopping = True

        if self.is_terminal:
            log.info(f"{scan_id=} — already {self._status.value}, skipping")
            return

        if self._process is None or self._process.returncode is not None:
            log.debug(f"{scan_id=} — no running process to stop")
            return

        try:
            pgid = getpgid(self._process.pid)
        except ProcessLookupError:
            log.debug(f"{scan_id=} — process already exited")
            return

        if force:
            log.warning(f"{scan_id=} — force kill via cgroup.kill")
            self._cgroup.kill()
            await self._await_exit(timeout=5.0)
            if not self.is_terminal:
                await self._set_status(ScanStatus.ABORTED)
            return

        # SIGTERM first; then schedule a SIGKILL failsafe via loop.call_later.
        # The failsafe is a synchronous event-loop callback and is therefore
        # DECOUPLED from any coroutine's cancellation. If our caller gets
        # cancelled mid-wait (e.g. Queen.run cancelling _message_loop after
        # a heartbeat-driven reconnect), the timer still fires on schedule
        # and the subprocess cannot outlive the graceful window.
        log.info(f"{scan_id=} — sending SIGTERM to process group {pgid}")
        self._kill_process_group(pgid, SIGTERM)

        loop = get_running_loop()
        escalate_handle: TimerHandle = loop.call_later(
            self._graceful_stop_timeout_s,
            self._force_kill_runaway,
        )
        log.log(5, f"{scan_id=} — scheduled SIGKILL failsafe in {self._graceful_stop_timeout_s:.1f}s")

        try:
            # Wait slightly past the failsafe deadline so the two don't race
            # on the same tick; if the subprocess is still alive at timeout,
            # the failsafe is already firing and we want to observe the exit.
            await self._await_exit(timeout=self._graceful_stop_timeout_s + 1.0)
            # Calling .cancel() on an already-fired handle is a safe no-op.
            escalate_handle.cancel()
            log.info(f"{scan_id=} — cooperative stop completed")
        except TimeoutError:
            # Failsafe is firing now; wait briefly for the SIGKILL to land
            # and the status callback to flow.
            log.warning(f"{scan_id=} — graceful timeout elapsed; SIGKILL failsafe engaged")
            with suppress(CancelledError, TimeoutError):
                await self._await_exit(timeout=10.0)
            if not self.is_terminal:
                await self._set_status(ScanStatus.ABORTED)
        # CancelledError is intentionally NOT caught. If the caller is cancelled,
        # we re-raise so its cancellation semantics propagate. The escalate_handle
        # timer still fires on the event loop; the failsafe's own
        # _set_status(ABORTED) task ensures terminal status is recorded
        # regardless of whether our caller survived.

    async def send_command(self, cmd: dict[str, Any]) -> None:
        """Send a command to the running scan subprocess via stdin.

        Args:
            cmd: Command dict, e.g. {"cmd": "kill_module", "module_name": "http"}.

        Raises:
            RuntimeError: If the subprocess is not running.
        """
        scan_id = self._scan_id
        if self._process is None or self._process.stdin is None:
            raise RuntimeError(f"Cannot send command: subprocess not running for {scan_id}")
        if self._process.returncode is not None:
            raise RuntimeError(f"Cannot send command: subprocess already exited for {scan_id}")

        data = dumps(cmd) + b"\n"
        log.log(5, f"{scan_id=} — stdin write: cmd={cmd.get('cmd', '?')}, size={len(data)}, data={data[:200]!r}")
        try:
            self._process.stdin.write(data)
            await self._process.stdin.drain()
            log.debug(f"{scan_id=} — sent {cmd.get('cmd', '?')} ({len(data)} bytes)")
        except BrokenPipeError as exc:
            raise RuntimeError(f"Cannot send command: subprocess stdin broken for {scan_id}") from exc

    async def _read_stdout(self) -> None:
        """Read JSON lines from subprocess stdout and dispatch events/status."""
        scan_id = self._scan_id
        log.debug(f"{scan_id=} — starting stdout reader")

        assert self._process is not None and self._process.stdout is not None
        async for raw_line in self._process.stdout:
            line = raw_line.decode().strip()
            if not line:
                continue

            log.log(5, f"{scan_id=} — raw line: {line[:200]}")

            try:
                data = loads(line)
            except JSONDecodeError as exc:
                log.warning(f"{scan_id=} — invalid JSON: {exc}")
                continue

            msg_type = data.get("_type")

            if msg_type == "status":
                status_str = data.get("status", "")
                scan_status = _BBOT_STATUS_MAP.get(status_str)
                if scan_status is not None:
                    log.debug(f"{scan_id=} — status: {status_str}")
                    await self._set_status(scan_status)
                else:
                    log.warning(f"{scan_id=} — unknown status: {status_str}")

            elif msg_type == "event":
                data.pop("_type", None)
                self._events_sent += 1
                event_type = data.get("type", "unknown")
                log.debug(f"{scan_id=} — event: {event_type} (total={self._events_sent})")
                await self._on_event(scan_id, data)

            elif msg_type == "cmd_result":
                data.pop("_type", None)
                cmd_name = data.get("cmd", "?")
                success = data.get("success", "n/a")
                log.debug(f"{scan_id=} — cmd_result: {cmd_name} {success=}")
                if self._on_cmd_result is not None:
                    await self._on_cmd_result(scan_id, data)

            else:
                log.warning(f"{scan_id=} — unknown _type: {msg_type}")

        log.debug(f"{scan_id=} — stdout stream closed")

    async def _read_stderr(self) -> None:
        """Read log lines from subprocess stderr, strip ANSI codes, and forward.

        bbot colorizes its log output with ANSI escape codes — strip them
        before forwarding since the logs go to the hive, not a terminal.
        """
        scan_id = self._scan_id
        log.debug(f"{scan_id=} — starting stderr reader")

        assert self._process is not None and self._process.stderr is not None
        async for raw_line in self._process.stderr:
            line = _ANSI_ESCAPE_RE.sub("", raw_line.decode().rstrip("\n"))
            if line:
                log.debug(f"{scan_id=} — log: {line[:100]}")
                await self._on_log_line(scan_id, line)

        log.debug(f"{scan_id=} — stderr stream closed")

    async def _monitor_process(self) -> None:
        """Monitor the subprocess and set terminal status when it exits."""
        scan_id = self._scan_id
        log.debug(f"{scan_id=} — waiting for subprocess exit")

        assert self._process is not None
        pid = self._process.pid
        log.log(5, f"{scan_id=} — waiting for {pid=} to exit")
        await self._process.wait()
        returncode = self._process.returncode
        log.log(5, f"{scan_id=} — {pid=} exited, {returncode=}, current_status={self._status.value}")
        log.info(f"{scan_id=} — subprocess exited with {returncode=}")

        if self._process.stdin is not None and not self._process.stdin.is_closing():
            self._process.stdin.close()
            log.debug(f"{scan_id=} — stdin closed")

        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                with suppress(CancelledError):
                    await task
        log.debug(f"{scan_id=} — readers drained")

        self._cgroup.kill()
        self._cgroup.wait_empty(timeout_s=_CGROUP_DRAIN_TIMEOUT_S)
        self._cgroup.cleanup()
        reap_zombies()

        # Status messages on stdout should have already set this; handle crashes.
        if not self.is_terminal:
            if returncode == 0:
                log.debug(f"{scan_id=} — exit 0, setting FINISHED")
                await self._set_status(ScanStatus.FINISHED)
            elif returncode == 2:
                log.debug(f"{scan_id=} — exit 2, setting ABORTED")
                await self._set_status(ScanStatus.ABORTED)
            elif self._stopping and returncode is not None and returncode < 0:
                # Signal-killed during a stop = the abort we issued, not a crash.
                log.debug(f"{scan_id=} — exit {returncode} during stop, setting ABORTED")
                await self._set_status(ScanStatus.ABORTED)
            else:
                log.warning(f"{scan_id=} — exit {returncode}, setting FAILED")
                await self._set_status(ScanStatus.FAILED)

    def _kill_process_group(self, pgid: int, sig: Signals) -> None:
        """Send a signal to the entire process group. Safe if already exited."""
        scan_id = self._scan_id
        try:
            killpg(pgid, sig)
            log.debug(f"{scan_id=} — sent {sig.name} to pgid={pgid}")
        except ProcessLookupError:
            log.debug(f"{scan_id=} — pgid={pgid} already exited")

    def _force_kill_runaway(self) -> None:
        """Synchronous cgroup.kill failsafe invoked by `loop.call_later`.

        Runs as an event-loop callback (not an awaited coroutine), so it is
        immune to `task.cancel()` — guarantees the scan cannot outlive the
        graceful-stop window even if `stop()` is cancelled mid-wait.
        """
        if self._process is None or self._process.returncode is not None:
            log.log(5, f"{self._scan_id=} — already exited, no-op")
            return

        scan_id = self._scan_id
        log.warning(f"{scan_id=} — cgroup.kill escalation after {self._graceful_stop_timeout_s:.1f}s graceful window")
        self._cgroup.kill()

        # ABORTED (code 8) supersedes any FAILED (code 7) the monitor may
        # set from the SIGKILL returncode, via the forward-only ratchet.
        create_task(
            self._set_status(ScanStatus.ABORTED),
            name=f"drone-{scan_id}-escalate-status",
        )

    async def _await_exit(self, timeout: float) -> None:
        """Wait for the monitor task to complete with a timeout."""
        if self._wait_task is not None and not self._wait_task.done():
            await wait_for(shield(self._wait_task), timeout=timeout)

    async def _set_status(self, new_status: ScanStatus) -> None:
        """Update status (forward-only ratchet) and fire the status-change callback."""
        old = self._status
        old_code = scan_status_code(old)
        new_code = scan_status_code(new_status)
        scan_id = self._scan_id

        log.log(5, f"{scan_id=} — attempting {old.value}({old_code}) -> {new_status.value}({new_code})")

        if new_code <= old_code:
            log.log(
                5,
                f"{scan_id=} — REJECTED: {old.value}({old_code}) -> "
                f"{new_status.value}({new_code}), forward-only",
            )
            log.debug(
                f"{scan_id=} — ignoring {old.value}({old_code}) -> "
                f"{new_status.value}({new_code}), status can only advance forward"
            )
            return

        self._status = new_status
        if new_status.is_terminal:
            self._finished_at = time()
            log.log(5, f"{scan_id=} — terminal, finished_at={self._finished_at}")
        log.log(5, f"{scan_id=} — ACCEPTED: {old.value} -> {new_status.value}, firing callback")
        log.info(f"{scan_id=} — {old.value} -> {new_status.value}")
        await self._on_status_change(scan_id, new_status)

    def to_info(self) -> ScanInfo:
        """Return a ScanInfo summary of the current state."""
        return ScanInfo(
            scan_id=self._scan_id,
            status=self._status,
            events_sent=self._events_sent,
            started_at=self._started_at,
        )
