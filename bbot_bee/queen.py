"""Queen — main agent that connects to the hive and manages drones.

The Queen is the central process in the bbot_bee package. It:
- Connects to the hive via WebSocket (3-layer reliable protocol)
- Receives scan commands (start_scan, stop_scan) from the hive
- Spawns Drone subprocesses for each scan
- Buffers events and logs from drones, flushes in batches to the hive
- Reports scan status changes immediately (CRITICAL priority)
- Sends state_sync on connect/reconnect
- Enforces max concurrent scan capacity
"""

from __future__ import annotations

from asyncio import CancelledError, Task, create_task, gather, sleep
from contextlib import suppress
from functools import partial
from logging import getLogger
from typing import Any

from swarm_common.channel import MessagePriority, ReliableChannel
from swarm_common.models import (
    BeeStatus,
    MessageType,
    ScanInfo,
    ScanStatus,
    StateSyncPayload,
    scan_status_code,
)
from swarm_common.protocol import make_message

from bbot_bee import cgroup as cgroup_module
from bbot_bee.config import BeeConfig
from bbot_bee.connection import ConnectionManager
from bbot_bee.drone import Drone, reap_zombies

log = getLogger(__name__)

# How often to re-send state_sync while connected.  The hive's diagnostic
# view of active scans (events_sent, started_at) freezes between state_sync
# messages, so this directly drives freshness of the hive's /diagnostics
# endpoint and the per-bee detail page in the UI.  Matches the UI's 10s
# live-refresh cadence.
STATE_SYNC_INTERVAL_S = 10.0


class Queen:
    """Main agent — connects to hive, manages drone subprocesses.

    Receives commands from the hive, spawns Drone instances for each scan,
    buffers events/logs, and reports status. Operates autonomously when the
    hive connection drops (drones keep running, events buffer).
    """

    def __init__(self, config: BeeConfig) -> None:
        """Initialize the Queen with configuration.

        Args:
            config: Drone configuration (hive URL, API key, capacity, etc.).

        Raises:
            RuntimeError: If cgroup v2 cgroup.kill is unavailable on this
                host. The bee refuses to boot without the kernel termination
                guarantee — no override.
        """
        if not cgroup_module.detect_cgroup_kill_supported():
            raise RuntimeError(
                "cgroup v2 cgroup.kill is required but not available — bee "
                "cannot start. Needs Linux 5.14+ with writable cgroup v2 "
                "(K8s 1.25+ with cgroupns=private + systemd cgroup driver).",
            )
        recovered = cgroup_module.recover_orphan_cgroups()
        if recovered:
            log.warning(f"__init__: recovered {recovered=} orphan cgroups from previous bee")

        self._config = config
        self._connection = ConnectionManager(
            url=config.hive_url,
            api_key=config.api_key,
            tls_verify=config.tls_verify,
        )
        self._channel = ReliableChannel(self._connection)

        # Drone management
        self._drones: dict[str, Drone] = {}
        self._init_max_scans = config.max_init_concurrent_scans
        self._max_concurrent_scans = config.max_init_concurrent_scans
        self._graceful_stop_timeout_s = config.graceful_stop_timeout_s

        # Event/log buffers — keyed by scan_id
        self._event_buffer: dict[str, list[dict[str, Any]]] = {}
        self._log_buffer: dict[str, list[str]] = {}

        # Background tasks tied to the current connection. Cancelled on
        # disconnect so they can be recreated on reconnect.
        self._tasks: list[Task[None]] = []

        # In-flight stop tasks, keyed by scan_id. These are SPAWNED from
        # the message loop rather than awaited in-line so that a
        # disconnect-triggered message_loop cancellation cannot cascade
        # into cancelling the escalation logic inside drone.stop().
        # Awaited in shutdown() before the queen tears down.
        self._pending_stops: dict[str, Task[None]] = {}

        log.debug(f"Queen initialized: bee_id={self.bee_id}, hive_url={config.hive_url}")

    # --- Properties -----------------------------------------------------------

    @property
    def bee_id(self) -> str:
        """The queen's unique identifier (used as bee_id in hive communication)."""
        return self._config.bee_id

    @property
    def available_capacity(self) -> int:
        """Number of additional scans that can be started."""
        return self._max_concurrent_scans - len(self._drones)

    # --- Main run loop --------------------------------------------------------

    async def run(self) -> None:
        """Main entry point — connect to hive with auto-reconnect.

        On disconnect the queen keeps drones running and buffers events.
        It reconnects with exponential backoff (1s base, 60s cap, 0.25 jitter)
        and sends a fresh state_sync on each reconnect.
        """
        log.debug(f"run: entry, bee_id={self.bee_id}, hive_url={self._config.hive_url}")
        log.info(f"run: starting queen {self.bee_id}")

        self._connection.on_connect = self._on_connect

        attempt = 0
        while True:
            self._connection.disconnected_event.clear()
            log.log(5, f"run: reconnect loop iteration — {attempt=}")

            try:
                log.info(f"run: connecting to hive at {self._config.hive_url}")
                await self._connection.connect()
                log.log(5, "run: connection established, resetting attempt counter")
                attempt = 0

                self._tasks = [
                    create_task(self._message_loop(), name=f"queen-{self.bee_id}-messages"),
                    create_task(self._event_flush_loop(), name=f"queen-{self.bee_id}-events"),
                    create_task(self._log_flush_loop(), name=f"queen-{self.bee_id}-logs"),
                    create_task(self._state_sync_loop(), name=f"queen-{self.bee_id}-state-sync"),
                    create_task(self._connection.run_heartbeat(), name=f"queen-{self.bee_id}-heartbeat"),
                ]
                log.log(5, f"run: background tasks created — {[t.get_name() for t in self._tasks]}")
                log.info(f"run: {len(self._tasks)} background tasks started")

                # Wait until disconnected (set by _mark_disconnected in ResilientWebSocket)
                await self._connection.disconnected_event.wait()
                log.warning("run: disconnect detected, will reconnect")

            except ConnectionError as exc:
                log.warning(f"run: connection failed — {exc}")
            except Exception as exc:
                log.error(f"run: unexpected error — {type(exc).__name__}: {exc}")

            # Cancel all background tasks before reconnecting
            active_tasks = [t for t in self._tasks if not t.done()]
            log.log(
                5,
                f"run: cancelling tasks — active={[t.get_name() for t in active_tasks]}, "
                f"done={[t.get_name() for t in self._tasks if t.done()]}",
            )
            log.debug(f"run: cancelling {len(active_tasks)} background tasks before reconnect")
            for task in self._tasks:
                if not task.done():
                    log.log(5, f"run: cancelling task {task.get_name()}")
                    task.cancel()
            for task in self._tasks:
                if not task.done():
                    with suppress(CancelledError):
                        await task
                    log.log(5, f"run: task {task.get_name()} cancelled and awaited")
            self._tasks.clear()
            log.debug("run: background tasks cancelled and cleared")

            # Ensure disconnected state
            await self._connection.disconnect()

            # Exponential backoff with jitter (delegated to ResilientWebSocket)
            sleep_time = self._connection.calc_backoff(attempt)
            log.log(5, f"run: reconnect backoff — {attempt=}, sleep={sleep_time:.2f}s")
            log.info(f"run: reconnecting in {sleep_time:.1f}s")
            await sleep(sleep_time)
            attempt += 1

    # --- Connection callbacks -------------------------------------------------

    async def _on_connect(self) -> None:
        """Called when the WebSocket connection is established or re-established."""
        log.info("_on_connect: connected to hive, sending state_sync")
        await self._send_state_sync()

    # --- State sync -----------------------------------------------------------

    def _build_state_sync(self) -> StateSyncPayload:
        """Build a StateSyncPayload from the current queen state."""
        active_scans = self.active_scans_info()
        available = self.available_capacity
        max_scans = self._max_concurrent_scans
        payload = StateSyncPayload(
            bee_id=self.bee_id,
            status=BeeStatus.ONLINE,
            active_scans=active_scans,
            capacity={
                "max_scans": max_scans,
                "init_max_scans": self._init_max_scans,
                "available": available,
            },
        )
        scan_count = len(active_scans)
        log.log(
            5,
            f"_build_state_sync: full payload — bee_id={self.bee_id}, status=ONLINE, "
            f"active_scans={dict(active_scans)}, capacity={{max_scans: {max_scans}, available: {available}}}, "
            f"drone_ids={list(self._drones.keys())}",
        )
        log.debug(f"_build_state_sync: {scan_count=}, {available=}/{max_scans}")
        return payload

    async def _send_state_sync(self) -> None:
        """Send the current queen state to the hive."""
        payload = self._build_state_sync()
        msg = make_message(MessageType.STATE_SYNC, payload.model_dump())
        await self._channel.send(msg, priority=MessagePriority.CRITICAL)
        log.info(f"_send_state_sync: state_sync sent with {len(payload.active_scans)} active scans")

    async def _state_sync_loop(self) -> None:
        """Periodically re-send state_sync so the hive's view of active scans stays fresh.

        Without this the hive only sees the bee's state from the initial
        connect — events_sent and started_at in ``bee_registry.active_scans``
        stay frozen at 0/None, and the UI's per-bee detail page shows stale
        counters regardless of how many events actually flowed.

        A final send on cancellation (disconnect) captures the latest counts
        so reconnect overlap doesn't lose the trailing update.
        """
        log.info(f"_state_sync_loop: starting ({STATE_SYNC_INTERVAL_S=:.1f}s)")
        while True:
            try:
                await sleep(STATE_SYNC_INTERVAL_S)
                await self._send_state_sync()
            except CancelledError:
                log.debug("_state_sync_loop: cancelled, sending final state_sync")
                with suppress(Exception):
                    await self._send_state_sync()
                break
            except Exception as exc:
                log.error(f"_state_sync_loop: error: {type(exc).__name__}: {exc}")

    # --- Drone management -----------------------------------------------------

    async def start_scan(
        self,
        scan_id: str,
        preset: dict[str, Any],
        name: str | None = None,
        _subprocess_script: str | None = None,
    ) -> None:
        """Start a new scan by spawning a Drone subprocess.

        Args:
            scan_id: Unique identifier for this scan.
            preset: BBOT preset dict (target, modules, config, etc.).
            name: Optional human-readable scan name.
            _subprocess_script: Override subprocess command for testing.

        Raises:
            RuntimeError: If at maximum capacity.
            ValueError: If scan_id already exists.
        """
        log.debug(f"start_scan: entry, {scan_id=}, {name=}, preset_keys={list(preset.keys())}")
        log.info(f"start_scan: {scan_id=}, {name=}")

        if scan_id in self._drones:
            log.error(f"start_scan: {scan_id=} already exists")
            raise ValueError(f"Scan already exists: {scan_id=}")

        if self.available_capacity <= 0:
            current = len(self._drones)
            max_scans = self._max_concurrent_scans
            log.error(f"start_scan: at capacity ({current}/{max_scans})")
            raise RuntimeError(f"At capacity: {current}/{max_scans} scans running")

        drone = Drone(
            scan_id=scan_id,
            preset=preset,
            on_event=self._handle_scan_event,
            on_status_change=self._handle_scan_status_change,
            on_log_line=self._handle_scan_log_line,
            on_cmd_result=self._handle_cmd_result,
            graceful_stop_timeout_s=self._graceful_stop_timeout_s,
            _subprocess_script=_subprocess_script,
        )
        self._drones[scan_id] = drone
        remaining = self.available_capacity
        log.debug(f"start_scan: {scan_id=} drone created, {remaining=} capacity")

        await drone.start()
        log.info(f"start_scan: {scan_id=} started successfully")

    async def stop_scan(self, scan_id: str, force: bool = False) -> None:
        """Stop a running scan.

        Args:
            scan_id: The scan to stop.
            force: If True, SIGKILL the subprocess immediately.

        Raises:
            KeyError: If scan_id is not tracked.
        """
        log.debug(f"stop_scan: entry, {scan_id=}, {force=}, tracked={scan_id in self._drones}")
        log.info(f"stop_scan: {scan_id=}, {force=}")

        if scan_id not in self._drones:
            log.warning(f"stop_scan: {scan_id=} not in tracking (may have already finished)")
            raise KeyError(f"Scan not found: {scan_id=}")

        drone = self._drones.get(scan_id)
        if drone is None:
            return

        await drone.stop(force=force)
        # _drones cleanup happens inside _handle_scan_status_change when the
        # drone transitions to a terminal state — no explicit pop needed here.
        remaining = self.available_capacity
        log.info(f"stop_scan: {scan_id=} stopped, {remaining=} capacity")

    def _on_stop_done(self, scan_id: str, task: Task[None]) -> None:
        """Done callback for fire-and-forget stop tasks spawned from the
        message loop. Clears tracking and surfaces any exception.
        """
        self._pending_stops.pop(scan_id, None)
        if task.cancelled():
            log.warning(f"_on_stop_done: stop task for {scan_id=} was cancelled")
            return
        exc = task.exception()
        if exc is None:
            log.debug(f"_on_stop_done: stop task for {scan_id=} completed")
            return
        if isinstance(exc, KeyError):
            log.warning(f"_on_stop_done: {scan_id=} not found (may have already finished)")
        else:
            log.error(f"_on_stop_done: stop task for {scan_id=} raised: {type(exc).__name__}: {exc}")

    async def stop_all(self) -> None:
        """Stop all running drones. Used during queen shutdown."""
        count = len(self._drones)
        log.debug(f"stop_all: entry, drone_count={count}")
        log.info(f"stop_all: stopping {count} drones")

        scan_ids = list(self._drones.keys())
        for scan_id in scan_ids:
            try:
                await self.stop_scan(scan_id)
            except Exception as exc:
                log.error(f"stop_all: error stopping {scan_id=}: {type(exc).__name__}: {exc}")

        log.info("stop_all: all drones stopped")

    def active_scans_info(self) -> dict[str, ScanInfo]:
        """Return ScanInfo for all active drones (for state_sync)."""
        log.debug(f"active_scans_info: entry, drone_count={len(self._drones)}")
        info = {scan_id: drone.to_info() for scan_id, drone in self._drones.items()}
        count = len(info)
        log.debug(f"active_scans_info: {count=} active drones")
        return info

    # --- Message processing ---------------------------------------------------

    async def _message_loop(self) -> None:
        """Receive messages from the hive and dispatch commands."""
        log.info(f"_message_loop: starting message receive loop for queen {self.bee_id}")
        while True:
            try:
                msg = await self._channel.recv()
                if msg is None:
                    continue

                msg_type = msg.type
                scan_id_hint = msg.payload.get("scan_id", "")
                log.log(
                    5,
                    f"_message_loop: received — type={msg_type.value}, msg_id={msg.msg_id}, "
                    f"scan_id={scan_id_hint!r}, payload_keys={list(msg.payload.keys())}",
                )
                log.debug(f"_message_loop: received {msg_type=}, msg_id={msg.msg_id}")

                if msg_type == MessageType.COMMAND:
                    await self._handle_command(msg.payload)
                else:
                    log.warning(f"_message_loop: unexpected message type: {msg_type}")

            except ConnectionError:
                log.warning("_message_loop: connection lost, exiting receive loop")
                break
            except CancelledError:
                log.debug("_message_loop: cancelled")
                break
            except Exception as exc:
                log.error(f"_message_loop: unexpected error: {type(exc).__name__}: {exc}")

        log.info("_message_loop: message loop exited")

    async def _handle_command(self, payload: dict[str, Any]) -> None:
        """Dispatch a command from the hive."""
        cmd = payload.get("cmd", "")
        log.info(f"_handle_command: {cmd=}")

        if cmd == "start_scan":
            scan_id = payload.get("scan_id", "")
            preset = payload.get("preset", {})
            name = payload.get("name")
            log.info(f"_handle_command: starting scan {scan_id=}, {name=}")
            try:
                await self.start_scan(scan_id, preset, name)
            except (RuntimeError, ValueError) as exc:
                log.error(f"_handle_command: failed to start scan: {exc}")
                # Report FAILED back to hive so it can re-dispatch or clean up
                await self._handle_scan_status_change(scan_id, ScanStatus.FAILED)

        elif cmd == "stop_scan":
            scan_id = payload.get("scan_id", "")
            force = payload.get("force", False)
            # Spawn as an independent task so the graceful-stop window
            # (up to graceful_stop_timeout_s) does not block the message
            # loop from processing other commands.  The task also
            # survives message_loop cancellation on reconnect.
            log.info(f"_handle_command: spawning stop task for {scan_id=}, {force=}")
            stop_task = create_task(
                self.stop_scan(scan_id, force=force),
                name=f"queen-{self.bee_id}-stop-{scan_id}",
            )
            self._pending_stops[scan_id] = stop_task
            stop_task.add_done_callback(partial(self._on_stop_done, scan_id))

        elif cmd in ("kill_module", "set_log_level", "toggle_log_level", "scan_status", "scope_check", "scan_config"):
            scan_id = payload.get("scan_id", "")
            log.info(f"_handle_command: forwarding {cmd} to drone {scan_id=}")
            if (drone := self._drones.get(scan_id)) is None:
                log.warning(f"_handle_command: {scan_id=} not found for {cmd}")
                return
            try:
                await drone.send_command(payload)
            except RuntimeError as exc:
                log.error(f"_handle_command: {cmd} failed for {scan_id=}: {exc}")

        elif cmd == "set_max_scans":
            new_max = int(payload["max_scans"])
            old = self._max_concurrent_scans
            self._max_concurrent_scans = new_max
            log.info(f"_handle_command: max_concurrent_scans {old} -> {new_max}")
            await self._send_state_sync()

        else:
            log.warning(f"_handle_command: unknown command: {cmd=}")

    # --- Scan callbacks -------------------------------------------------------

    async def _handle_scan_event(self, scan_id: str, event: dict[str, Any]) -> None:
        """Buffer a scan event for batch sending."""
        log.debug(f"_handle_scan_event: {scan_id=}, type={event.get('type', '?')}")
        if scan_id not in self._event_buffer:
            self._event_buffer[scan_id] = []
        self._event_buffer[scan_id].append(event)
        count = len(self._event_buffer[scan_id])
        log.debug(f"_handle_scan_event: {scan_id=} buffered event ({count=})")

    async def _handle_scan_status_change(self, scan_id: str, status: ScanStatus) -> None:
        """Send a scan status change to the hive immediately (critical priority)."""
        log.info(f"_handle_scan_status_change: {scan_id=}, {status=}")
        log.log(
            5,
            f"_handle_scan_status_change: enter — {scan_id=}, status={status.value}, "
            f"_drones_before={list(self._drones.keys())}",
        )
        code = scan_status_code(status)
        payload: dict[str, object] = {
            "scan_id": scan_id,
            "status": status.value,
            "status_code": code,
        }

        # Include timestamps from the drone so the hive can persist them
        drone = self._drones.get(scan_id)
        if drone is not None:
            if drone.started_at is not None:
                payload["started_at"] = drone.started_at
            if drone.finished_at is not None:
                payload["finished_at"] = drone.finished_at

        log.log(5, f"_handle_scan_status_change: full payload={payload}")

        # Remove drones BEFORE sending — the send may fail if the hive is
        # down, but the drone must be cleaned up regardless.  The channel
        # buffers unsent messages for replay on reconnect.
        if status.is_terminal and scan_id in self._drones:
            self._drones.pop(scan_id, None)
            remaining = self.available_capacity
            log.log(
                5,
                f"_handle_scan_status_change: {scan_id=} terminal ({status.value}), "
                f"popped from _drones, _drones_after={list(self._drones.keys())}, {remaining=}",
            )
            log.info(
                f"_handle_scan_status_change: {scan_id=} reached {status.value}, removed from tracking ({remaining=})"
            )

        msg = make_message(MessageType.SCAN_STATUS, payload)
        await self._channel.send(msg, priority=MessagePriority.CRITICAL)
        log.debug(f"_handle_scan_status_change: {scan_id=} status sent")

    async def _handle_scan_log_line(self, scan_id: str, line: str) -> None:
        """Buffer a scan log line for batch sending."""
        if scan_id not in self._log_buffer:
            self._log_buffer[scan_id] = []
        self._log_buffer[scan_id].append(line)
        count = len(self._log_buffer[scan_id])
        log.debug(f"_handle_scan_log_line: {scan_id=} buffered log line ({count=})")

    async def _handle_cmd_result(self, scan_id: str, data: dict[str, Any]) -> None:
        """Forward a command result from a drone subprocess to the hive."""
        cmd = data.get("cmd", "?")
        request_id = data.get("request_id", "")
        log.info(f"_handle_cmd_result: {scan_id=}, {cmd=}, {request_id=}")
        payload = {"scan_id": scan_id, **data}
        msg = make_message(MessageType.CMD_RESULT, payload)
        await self._channel.send(msg, priority=MessagePriority.CRITICAL)

    # --- Flush loops ----------------------------------------------------------

    async def _event_flush_loop(self) -> None:
        """Periodically flush buffered events as event_batch messages."""
        interval = self._config.event_flush_interval_s
        batch_size = self._config.event_batch_size
        log.info(f"_event_flush_loop: starting ({interval=:.1f}s, {batch_size=})")

        while True:
            try:
                await sleep(interval)
                await self._flush_events(batch_size)
                reap_zombies()
            except CancelledError:
                log.debug("_event_flush_loop: cancelled, flushing remaining events")
                await self._flush_events(batch_size)
                break
            except Exception as exc:
                log.error(f"_event_flush_loop: error: {type(exc).__name__}: {exc}")

    async def _log_flush_loop(self) -> None:
        """Periodically flush buffered logs as log_batch messages."""
        interval = self._config.log_flush_interval_s
        batch_size = self._config.log_batch_size
        log.info(f"_log_flush_loop: starting ({interval=:.1f}s, {batch_size=})")

        while True:
            try:
                await sleep(interval)
                await self._flush_logs(batch_size)
            except CancelledError:
                log.debug("_log_flush_loop: cancelled, flushing remaining logs")
                await self._flush_logs(batch_size)
                break
            except Exception as exc:
                log.error(f"_log_flush_loop: error: {type(exc).__name__}: {exc}")

    async def _flush_events(self, batch_size: int) -> None:
        """Flush all buffered events, sending in batches."""
        for scan_id in list(self._event_buffer.keys()):
            events = self._event_buffer.pop(scan_id, [])
            while events:
                batch = events[:batch_size]
                events = events[batch_size:]
                msg = make_message(MessageType.EVENT_BATCH, {"scan_id": scan_id, "events": batch})
                try:
                    await self._channel.send(msg, priority=MessagePriority.NORMAL)
                    count = len(batch)
                    log.debug(f"_flush_events: {scan_id=} flushed {count=} events")
                except ConnectionError:
                    remaining = batch + events
                    self._event_buffer[scan_id] = remaining
                    count = len(remaining)
                    log.warning(f"_flush_events: connection lost, re-buffered {count=} events for {scan_id=}")
                    break

    async def _flush_logs(self, batch_size: int) -> None:
        """Flush all buffered logs, sending in batches."""
        for scan_id in list(self._log_buffer.keys()):
            logs = self._log_buffer.pop(scan_id, [])
            while logs:
                batch = logs[:batch_size]
                logs = logs[batch_size:]
                msg = make_message(MessageType.LOG_BATCH, {"scan_id": scan_id, "lines": batch})
                try:
                    await self._channel.send(msg, priority=MessagePriority.NORMAL)
                    count = len(batch)
                    log.debug(f"_flush_logs: {scan_id=} flushed {count=} lines")
                except ConnectionError:
                    remaining = batch + logs
                    self._log_buffer[scan_id] = remaining
                    count = len(remaining)
                    log.warning(f"_flush_logs: connection lost, re-buffered {count=} lines for {scan_id=}")
                    break

    # --- Shutdown -------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully shut down the queen."""
        log.debug(
            f"shutdown: entry, bee_id={self.bee_id}, drone_count={len(self._drones)}, task_count={len(self._tasks)}",
        )
        log.info(f"shutdown: shutting down queen {self.bee_id}")

        log.info("shutdown: stopping all drones")
        await self.stop_all()

        # Drain any fire-and-forget stop tasks kicked off via the hive
        # command channel (these run outside stop_all's await chain).
        pending = list(self._pending_stops.values())
        if pending:
            log.info(f"shutdown: awaiting {len(pending)} in-flight stop task(s)")
            await gather(*pending, return_exceptions=True)

        log.info("shutdown: flushing remaining event/log buffers")
        try:
            await self._flush_events(self._config.event_batch_size)
            await self._flush_logs(self._config.log_batch_size)
        except Exception as exc:
            log.warning(f"shutdown: error flushing buffers: {type(exc).__name__}: {exc}")

        for task in self._tasks:
            if not task.done():
                task.cancel()
        log.debug(f"shutdown: cancelled {len(self._tasks)} background tasks")

        await self._connection.disconnect()

        with suppress(Exception):
            cgroup_module.recover_orphan_cgroups()
        log.info(f"shutdown: queen {self.bee_id} shutdown complete")
