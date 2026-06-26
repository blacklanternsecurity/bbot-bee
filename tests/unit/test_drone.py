"""Tests for bbot_bee.drone — Drone subprocess lifecycle manager.

Drone manages one bbot scan running in a separate Python subprocess.
Tests use mock subprocesses (inline Python scripts via -c) for speed.
Real bbot integration is tested in test_scan_process.py and E2E tests.

The autouse fixture in ``tests/conftest.py`` swaps ``ScanCgroup`` for
``_FakeScanCgroup`` so tests don't touch ``/sys/fs/cgroup``.
"""

from __future__ import annotations

import asyncio
import textwrap
from unittest.mock import AsyncMock

from swarm_common.models import ScanInfo, ScanStatus, scan_status_code

from bbot_bee.drone import Drone
from tests.conftest import _FakeScanCgroup

# ---------------------------------------------------------------------------
# Mock subprocess scripts (inline Python via -c)
# ---------------------------------------------------------------------------

# Successful scan: emits statuses and an event, exits 0
_MOCK_SUCCESS_SCRIPT = textwrap.dedent("""\
    import json, sys, time
    config = json.loads(sys.stdin.readline())
    scan_id = config["scan_id"]
    for status, code in [("STARTING", 2), ("RUNNING", 3), ("FINISHING", 4), ("FINISHED", 6)]:
        print(json.dumps({"_type": "status", "status": status, "status_code": code}), flush=True)
        time.sleep(0.02)
    print(json.dumps({"_type": "event", "type": "DNS_NAME", "data": "example.com", "scan": scan_id}), flush=True)
    print("a log line on stderr", file=sys.stderr, flush=True)
    print("\\033[1;38;5;69m[INFO]\\033[0m colorized bbot log", file=sys.stderr, flush=True)
    sys.exit(0)
""")

# Crash script: exits with code 1 after partial output
_MOCK_CRASH_SCRIPT = textwrap.dedent("""\
    import json, sys
    config = json.loads(sys.stdin.readline())
    print(json.dumps({"_type": "status", "status": "STARTING", "status_code": 2}), flush=True)
    sys.exit(1)
""")

# Slow script: runs until SIGTERM, then exits 2
_MOCK_SLOW_SCRIPT = textwrap.dedent("""\
    import json, signal, sys, time
    config = json.loads(sys.stdin.readline())
    print(json.dumps({"_type": "status", "status": "STARTING", "status_code": 2}), flush=True)
    print(json.dumps({"_type": "status", "status": "RUNNING", "status_code": 3}), flush=True)
    sys.stdout.flush()
    stopping = False
    def _handle_sigterm(sig, frame):
        global stopping
        stopping = True
        print(json.dumps({"_type": "status", "status": "ABORTING", "status_code": 5}), flush=True)
        print(json.dumps({"_type": "status", "status": "ABORTED", "status_code": 8}), flush=True)
    signal.signal(signal.SIGTERM, _handle_sigterm)
    while not stopping:
        time.sleep(0.1)
    sys.exit(2)
""")

# Hang script: ignores SIGTERM, must be SIGKILL'd
_MOCK_HANG_SCRIPT = textwrap.dedent("""\
    import json, signal, sys, time
    config = json.loads(sys.stdin.readline())
    print(json.dumps({"_type": "status", "status": "STARTING", "status_code": 2}), flush=True)
    print(json.dumps({"_type": "status", "status": "RUNNING", "status_code": 3}), flush=True)
    sys.stdout.flush()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.1)
""")

# Preset used for all mock tests (doesn't matter — mock scripts ignore it)
_MOCK_PRESET = {"target": ["127.0.0.1"], "modules": []}


class TestDroneInit:
    """Tests for Drone construction."""

    def test_initial_status(self) -> None:
        """New Drone should start in NOT_STARTED status."""
        drone = Drone(
            scan_id="scan-001",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
        )
        assert drone.status == ScanStatus.NOT_STARTED
        assert drone.scan_id == "scan-001"
        assert drone.events_sent == 0
        assert drone.started_at is None

    def test_is_terminal_initially_false(self) -> None:
        """New Drone should not be in terminal state."""
        drone = Drone(
            scan_id="s1",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
        )
        assert drone.is_terminal is False


class TestDroneLifecycle:
    """Tests for the subprocess lifecycle (start, wait, status progression)."""

    async def test_start_and_wait_success(self) -> None:
        """Drone should start a subprocess, collect output, and reach FINISHED."""
        statuses: list[ScanStatus] = []

        async def _on_status(scan_id: str, status: ScanStatus) -> None:
            statuses.append(status)

        drone = Drone(
            scan_id="scan-lifecycle",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=_on_status,
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        await drone.start()
        await drone.wait()

        assert drone.status == ScanStatus.FINISHED
        assert drone.is_terminal
        assert ScanStatus.STARTING in statuses
        assert ScanStatus.RUNNING in statuses
        assert ScanStatus.FINISHED in statuses

    async def test_events_dispatched(self) -> None:
        """Events from the subprocess should be dispatched to the on_event callback."""
        events: list[dict] = []

        async def _on_event(scan_id: str, event: dict) -> None:
            events.append(event)

        drone = Drone(
            scan_id="scan-events",
            preset=_MOCK_PRESET,
            on_event=_on_event,
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        await drone.start()
        await drone.wait()

        assert len(events) >= 1
        assert events[0]["type"] == "DNS_NAME"
        assert events[0]["data"] == "example.com"
        # _type discriminator should be stripped before dispatch
        assert "_type" not in events[0]

    async def test_events_sent_counter(self) -> None:
        """events_sent should count dispatched events."""
        drone = Drone(
            scan_id="scan-count",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        await drone.start()
        await drone.wait()

        assert drone.events_sent >= 1

    async def test_log_lines_dispatched(self) -> None:
        """Log lines from stderr should be dispatched to on_log_line."""
        log_lines: list[str] = []

        async def _on_log(scan_id: str, line: str) -> None:
            log_lines.append(line)

        drone = Drone(
            scan_id="scan-logs",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=_on_log,
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        await drone.start()
        await drone.wait()

        assert any("a log line on stderr" in line for line in log_lines)

    async def test_ansi_codes_stripped_from_logs(self) -> None:
        """ANSI escape codes from bbot's colorized output should be stripped."""
        log_lines: list[str] = []

        async def _on_log(scan_id: str, line: str) -> None:
            log_lines.append(line)

        drone = Drone(
            scan_id="scan-ansi",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=_on_log,
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        await drone.start()
        await drone.wait()

        colorized = [line for line in log_lines if "colorized bbot log" in line]
        assert len(colorized) == 1, f"Expected 1 colorized log line, got {colorized}"
        # Should be stripped clean — no ANSI escape sequences
        assert colorized[0] == "[INFO] colorized bbot log"
        assert "\033" not in colorized[0]

    async def test_status_codes_advance_forward(self) -> None:
        """Status codes should only increase (forward-only progression)."""
        statuses: list[ScanStatus] = []

        async def _on_status(scan_id: str, status: ScanStatus) -> None:
            statuses.append(status)

        drone = Drone(
            scan_id="scan-forward",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=_on_status,
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        await drone.start()
        await drone.wait()

        codes = [scan_status_code(s) for s in statuses]
        for i in range(1, len(codes)):
            assert codes[i] > codes[i - 1], f"Status regressed: {statuses[i - 1]} -> {statuses[i]}"

    async def test_started_at_set(self) -> None:
        """started_at should be set after start()."""
        drone = Drone(
            scan_id="scan-time",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        assert drone.started_at is None
        await drone.start()
        assert drone.started_at is not None
        await drone.wait()


class TestDroneCrash:
    """Tests for subprocess crash detection."""

    async def test_crash_sets_failed(self) -> None:
        """A subprocess crash (exit code 1) should set status to FAILED."""
        statuses: list[ScanStatus] = []

        async def _on_status(scan_id: str, status: ScanStatus) -> None:
            statuses.append(status)

        drone = Drone(
            scan_id="scan-crash",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=_on_status,
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_CRASH_SCRIPT,
        )
        await drone.start()
        await drone.wait()

        assert drone.status == ScanStatus.FAILED
        assert drone.is_terminal


class TestDroneStop:
    """Tests for stopping a drone (cooperative and forced)."""

    async def test_cooperative_stop(self) -> None:
        """stop() should send SIGTERM and wait for graceful exit."""
        statuses: list[ScanStatus] = []

        async def _on_status(scan_id: str, status: ScanStatus) -> None:
            statuses.append(status)

        drone = Drone(
            scan_id="scan-stop",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=_on_status,
            on_log_line=AsyncMock(),
            graceful_stop_timeout_s=5.0,
            _subprocess_script=_MOCK_SLOW_SCRIPT,
        )
        await drone.start()
        # Wait for scan to reach RUNNING
        for _ in range(50):
            if drone.status == ScanStatus.RUNNING:
                break
            await asyncio.sleep(0.1)
        assert drone.status == ScanStatus.RUNNING

        await drone.stop()
        assert drone.is_terminal
        assert ScanStatus.ABORTED in statuses or ScanStatus.ABORTING in statuses

    async def test_force_stop(self) -> None:
        """stop(force=True) should atomically kill the cgroup."""
        drone = Drone(
            scan_id="scan-force",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_HANG_SCRIPT,
        )
        await drone.start()
        # Wait for scan to reach RUNNING
        for _ in range(50):
            if drone.status == ScanStatus.RUNNING:
                break
            await asyncio.sleep(0.1)

        await drone.stop(force=True)
        assert drone.is_terminal
        # cgroup.kill must have been invoked at least once on the force path
        kill_calls = [e for e in _FakeScanCgroup.instances[0].events if e[0] == "kill"]
        assert len(kill_calls) >= 1, f"force stop must call cgroup.kill; events={_FakeScanCgroup.instances[0].events}"

    async def test_stop_escalates_to_sigkill(self) -> None:
        """Cooperative stop should escalate to SIGKILL after timeout."""
        drone = Drone(
            scan_id="scan-escalate",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            graceful_stop_timeout_s=0.5,  # Very short timeout to test escalation
            _subprocess_script=_MOCK_HANG_SCRIPT,
        )
        await drone.start()
        for _ in range(50):
            if drone.status == ScanStatus.RUNNING:
                break
            await asyncio.sleep(0.1)

        await drone.stop()  # Should SIGTERM, timeout, then SIGKILL
        assert drone.is_terminal

    async def test_stop_already_finished_is_noop(self) -> None:
        """Stopping an already-finished drone should be a no-op."""
        drone = Drone(
            scan_id="scan-noop",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        await drone.start()
        await drone.wait()
        assert drone.is_terminal

        final_status = drone.status
        await drone.stop()
        assert drone.status == final_status

    async def test_stop_not_started_is_noop(self) -> None:
        """Stopping a drone that was never started should not raise."""
        drone = Drone(
            scan_id="scan-never",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
        )
        await drone.stop()  # Should not raise


class TestDroneCgroupLifecycle:
    """Pins the cgroup integration contract:

    - start() creates the cgroup, then populates it with the subprocess PID
      *before* the preset is written to stdin.
    - The subprocess inherits ``BBOT_BEE_SCAN_CGROUP`` so it can self-enroll.
    - The SIGKILL escalation path uses cgroup.kill (synchronous, no await).
    - _monitor_process cleans up: kill → wait_empty → cleanup.
    - Concurrent drones get isolated cgroups — one drone's kill cannot
      touch another's procs.
    """

    async def test_start_creates_then_populates_cgroup(self) -> None:
        """create() must run before populate(), and populate(pid) must use
        the actual subprocess PID.
        """
        drone = Drone(
            scan_id="scan-cg-start",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        await drone.start()
        cg = _FakeScanCgroup.instances[0]
        names = [e[0] for e in cg.events]
        assert names[:2] == ["create", "populate"], f"events={cg.events}"
        # The populated PID must equal the spawned subprocess's PID.
        populate_event = next(e for e in cg.events if e[0] == "populate")
        assert populate_event[1] == drone._process.pid
        await drone.wait()

    async def test_monitor_kills_waits_and_cleans_up(self) -> None:
        """After subprocess exit, the lifecycle hook in _monitor_process
        must do kill → wait_empty → cleanup, in that order, to atomically
        clear any escaped descendants and reclaim the cgroup directory.
        """
        drone = Drone(
            scan_id="scan-cg-monitor",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        await drone.start()
        await drone.wait()

        cg = _FakeScanCgroup.instances[0]
        names = [e[0] for e in cg.events]
        # Strictly ordered: kill → wait_empty → cleanup, after the initial
        # create → populate from start().
        assert names == ["create", "populate", "kill", "wait_empty", "cleanup"], f"events={cg.events}"

    async def test_failsafe_kills_cgroup_synchronously(self) -> None:
        """The loop.call_later SIGKILL failsafe must fire cgroup.kill from
        within a synchronous callback. Use a short graceful_stop_timeout_s
        and a SIGTERM-ignoring subprocess to force the escalation path.
        """
        drone = Drone(
            scan_id="scan-cg-failsafe",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            graceful_stop_timeout_s=0.3,
            _subprocess_script=_MOCK_HANG_SCRIPT,
        )
        await drone.start()
        for _ in range(50):
            if drone.status == ScanStatus.RUNNING:
                break
            await asyncio.sleep(0.1)

        await drone.stop()  # cooperative → SIGTERM ignored → failsafe → cgroup.kill
        assert drone.is_terminal
        cg = _FakeScanCgroup.instances[0]
        kill_events = [e for e in cg.events if e[0] == "kill"]
        assert len(kill_events) >= 1, f"failsafe must invoke cgroup.kill; events={cg.events}"

    async def test_concurrent_drones_get_isolated_cgroups(self) -> None:
        """Two drones must get independent ScanCgroup instances with
        different paths — killing one must not appear in the other's
        event log. This is the property the old cmdline-based filter
        existed to provide; cgroup isolation makes it kernel-enforced.
        """
        drone_a = Drone(
            scan_id="scan-iso-A",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_HANG_SCRIPT,
        )
        drone_b = Drone(
            scan_id="scan-iso-B",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_HANG_SCRIPT,
        )
        await drone_a.start()
        await drone_b.start()

        assert len(_FakeScanCgroup.instances) == 2
        cg_a, cg_b = _FakeScanCgroup.instances
        assert cg_a.path != cg_b.path
        assert "scan-iso-A" in str(cg_a.path)
        assert "scan-iso-B" in str(cg_b.path)

        # Force-stop only drone A.
        await drone_a.stop(force=True)

        # cg_a got kill events; cg_b has no kill events yet.
        a_kills = [e for e in cg_a.events if e[0] == "kill"]
        b_kills = [e for e in cg_b.events if e[0] == "kill"]
        assert len(a_kills) >= 1
        assert b_kills == [], f"drone A's kill must not touch drone B's cgroup; B events={cg_b.events}"

        # Clean up drone B for hygiene.
        await drone_b.stop(force=True)


class TestDroneToInfo:
    """Tests for the to_info() summary method."""

    def test_to_info_initial(self) -> None:
        """to_info() on a fresh drone should show NOT_STARTED."""
        drone = Drone(
            scan_id="s1",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
        )
        info = drone.to_info()
        assert isinstance(info, ScanInfo)
        assert info.scan_id == "s1"
        assert info.status == ScanStatus.NOT_STARTED
        assert info.events_sent == 0

    async def test_to_info_after_completion(self) -> None:
        """to_info() after scan completion should show FINISHED and event count."""
        drone = Drone(
            scan_id="s1",
            preset=_MOCK_PRESET,
            on_event=AsyncMock(),
            on_status_change=AsyncMock(),
            on_log_line=AsyncMock(),
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        await drone.start()
        await drone.wait()

        info = drone.to_info()
        assert info.status == ScanStatus.FINISHED
        assert info.events_sent >= 1
        assert info.started_at is not None
