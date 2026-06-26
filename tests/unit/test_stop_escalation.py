"""Tests for Drone.stop() SIGKILL escalation robustness.

These regressions pin the contract that the graceful-stop escalation
(SIGTERM → wait → SIGKILL) is decoupled from the calling coroutine's
cancellation. Without this decoupling, any external ``task.cancel()``
on whatever coroutine is awaiting ``drone.stop()`` also cancels the
SIGKILL-escalation branch — leaving a stuck SIGTERM-ignoring subprocess
running forever.

The bug was hit in production when the bee's WebSocket heartbeat timed
out mid-stop (hive was saturated draining a 10-minute scan's event
backlog), triggering a reconnect that cancelled ``_message_loop``,
which cascaded into cancelling the in-flight ``drone.stop()``, which
cancelled its ``await _await_exit`` — bypassing the ``except
TimeoutError`` branch that would have fired SIGKILL.

The fix: schedule SIGKILL via ``loop.call_later`` so it runs as a
synchronous event-loop callback immune to coroutine cancellation.
"""

from __future__ import annotations

import asyncio
import textwrap
from unittest.mock import AsyncMock

import pytest
from swarm_common.models import ScanStatus

from bbot_bee.drone import Drone

# Subprocess that installs SIGTERM → SIG_IGN and then spins forever.
# SIGKILL is the only way out.
_IGNORE_SIGTERM = textwrap.dedent("""\
    import json, signal, sys, time
    config = json.loads(sys.stdin.readline())
    print(json.dumps({"_type": "status", "status": "STARTING", "status_code": 2}), flush=True)
    print(json.dumps({"_type": "status", "status": "RUNNING", "status_code": 3}), flush=True)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.1)
""")

# Subprocess that exits cleanly on SIGTERM with code 2 (ABORTED).
_HONOR_SIGTERM = textwrap.dedent("""\
    import json, signal, sys, time
    config = json.loads(sys.stdin.readline())
    print(json.dumps({"_type": "status", "status": "STARTING", "status_code": 2}), flush=True)
    print(json.dumps({"_type": "status", "status": "RUNNING", "status_code": 3}), flush=True)
    def _bye(sig, frame):
        print(json.dumps({"_type": "status", "status": "ABORTED", "status_code": 8}), flush=True)
        sys.exit(2)
    signal.signal(signal.SIGTERM, _bye)
    while True:
        time.sleep(0.1)
""")

_PRESET = {"target": ["127.0.0.1"], "modules": []}


async def _make_drone(script: str, graceful_timeout: float) -> Drone:
    """Spawn a Drone with inline subprocess, wait for RUNNING."""
    drone = Drone(
        scan_id="escalate-test",
        preset=_PRESET,
        on_event=AsyncMock(),
        on_status_change=AsyncMock(),
        on_log_line=AsyncMock(),
        graceful_stop_timeout_s=graceful_timeout,
        _subprocess_script=script,
    )
    await drone.start()
    for _ in range(50):
        if drone.status == ScanStatus.RUNNING:
            return drone
        await asyncio.sleep(0.1)
    raise AssertionError(f"subprocess never reached RUNNING (last: {drone.status})")


class TestStopEscalationRobustness:
    """The SIGKILL failsafe must fire regardless of caller cancellation."""

    async def test_sigkill_fires_when_subprocess_ignores_sigterm(self) -> None:
        """Subprocess with SIGTERM handler = SIG_IGN must still be killed.

        Validates the baseline escalation path: the test fixture is designed
        such that nothing except SIGKILL will stop the subprocess.
        """
        drone = await _make_drone(_IGNORE_SIGTERM, graceful_timeout=0.5)
        assert drone._process is not None

        await drone.stop(force=False)

        assert drone._process.returncode is not None, "subprocess still running"
        # Python's subprocess reports SIGKILL-terminated processes with
        # returncode = -9 (negative of the signal number).
        assert drone._process.returncode == -9, f"expected SIGKILL (-9), got {drone._process.returncode}"

    async def test_sigkill_fires_even_when_caller_cancelled(self) -> None:
        """The core regression.

        Cancelling the task that is awaiting ``drone.stop()`` must NOT
        prevent the scheduled SIGKILL from firing. Pre-fix, the cancel
        cascaded into ``_await_exit`` and bypassed the ``except
        TimeoutError`` escalation branch.
        """
        drone = await _make_drone(_IGNORE_SIGTERM, graceful_timeout=1.0)
        assert drone._process is not None

        stop_task = asyncio.create_task(drone.stop(force=False))
        # Cancel mid-wait, well before the graceful timeout would elapse.
        await asyncio.sleep(0.2)
        stop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stop_task

        # Despite the cancellation, the scheduled SIGKILL must fire and
        # kill the subprocess within graceful_timeout + slack.
        for _ in range(40):  # 4s total
            if drone._process.returncode is not None:
                break
            await asyncio.sleep(0.1)

        assert drone._process.returncode == -9, (
            f"subprocess survived caller cancellation — returncode={drone._process.returncode}"
        )

    async def test_cooperative_exit_cancels_failsafe_timer(self) -> None:
        """When SIGTERM works, SIGKILL must NOT fire.

        The escalation timer is set unconditionally when SIGTERM goes out;
        if the subprocess exits cooperatively, the timer handle has to be
        cancelled so no redundant SIGKILL is sent.
        """
        drone = await _make_drone(_HONOR_SIGTERM, graceful_timeout=5.0)
        assert drone._process is not None

        await drone.stop(force=False)

        rc = drone._process.returncode
        # Must NOT be -9 (SIGKILL). Accept either clean exit(2) or
        # SIGTERM-terminated (-15) depending on the subprocess's handler.
        assert rc != -9, f"SIGKILL fired unexpectedly (returncode={rc})"
        assert rc is not None, "subprocess still running"

    async def test_aborted_status_wins_after_escalation(self) -> None:
        """After the failsafe SIGKILL lands, status must be ABORTED.

        _monitor_process will observe a non-zero SIGKILL returncode and
        attempt to set FAILED; the failsafe's explicit _set_status(ABORTED)
        must win via forward-only comparison (ABORTED=8 > FAILED=7).
        """
        drone = await _make_drone(_IGNORE_SIGTERM, graceful_timeout=0.5)

        await drone.stop(force=False)
        # _set_status(ABORTED) is dispatched as a task from the timer
        # callback; allow a moment for it to run after stop() returns.
        for _ in range(20):
            if drone.status == ScanStatus.ABORTED:
                break
            await asyncio.sleep(0.05)

        assert drone.status == ScanStatus.ABORTED, (
            f"status should be ABORTED after forced escalation, got {drone.status.value}"
        )
