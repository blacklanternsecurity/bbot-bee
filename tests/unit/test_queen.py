"""Tests for bbot_bee.queen — Queen class orchestrating drones and hive communication.

Queen is the main agent process: connects to the hive via WebSocket,
manages drone subprocesses, buffers events/logs, and reports state.
Merges the responsibilities of the old Drone + ScanSupervisor classes.
"""

from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock

import pytest
from swarm_common.models import ScanStatus, StateSyncPayload

from bbot_bee.config import BeeConfig
from bbot_bee.queen import Queen

# Mock subprocess script for fast scan lifecycle tests
_MOCK_SUCCESS_SCRIPT = textwrap.dedent("""\
    import json, sys, time
    config = json.loads(sys.stdin.readline())
    scan_id = config["scan_id"]
    for status, code in [("STARTING", 2), ("RUNNING", 3), ("FINISHING", 4), ("FINISHED", 6)]:
        print(json.dumps({"_type": "status", "status": status, "status_code": code}), flush=True)
        time.sleep(0.02)
    print(json.dumps({"_type": "event", "type": "DNS_NAME", "data": "example.com", "scan": scan_id}), flush=True)
    sys.exit(0)
""")


def _make_config(**overrides: object) -> BeeConfig:
    """Create a test BeeConfig with sensible defaults."""
    defaults = {
        "bee_id": "test-queen",
        "hive_url": "ws://localhost:8100/drones/ws/test-queen",
        "api_key": "test-key",
        "max_init_concurrent_scans": 3,
        "event_batch_size": 5,
        "event_flush_interval_s": 0.1,
        "log_batch_size": 5,
        "log_flush_interval_s": 0.1,
        "graceful_stop_timeout_s": 5.0,
        "log_level": "DEBUG",
    }
    defaults.update(overrides)
    return BeeConfig(**defaults)  # type: ignore[arg-type]


# Dummy preset — mock scripts ignore it
_MOCK_PRESET = {"target": ["127.0.0.1"], "modules": []}


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestQueenCgroupGate:
    """Queen refuses to boot if cgroup v2 / cgroup.kill is not available.

    Mandatory by user requirement — no env flag override. The bee must
    fail loudly rather than silently fall back to soft-guarantee mode.
    """

    def test_init_raises_when_cgroup_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """detect_cgroup_kill_supported() == False → Queen.__init__ raises."""
        monkeypatch.setattr("bbot_bee.cgroup.detect_cgroup_kill_supported", lambda: False)
        with pytest.raises(RuntimeError, match="cgroup"):
            Queen(_make_config())

    def test_init_recovers_orphan_cgroups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stale cgroups from a crashed bee must be reclaimed at startup."""
        calls = {"n": 0}

        def fake_recover() -> int:
            calls["n"] += 1
            return 2

        monkeypatch.setattr("bbot_bee.cgroup.recover_orphan_cgroups", fake_recover)
        Queen(_make_config())
        assert calls["n"] == 1

    async def test_shutdown_recovers_orphan_cgroups(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Best-effort cleanup on shutdown — recover called after stop_all."""
        calls: list[str] = []

        monkeypatch.setattr(
            "bbot_bee.cgroup.recover_orphan_cgroups",
            lambda: calls.append("recover") or 0,
        )
        queen = Queen(_make_config())
        queen._connection = AsyncMock()  # type: ignore[assignment]
        await queen.shutdown()
        # Called once at init AND once at shutdown.
        assert calls.count("recover") == 2


class TestQueenInit:
    """Tests for Queen construction."""

    def test_creates_with_config(self) -> None:
        """Queen should accept a BeeConfig."""
        config = _make_config()
        queen = Queen(config)
        assert queen.bee_id == "test-queen"

    def test_initial_state(self) -> None:
        """Queen should start with empty buffers and no active drones."""
        config = _make_config()
        queen = Queen(config)
        assert queen._event_buffer == {}
        assert queen._log_buffer == {}
        assert queen._drones == {}


# ---------------------------------------------------------------------------
# Drone management (capacity, start, stop)
# ---------------------------------------------------------------------------


class TestQueenStartScan:
    """Tests for starting scans with capacity enforcement."""

    async def test_start_scan_creates_drone(self) -> None:
        """start_scan should create a Drone and track it."""
        queen = Queen(_make_config())
        await queen.start_scan(
            "scan-001",
            preset=_MOCK_PRESET,
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        assert "scan-001" in queen._drones

    async def test_capacity_exceeded_raises(self) -> None:
        """Starting a scan beyond capacity should raise RuntimeError."""
        queen = Queen(_make_config(max_init_concurrent_scans=1))
        await queen.start_scan(
            "scan-001",
            preset=_MOCK_PRESET,
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        with pytest.raises(RuntimeError, match="capacity"):
            await queen.start_scan(
                "scan-002",
                preset=_MOCK_PRESET,
                _subprocess_script=_MOCK_SUCCESS_SCRIPT,
            )
        await queen.stop_all()

    async def test_duplicate_scan_id_raises(self) -> None:
        """Starting a scan with a duplicate ID should raise ValueError."""
        queen = Queen(_make_config(max_init_concurrent_scans=5))
        await queen.start_scan(
            "scan-001",
            preset=_MOCK_PRESET,
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        with pytest.raises(ValueError, match="already exists"):
            await queen.start_scan(
                "scan-001",
                preset=_MOCK_PRESET,
                _subprocess_script=_MOCK_SUCCESS_SCRIPT,
            )
        await queen.stop_all()


class TestQueenStopScan:
    """Tests for stopping individual scans."""

    async def test_stop_unknown_scan_raises(self) -> None:
        """Stopping a non-existent scan should raise KeyError."""
        queen = Queen(_make_config())
        with pytest.raises(KeyError):
            await queen.stop_scan("nonexistent")


class TestQueenStopAll:
    """Tests for stopping all scans at once."""

    async def test_stop_all_on_empty_is_safe(self) -> None:
        """stop_all with no scans should not raise."""
        queen = Queen(_make_config())
        await queen.stop_all()


class TestQueenCapacity:
    """Tests for capacity reporting."""

    def test_available_capacity_with_no_scans(self) -> None:
        """available_capacity should equal max when no scans running."""
        queen = Queen(_make_config(max_init_concurrent_scans=3))
        assert queen.available_capacity == 3


# ---------------------------------------------------------------------------
# Event buffering
# ---------------------------------------------------------------------------


class TestQueenEventBuffering:
    """Tests for event buffering and flushing."""

    async def test_buffer_event(self) -> None:
        """Events should be buffered by scan_id."""
        queen = Queen(_make_config())
        await queen._handle_scan_event("scan-1", {"type": "DNS_NAME", "data": "example.com"})
        assert len(queen._event_buffer["scan-1"]) == 1

    async def test_buffer_events_separate_scans(self) -> None:
        """Events from different scans should be buffered separately."""
        queen = Queen(_make_config())
        await queen._handle_scan_event("scan-1", {"type": "DNS_NAME"})
        await queen._handle_scan_event("scan-2", {"type": "IP_ADDRESS"})
        assert len(queen._event_buffer["scan-1"]) == 1
        assert len(queen._event_buffer["scan-2"]) == 1


# ---------------------------------------------------------------------------
# Log buffering
# ---------------------------------------------------------------------------


class TestQueenLogBuffering:
    """Tests for log line buffering."""

    async def test_buffer_log_line(self) -> None:
        """Log lines should be buffered by scan_id."""
        queen = Queen(_make_config())
        await queen._handle_scan_log_line("scan-1", "some log message")
        assert queen._log_buffer["scan-1"][0] == "some log message"


# ---------------------------------------------------------------------------
# Status change handling
# ---------------------------------------------------------------------------


class TestQueenStatusChange:
    """Tests for scan status change handling."""

    async def test_status_change_queued_for_send(self) -> None:
        """Status changes should be queued for sending to the hive."""
        queen = Queen(_make_config())
        queen._channel = AsyncMock()
        queen._channel.send = AsyncMock()
        await queen._handle_scan_status_change("scan-1", ScanStatus.RUNNING)
        queen._channel.send.assert_awaited_once()

    async def test_terminal_status_auto_removes_drone(self) -> None:
        """A terminal status should auto-remove the drone from tracking."""
        queen = Queen(_make_config())
        # Mock the channel so status sends don't fail (no real hive connection)
        queen._channel = AsyncMock()
        queen._channel.send = AsyncMock()

        await queen.start_scan(
            "scan-auto",
            preset=_MOCK_PRESET,
            _subprocess_script=_MOCK_SUCCESS_SCRIPT,
        )
        assert "scan-auto" in queen._drones

        # Wait for the fast mock scan to finish (auto-cleans)
        drone = queen._drones.get("scan-auto")
        if drone is not None:
            await drone.wait()

        # The terminal status callback should have auto-removed it
        assert "scan-auto" not in queen._drones


# ---------------------------------------------------------------------------
# State sync
# ---------------------------------------------------------------------------


class TestQueenStateSync:
    """Tests for state sync payload construction."""

    def test_build_state_sync(self) -> None:
        """_build_state_sync should return a StateSyncPayload."""
        queen = Queen(_make_config())
        payload = queen._build_state_sync()
        assert isinstance(payload, StateSyncPayload)
        assert payload.bee_id == "test-queen"
        assert payload.capacity["max_scans"] == 3
        assert payload.capacity["init_max_scans"] == 3
        assert payload.capacity["available"] == 3
        assert payload.active_scans == {}

    async def test_set_max_scans_updates_limit_and_sends_state_sync(self) -> None:
        """set_max_scans command should update the runtime limit and emit state_sync."""
        queen = Queen(_make_config(max_init_concurrent_scans=3))
        sent: list = []

        async def _capture(msg: object, priority: object = None) -> None:
            sent.append(msg)

        queen._channel.send = _capture  # type: ignore[method-assign]

        await queen._handle_command({"cmd": "set_max_scans", "max_scans": 7})

        assert queen._max_concurrent_scans == 7
        assert queen._init_max_scans == 3
        assert len(sent) == 1
        assert sent[0].payload["capacity"]["max_scans"] == 7
        assert sent[0].payload["capacity"]["init_max_scans"] == 3


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestQueenShutdown:
    """Tests for graceful shutdown."""

    async def test_shutdown_disconnects(self) -> None:
        """shutdown() should disconnect from the hive."""
        queen = Queen(_make_config())
        queen._connection = AsyncMock()
        queen._connection.disconnect = AsyncMock()
        await queen.shutdown()
        queen._connection.disconnect.assert_awaited_once()
