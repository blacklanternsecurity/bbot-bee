"""Tests for drone cleanup on scan finish."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from swarm_common.models import ScanInfo, ScanStatus

from bbot_bee.config import BeeConfig
from bbot_bee.queen import Queen


def _make_config(**overrides: object) -> BeeConfig:
    """Create a BeeConfig for testing."""
    defaults = {
        "hive_url": "ws://localhost:8100/bees/ws/test-bee",
        "api_key": "test-key",
        "max_init_concurrent_scans": 3,
    }
    defaults.update(overrides)
    return BeeConfig(**defaults)


class TestDroneCleanupOnFinish:
    """_drones.pop must happen regardless of channel.send success."""

    async def test_drone_removed_on_finished(self) -> None:
        """Drone is removed from _drones when scan reaches FINISHED."""
        queen = Queen(_make_config())
        mock_drone = MagicMock()
        mock_drone.started_at = 1000.0
        mock_drone.finished_at = 1010.0
        queen._drones["scan-1"] = mock_drone

        assert queen.available_capacity == 2

        queen._channel = AsyncMock()
        queen._channel.send = AsyncMock()

        await queen._handle_scan_status_change("scan-1", ScanStatus.FINISHED)

        assert "scan-1" not in queen._drones
        assert queen.available_capacity == 3

    async def test_drone_removed_even_when_send_fails(self) -> None:
        """Drone is removed from _drones even when channel.send raises — pop happens before send."""
        queen = Queen(_make_config())
        mock_drone = MagicMock()
        mock_drone.started_at = 1000.0
        mock_drone.finished_at = 1010.0
        queen._drones["scan-1"] = mock_drone

        queen._channel = AsyncMock()
        queen._channel.send = AsyncMock(side_effect=ConnectionError("hive down"))

        with pytest.raises(ConnectionError):
            await queen._handle_scan_status_change("scan-1", ScanStatus.FINISHED)

        assert "scan-1" not in queen._drones
        assert queen.available_capacity == 3

    async def test_drone_not_removed_on_running(self) -> None:
        """Drone stays in _drones for non-terminal statuses."""
        queen = Queen(_make_config())
        mock_drone = MagicMock()
        mock_drone.started_at = 1000.0
        mock_drone.finished_at = None
        queen._drones["scan-1"] = mock_drone

        queen._channel = AsyncMock()
        queen._channel.send = AsyncMock()

        await queen._handle_scan_status_change("scan-1", ScanStatus.RUNNING)

        assert "scan-1" in queen._drones
        assert queen.available_capacity == 2

    async def test_drone_removed_on_failed(self) -> None:
        """Drone is removed on FAILED status."""
        queen = Queen(_make_config())
        mock_drone = MagicMock()
        mock_drone.started_at = 1000.0
        mock_drone.finished_at = 1005.0
        queen._drones["scan-1"] = mock_drone

        queen._channel = AsyncMock()
        queen._channel.send = AsyncMock()

        await queen._handle_scan_status_change("scan-1", ScanStatus.FAILED)

        assert "scan-1" not in queen._drones

    async def test_drone_removed_on_aborted(self) -> None:
        """Drone is removed on ABORTED status."""
        queen = Queen(_make_config())
        mock_drone = MagicMock()
        mock_drone.started_at = 1000.0
        mock_drone.finished_at = 1005.0
        queen._drones["scan-1"] = mock_drone

        queen._channel = AsyncMock()
        queen._channel.send = AsyncMock()

        await queen._handle_scan_status_change("scan-1", ScanStatus.ABORTED)

        assert "scan-1" not in queen._drones

    async def test_state_sync_accurate_after_failed_send(self) -> None:
        """State sync payload reflects cleaned-up drones even after send failure."""
        queen = Queen(_make_config())
        mock_drone_1 = MagicMock()
        mock_drone_1.started_at = 1000.0
        mock_drone_1.finished_at = 1010.0
        mock_drone_1.to_info.return_value = ScanInfo(scan_id="scan-1", status=ScanStatus.FINISHED)
        queen._drones["scan-1"] = mock_drone_1

        mock_drone_2 = MagicMock()
        mock_drone_2.started_at = 1000.0
        mock_drone_2.finished_at = None
        mock_drone_2.to_info.return_value = ScanInfo(scan_id="scan-2", status=ScanStatus.RUNNING, events_sent=5)
        queen._drones["scan-2"] = mock_drone_2

        queen._channel = AsyncMock()
        queen._channel.send = AsyncMock(side_effect=ConnectionError("hive down"))

        with pytest.raises(ConnectionError):
            await queen._handle_scan_status_change("scan-1", ScanStatus.FINISHED)

        payload = queen._build_state_sync()
        assert "scan-1" not in payload.active_scans
        assert len(payload.active_scans) == 1
        assert "scan-2" in payload.active_scans
        assert payload.capacity["available"] == 2
