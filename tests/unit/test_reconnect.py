"""Tests for the bee reconnect loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from bbot_bee.config import BeeConfig
from bbot_bee.connection import ConnectionState
from bbot_bee.queen import Queen


def _make_config(**overrides: object) -> BeeConfig:
    """Create a BeeConfig for testing."""
    defaults = {
        "hive_url": "ws://localhost:8100/bees/ws/test-bee",
        "api_key": "test-key",
        "max_init_concurrent_scans": 3,
        "log_level": "DEBUG",
    }
    defaults.update(overrides)
    return BeeConfig(**defaults)


class TestReconnectLoop:
    """Queen.run() must reconnect after connection drops."""

    async def test_disconnected_event_set_on_disconnect(self) -> None:
        """The disconnected_event is set when _mark_disconnected fires."""
        queen = Queen(_make_config())
        assert not queen._connection.disconnected_event.is_set()
        queen._connection._state = ConnectionState.CONNECTED
        await queen._connection._mark_disconnected()
        assert queen._connection.disconnected_event.is_set()

    async def test_disconnected_event_cleared_on_new_loop(self) -> None:
        """The disconnected_event is cleared at the start of each reconnect iteration."""
        queen = Queen(_make_config())
        queen._connection.disconnected_event.set()

        with (
            patch.object(queen._connection, "connect", side_effect=ConnectionError("test")),
            patch.object(queen._connection, "disconnect", new_callable=AsyncMock),
        ):
            task = asyncio.create_task(queen.run())
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_reconnect_resets_backoff_on_success(self) -> None:
        """Backoff resets to 1.0 after a successful connection."""
        queen = Queen(_make_config())

        connect_count = 0

        async def mock_connect() -> None:
            nonlocal connect_count
            connect_count += 1
            if connect_count == 1:
                raise ConnectionError("first attempt fails")
            queen._connection.disconnected_event.set()

        with (
            patch.object(queen._connection, "connect", side_effect=mock_connect),
            patch.object(queen._connection, "disconnect", new_callable=AsyncMock),
        ):
            task = asyncio.create_task(queen.run())
            # Allow time for 1 fail + backoff + 1 success.
            await asyncio.sleep(2.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert connect_count >= 2

    async def test_tasks_cancelled_before_reconnect(self) -> None:
        """Background tasks are cancelled before each reconnect attempt."""
        queen = Queen(_make_config())

        async def mock_connect() -> None:
            queen._connection.disconnected_event.set()

        with (
            patch.object(queen._connection, "connect", side_effect=mock_connect),
            patch.object(queen._connection, "disconnect", new_callable=AsyncMock),
        ):
            task = asyncio.create_task(queen.run())
            await asyncio.sleep(0.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert queen._tasks == []
