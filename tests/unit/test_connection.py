"""Tests for bbot_bee.connection."""

from ssl import CERT_NONE, SSLContext
from unittest.mock import AsyncMock, patch

import pytest

from bbot_bee.connection import ConnectionManager, ConnectionState

_WS_CONNECT = "swarm_common.resilient_websocket.websockets_connect"


class TestConnectionState:
    """Tests for the ConnectionState enum."""

    def test_all_states_present(self) -> None:
        """Verify all expected states exist."""
        expected = {"DISCONNECTED", "CONNECTING", "CONNECTED"}
        assert {s.value for s in ConnectionState} == expected


class TestConnectionManagerInit:
    """Tests for ConnectionManager construction."""

    def test_initial_state_is_disconnected(self) -> None:
        """Client-mode manager starts DISCONNECTED (hasn't connected yet)."""
        mgr = ConnectionManager(url="ws://localhost:8100/bees/ws/d1", api_key="key")
        assert mgr.state == ConnectionState.DISCONNECTED

    def test_stores_url(self) -> None:
        """The URL should be stored."""
        mgr = ConnectionManager(url="ws://localhost:8100/bees/ws/d1", api_key="key")
        assert mgr.url == "ws://localhost:8100/bees/ws/d1"

    def test_default_config(self) -> None:
        """Verify sensible defaults."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        assert mgr._backoff_base_s == 1.0
        assert mgr._backoff_max_s == 60.0
        assert mgr._backoff_jitter_factor == 0.25
        assert mgr._heartbeat is not None

    def test_custom_config(self) -> None:
        """Custom config values should be accepted."""
        mgr = ConnectionManager(
            url="wss://hive.example.com/bees/ws/d1",
            api_key="key",
            heartbeat_interval_s=5.0,
            heartbeat_timeout_s=10.0,
            backoff_base_s=2.0,
            backoff_max_s=30.0,
        )
        assert mgr._backoff_base_s == 2.0
        assert mgr._backoff_max_s == 30.0
        assert mgr._heartbeat._interval_s == 5.0
        assert mgr._heartbeat._timeout_s == 10.0


class TestConnectionManagerTlsVerify:
    """Tests for the tls_verify flag's effect on the underlying SSLContext."""

    def test_default_leaves_ssl_context_none(self) -> None:
        """tls_verify=True (default) must leave ssl_context unset so websockets builds its own verifying context for wss://."""
        mgr = ConnectionManager(url="wss://hive.example.com/ws", api_key="k")
        assert mgr._ssl_context is None

    def test_tls_verify_false_builds_permissive_context(self) -> None:
        """tls_verify=False must produce an SSLContext with verification fully disabled."""
        mgr = ConnectionManager(
            url="wss://hive.example.com/ws",
            api_key="k",
            tls_verify=False,
        )
        assert isinstance(mgr._ssl_context, SSLContext)
        assert mgr._ssl_context.check_hostname is False
        assert mgr._ssl_context.verify_mode == CERT_NONE

    def test_tls_verify_true_explicit_leaves_ssl_context_none(self) -> None:
        """Explicit tls_verify=True is equivalent to the default."""
        mgr = ConnectionManager(
            url="wss://hive.example.com/ws",
            api_key="k",
            tls_verify=True,
        )
        assert mgr._ssl_context is None

    def test_tls_verify_false_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Disabling TLS verification must emit a WARNING so it's visible in ops logs."""
        with caplog.at_level("WARNING", logger="bbot_bee.connection"):
            ConnectionManager(
                url="wss://hive.example.com/ws",
                api_key="k",
                tls_verify=False,
            )
        assert any("TLS certificate verification is DISABLED" in rec.message for rec in caplog.records), (
            "expected a WARNING about disabled TLS verification"
        )

    def test_tls_verify_false_on_ws_url_leaves_ssl_context_none(self) -> None:
        """tls_verify=False on a ws:// URL is a no-op: a plaintext link has no certificate to verify,
        and websockets rejects an ssl argument on a ws:// URI."""
        mgr = ConnectionManager(url="ws://hive.internal:8000/ws", api_key="k", tls_verify=False)
        assert mgr._ssl_context is None

    def test_tls_verify_false_on_ws_url_warns_noop(self, caplog: pytest.LogCaptureFixture) -> None:
        """A no-op --no-tls-verify on a ws:// URL should warn so operators see it had no effect."""
        with caplog.at_level("WARNING", logger="bbot_bee.connection"):
            ConnectionManager(url="ws://hive.internal:8000/ws", api_key="k", tls_verify=False)
        assert any("non-TLS" in rec.message and "ignored" in rec.message for rec in caplog.records), (
            "expected a WARNING that tls_verify=False was ignored on a non-TLS URL"
        )


class TestConnectionManagerConnect:
    """Tests for the connect method."""

    async def test_connect_success(self) -> None:
        """Successful connect should transition to CONNECTED."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        mock_ws = AsyncMock()
        with patch(_WS_CONNECT, AsyncMock(return_value=mock_ws)):
            await mgr.connect()
        assert mgr.state == ConnectionState.CONNECTED

    async def test_connect_fires_on_connect_callback(self) -> None:
        """on_connect callback should fire after successful connection."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        callback = AsyncMock()
        mgr.on_connect = callback
        mock_ws = AsyncMock()
        with patch(_WS_CONNECT, AsyncMock(return_value=mock_ws)):
            await mgr.connect()
        callback.assert_awaited_once()

    async def test_connect_failure_stays_disconnected(self) -> None:
        """Failed connect should remain DISCONNECTED."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        with (
            patch(_WS_CONNECT, AsyncMock(side_effect=OSError("refused"))),
            pytest.raises(ConnectionError),
        ):
            await mgr.connect()
        assert mgr.state == ConnectionState.DISCONNECTED

    async def test_connect_sends_auth_header(self) -> None:
        """Connect should pass Authorization header."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="my-secret")
        mock_ws = AsyncMock()
        mock_connect = AsyncMock(return_value=mock_ws)
        with patch(_WS_CONNECT, mock_connect):
            await mgr.connect()
        call_kwargs = mock_connect.call_args
        headers = call_kwargs.kwargs.get("additional_headers", {})
        assert headers.get("Authorization") == "Bearer my-secret"

    async def test_connect_ws_url_no_tls_verify_omits_ssl_kwarg(self) -> None:
        """On ws:// with tls_verify=False, connect() must NOT pass an ssl kwarg —
        websockets raises ValueError if ssl is supplied for a ws:// URI."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key", tls_verify=False)
        mock_connect = AsyncMock(return_value=AsyncMock())
        with patch(_WS_CONNECT, mock_connect):
            await mgr.connect()
        assert "ssl" not in mock_connect.call_args.kwargs

    async def test_connect_wss_url_no_tls_verify_passes_ssl_context(self) -> None:
        """On wss:// with tls_verify=False, connect() MUST pass the permissive SSLContext,
        else websockets falls back to its default verifying context."""
        mgr = ConnectionManager(url="wss://localhost/ws", api_key="key", tls_verify=False)
        mock_connect = AsyncMock(return_value=AsyncMock())
        with patch(_WS_CONNECT, mock_connect):
            await mgr.connect()
        assert isinstance(mock_connect.call_args.kwargs.get("ssl"), SSLContext)


class TestConnectionManagerSendRecv:
    """Tests for send_raw and recv_raw."""

    async def test_send_raw(self) -> None:
        """send_raw should delegate to the WebSocket."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        mock_ws = AsyncMock()
        with patch(_WS_CONNECT, AsyncMock(return_value=mock_ws)):
            await mgr.connect()
        await mgr.send_raw(b"hello")
        mock_ws.send.assert_awaited_once_with(b"hello")

    async def test_recv_raw(self) -> None:
        """recv_raw should delegate to the WebSocket."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = b"world"
        with patch(_WS_CONNECT, AsyncMock(return_value=mock_ws)):
            await mgr.connect()
        result = await mgr.recv_raw()
        assert result == b"world"

    async def test_send_when_disconnected_raises(self) -> None:
        """Sending while disconnected should raise ConnectionError."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        with pytest.raises(ConnectionError):
            await mgr.send_raw(b"data")

    async def test_recv_when_disconnected_raises(self) -> None:
        """Receiving while disconnected should raise ConnectionError."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        with pytest.raises(ConnectionError):
            await mgr.recv_raw()


class TestConnectionManagerDisconnect:
    """Tests for the disconnect method."""

    async def test_disconnect_closes_ws(self) -> None:
        """disconnect should close the underlying WebSocket."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        mock_ws = AsyncMock()
        with patch(_WS_CONNECT, AsyncMock(return_value=mock_ws)):
            await mgr.connect()
        await mgr.disconnect()
        mock_ws.close.assert_awaited_once()
        assert mgr.state == ConnectionState.DISCONNECTED

    async def test_disconnect_fires_callback(self) -> None:
        """on_disconnect callback should fire."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        callback = AsyncMock()
        mgr.on_disconnect = callback
        mock_ws = AsyncMock()
        with patch(_WS_CONNECT, AsyncMock(return_value=mock_ws)):
            await mgr.connect()
        await mgr.disconnect()
        callback.assert_awaited_once()

    async def test_double_disconnect_safe(self) -> None:
        """Calling disconnect twice should not raise."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        await mgr.disconnect()
        await mgr.disconnect()


class TestBackoffCalculation:
    """Tests for the exponential backoff logic."""

    def test_backoff_increases_exponentially(self) -> None:
        """Delay should double with each attempt (before jitter)."""
        mgr = ConnectionManager(
            url="ws://localhost/ws",
            api_key="key",
            backoff_base_s=1.0,
            backoff_max_s=60.0,
            backoff_jitter_factor=0.0,
        )
        assert mgr.calc_backoff(0) == 1.0
        assert mgr.calc_backoff(1) == 2.0
        assert mgr.calc_backoff(2) == 4.0
        assert mgr.calc_backoff(3) == 8.0

    def test_backoff_capped_at_max(self) -> None:
        """Delay should never exceed backoff_max_s."""
        mgr = ConnectionManager(
            url="ws://localhost/ws",
            api_key="key",
            backoff_base_s=1.0,
            backoff_max_s=10.0,
            backoff_jitter_factor=0.0,
        )
        assert mgr.calc_backoff(100) == 10.0

    def test_backoff_with_jitter(self) -> None:
        """Jitter should add randomness within factor * base_delay."""
        mgr = ConnectionManager(
            url="ws://localhost/ws",
            api_key="key",
            backoff_base_s=1.0,
            backoff_max_s=60.0,
            backoff_jitter_factor=0.25,
        )
        delays = {mgr.calc_backoff(0) for _ in range(100)}
        assert all(1.0 <= d <= 1.25 for d in delays)
        assert len(delays) > 1


class TestConnectionManagerProperties:
    """Tests for convenience properties."""

    def test_is_connected_false_initially(self) -> None:
        """is_connected should be False before connecting."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        assert mgr.is_connected is False

    async def test_is_connected_true_after_connect(self) -> None:
        """is_connected should be True after successful connect."""
        mgr = ConnectionManager(url="ws://localhost/ws", api_key="key")
        mock_ws = AsyncMock()
        with patch(_WS_CONNECT, AsyncMock(return_value=mock_ws)):
            await mgr.connect()
        assert mgr.is_connected is True
