"""Layer 1: ConnectionManager (client mode).

Thin subclass of swarm_common.resilient_websocket.ResilientWebSocket
specialized for bee→hive connections:
- Constructs auth headers from API key
- Configures AppLevelHeartbeat with bee-ping/hive-pong payloads

The heavy lifting (connect, disconnect, send_raw, recv_raw, heartbeat,
backoff, disconnect detection) lives in ResilientWebSocket.
"""

from __future__ import annotations

from logging import getLogger
from ssl import CERT_NONE, PROTOCOL_TLS_CLIENT, SSLContext

from swarm_common.resilient_websocket import (
    AppLevelHeartbeat,
    ConnectionState,
    ResilientWebSocket,
)

log = getLogger(__name__)

__all__ = ["ConnectionManager", "ConnectionState"]

# Application-level ping/pong payloads (must match hive's bee_ws.py)
_PING_PAYLOAD = b"bee-ping"
_PONG_PAYLOAD = b"hive-pong"


class ConnectionManager(ResilientWebSocket):
    """Layer 1: WebSocket client connecting to the hive.

    Subclass of ResilientWebSocket with bee-specific defaults:
    - Bearer token auth from API key
    - App-level heartbeat with bee-ping/hive-pong
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        heartbeat_interval_s: float = 15.0,
        heartbeat_timeout_s: float = 30.0,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        backoff_jitter_factor: float = 0.25,
        tls_verify: bool = True,
    ) -> None:
        """Initialize the bee-side connection manager.

        Args:
            url: WebSocket URL of the hive endpoint the bee should connect to.
            api_key: Bee API key, sent as a ``Bearer`` token in the handshake.
            heartbeat_interval_s: Seconds between app-level ``bee-ping`` frames.
            heartbeat_timeout_s: Seconds to wait for a matching ``hive-pong``
                before declaring the connection dead.
            backoff_base_s: Initial reconnect delay.
            backoff_max_s: Maximum reconnect delay.
            backoff_jitter_factor: Random jitter applied to each reconnect delay,
                expressed as a fraction of the current delay.
            tls_verify: If False, skip TLS certificate verification (dev only).
        """
        heartbeat = AppLevelHeartbeat(
            ping_payload=_PING_PAYLOAD,
            pong_payload=_PONG_PAYLOAD,
            interval_s=heartbeat_interval_s,
            timeout_s=heartbeat_timeout_s,
        )

        # Build an SSLContext only when verification is being disabled.
        # When tls_verify=True we leave ssl_context=None so that websockets
        # constructs its own default context (with verification enabled)
        # for wss:// URLs.
        ssl_context: SSLContext | None = None
        if not tls_verify:
            ssl_context = SSLContext(PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = CERT_NONE
            log.warning(f"ConnectionManager: TLS certificate verification is DISABLED for {url} (dev only)")

        super().__init__(
            url=url,
            headers={"Authorization": f"Bearer {api_key}"},
            heartbeat=heartbeat,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
            backoff_jitter_factor=backoff_jitter_factor,
            ssl_context=ssl_context,
        )

        log.debug(
            f"ConnectionManager initialized: {url=}, {heartbeat_interval_s=:.1f}s, "
            f"{backoff_base_s=:.1f}s, {backoff_max_s=:.1f}s, {tls_verify=}"
        )
