"""Configuration for the bbot_bee agent (env vars with `BBOT_BEE_` prefix, CLI args, or direct construction)."""

from __future__ import annotations

from uuid import uuid4

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_bee_id() -> str:
    """Generate a default drone ID."""
    return f"drone-{uuid4().hex[:12]}"


class BeeConfig(BaseSettings):
    """Configuration for the bbot_bee agent."""

    bee_id: str = Field(default_factory=_default_bee_id)
    hive_url: str
    api_key: str

    # Initial value reported to hive on connect; the hive owns the runtime
    # value via set_max_scans commands.
    max_init_concurrent_scans: int = Field(default=3, gt=0)

    tls_verify: bool = True
    tls_ca_bundle: str | None = None

    graceful_stop_timeout_s: float = 30.0

    event_batch_size: int = 100
    event_flush_interval_s: float = 2.0
    log_batch_size: int = 50
    log_flush_interval_s: float = 5.0

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="BBOT_BEE_")
