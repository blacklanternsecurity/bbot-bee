"""Tests for bbot_bee.config — BeeConfig pydantic-settings model."""

import pytest
from pydantic import ValidationError

from bbot_bee.config import BeeConfig


class TestBeeConfigDefaults:
    """Tests for BeeConfig default values."""

    def test_default_max_init_concurrent_scans(self) -> None:
        """Default max_init_concurrent_scans should be 3."""
        config = BeeConfig(hive_url="ws://localhost/ws", api_key="k")
        assert config.max_init_concurrent_scans == 3

    def test_default_tls_verify(self) -> None:
        """Default TLS verify should be True."""
        config = BeeConfig(hive_url="ws://localhost/ws", api_key="k")
        assert config.tls_verify is True

    def test_default_event_batch_size(self) -> None:
        """Default event batch size should be 100."""
        config = BeeConfig(hive_url="ws://localhost/ws", api_key="k")
        assert config.event_batch_size == 100

    def test_default_log_level(self) -> None:
        """Default log level should be INFO."""
        config = BeeConfig(hive_url="ws://localhost/ws", api_key="k")
        assert config.log_level == "INFO"

    def test_default_graceful_stop_timeout(self) -> None:
        """Default graceful stop timeout should be 30 seconds."""
        config = BeeConfig(hive_url="ws://localhost/ws", api_key="k")
        assert config.graceful_stop_timeout_s == 30.0

    def test_bee_id_auto_generated(self) -> None:
        """bee_id should be auto-generated if not provided."""
        config = BeeConfig(hive_url="ws://localhost/ws", api_key="k")
        assert config.bee_id is not None
        assert len(config.bee_id) > 0


class TestBeeConfigCustom:
    """Tests for BeeConfig with custom values."""

    def test_custom_values(self) -> None:
        """All custom values should be accepted."""
        config = BeeConfig(
            bee_id="my-drone",
            hive_url="wss://hive.example.com/drones/ws/my-drone",
            api_key="my-secret",
            max_init_concurrent_scans=5,
            tls_verify=False,
            event_batch_size=200,
            log_level="DEBUG",
        )
        assert config.bee_id == "my-drone"
        assert config.hive_url == "wss://hive.example.com/drones/ws/my-drone"
        assert config.max_init_concurrent_scans == 5
        assert config.tls_verify is False

    def test_rejects_zero_max_init_concurrent_scans(self) -> None:
        """max_init_concurrent_scans=0 should fail validation."""
        with pytest.raises(ValidationError):
            BeeConfig(hive_url="ws://localhost/ws", api_key="k", max_init_concurrent_scans=0)

    def test_hive_url_required(self) -> None:
        """hive_url should be required."""
        with pytest.raises(ValidationError):
            BeeConfig(api_key="k")  # type: ignore[call-arg]

    def test_api_key_required(self) -> None:
        """api_key should be required."""
        with pytest.raises(ValidationError):
            BeeConfig(hive_url="ws://localhost/ws")  # type: ignore[call-arg]


class TestBeeConfigEnvPrefix:
    """Tests for environment variable loading."""

    def test_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config should load from BBOT_BEE_ prefixed env vars."""
        monkeypatch.setenv("BBOT_BEE_HIVE_URL", "ws://env-hive/ws")
        monkeypatch.setenv("BBOT_BEE_API_KEY", "env-key")
        monkeypatch.setenv("BBOT_BEE_MAX_INIT_CONCURRENT_SCANS", "10")
        config = BeeConfig()  # type: ignore[call-arg]
        assert config.hive_url == "ws://env-hive/ws"
        assert config.api_key == "env-key"
        assert config.max_init_concurrent_scans == 10
