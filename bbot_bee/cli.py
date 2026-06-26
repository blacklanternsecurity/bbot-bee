"""CLI entry point for the bbot_bee agent.

Usage:
    bbot-bee --hive-url ws://hive:8100/drones/ws/my-drone --api-key <key>
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from asyncio import create_task, get_running_loop, run
from logging import INFO, addLevelName, basicConfig, getLevelName, getLogger
from signal import SIGINT, SIGTERM

from bbot_bee.config import BeeConfig
from bbot_bee.queen import Queen

# Register TRACE level (5) so log output shows "TRACE" not "Level 5".
# Safe to call at module import time; affects only the logging module's
# internal name table.
addLevelName(5, "TRACE")

log = getLogger(__name__)


def _build_parser() -> ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = ArgumentParser(
        prog="bbot-bee",
        description="BBOT Drone — agent that wraps BBOT to execute distributed scans",
    )
    parser.add_argument("--hive-url", default=None, help="WebSocket URL of the hive orchestrator")
    parser.add_argument("--api-key", default=None, help="Drone API key for hive authentication")
    parser.add_argument("--bee-id", default=None, help="Bee ID (default: from env)")
    parser.add_argument(
        "--max-init-scans",
        type=int,
        default=None,
        help="Initial max concurrent scans (hive can override at runtime)",
    )
    parser.add_argument("--no-tls-verify", action="store_true", help="Disable TLS certificate verification")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Log level (default: INFO)",
    )
    return parser


def _build_config(args: Namespace) -> BeeConfig:
    """Build a BeeConfig from CLI args + environment variables."""
    overrides: dict[str, str | int | bool] = {}
    if args.hive_url is not None:
        overrides["hive_url"] = args.hive_url
    if args.api_key is not None:
        overrides["api_key"] = args.api_key
    if args.bee_id is not None:
        overrides["bee_id"] = args.bee_id
    if args.max_init_scans is not None:
        overrides["max_init_concurrent_scans"] = args.max_init_scans
    if args.no_tls_verify:
        overrides["tls_verify"] = False
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    return BeeConfig(**overrides)  # type: ignore[arg-type]


def main() -> None:
    """Main entry point for the bbot-bee CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    config = _build_config(args)

    log_level = getLevelName(config.log_level)
    if not isinstance(log_level, int):
        # Support custom levels like TRACE=5
        log_level = int(config.log_level) if config.log_level.isdigit() else INFO
    basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info(f"Starting bbot-bee {config.bee_id}")
    log.debug(f"Config: hive_url={config.hive_url}, max_init_scans={config.max_init_concurrent_scans}")

    queen = Queen(config)

    async def _run() -> None:
        loop = get_running_loop()
        for sig in (SIGTERM, SIGINT):
            loop.add_signal_handler(sig, lambda: create_task(queen.shutdown()))
        await queen.run()

    run(_run())


if __name__ == "__main__":
    main()
