"""Subprocess entry point — runs a bbot Scanner in isolation with stdin/stdout IPC."""

from __future__ import annotations

from asyncio import CancelledError, StreamReader, StreamReaderProtocol, create_task, get_running_loop, run
from contextlib import suppress
from logging import DEBUG, Formatter, StreamHandler, getLevelName, getLogger
from os import environ, getpid
from pathlib import Path
from signal import SIGTERM
from sys import exit as sys_exit
from sys import stderr, stdin, stdout
from typing import Any

from bbot.constants import get_scan_status_code
from bbot.scanner import Preset, Scanner
from bbot.scanner.dispatcher import Dispatcher
from orjson import JSONDecodeError, dumps, loads

log = getLogger(__name__)


def _configure_logging() -> None:
    """Configure Python logging to write to stderr."""
    handler = StreamHandler(stderr)
    handler.setFormatter(Formatter("%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s"))
    root = getLogger()
    root.addHandler(handler)
    root.setLevel(DEBUG)
    log.debug("logging configured to stderr")


def _write_stdout(obj: dict[str, object]) -> None:
    """Write a JSON object as a single line to stdout and flush."""
    data = dumps(obj, default=str) + b"\n"
    stdout.buffer.write(data)
    stdout.buffer.flush()


# bbot's Dispatcher base class is untyped, so mypy flags the subclass.
class _SubprocessDispatcher(Dispatcher):  # type: ignore[misc]
    """Custom bbot Dispatcher that writes status changes to stdout as JSON lines."""

    async def on_status(self, status: str, scan_id: str) -> None:
        """Write a status change to stdout as a JSON line."""
        code = get_scan_status_code(status)
        log.debug(f"_SubprocessDispatcher.on_status: {status=}, {scan_id=}, {code=}")
        _write_stdout({"_type": "status", "status": status, "status_code": code})

    async def on_start(self, scan: object) -> None:
        """Called when the scan starts."""
        log.info("_SubprocessDispatcher.on_start: scan started")

    async def on_finish(self, scan: object) -> None:
        """Called when the scan finishes."""
        log.info("_SubprocessDispatcher.on_finish: scan finished")


def _handle_kill_module(scanner: Scanner, payload: dict[str, Any]) -> dict[str, Any]:
    """Kill a named module, refusing if the module has ``_intercept=True``."""
    module_name = payload.get("module_name", "")
    message = payload.get("message")
    log.info(f"{module_name=}, {message=}")

    if module_name not in scanner.modules:
        return {"success": False, "error": f"Module '{module_name}' not found"}

    if getattr(scanner.modules[module_name], "_intercept", False):
        return {"success": False, "error": f"Module '{module_name}' is critical and cannot be killed"}

    scanner.kill_module(module_name, message)
    return {"success": True}


def _handle_set_log_level(scanner: Scanner, payload: dict[str, Any]) -> dict[str, Any]:
    level = payload.get("level", "INFO")
    log.info(f"{level=}")
    scanner.core.logger.set_log_level(level)
    return {"success": True, "level": getLevelName(scanner.log_level)}


def _handle_toggle_log_level(scanner: Scanner, payload: dict[str, Any]) -> dict[str, Any]:
    log.info("_handle_toggle_log_level")
    scanner.core.logger.toggle_log_level()
    return {"success": True, "level": getLevelName(scanner.log_level)}


def _handle_scan_status(scanner: Scanner, payload: dict[str, Any]) -> dict[str, Any]:
    """Handle scan_status query — comprehensive scan snapshot."""
    log.debug("gathering scan status")
    stats = {
        "overall": {
            "speed": scanner.stats.speedometer.speed,
            "total_emitted_events": sum(scanner.stats.events_emitted_by_type.values()),
            "total_queued_events": scanner.num_queued_events,
            "event_types": scanner.stats.events_emitted_by_type,
            "aborting": scanner.aborting,
            "stopping": scanner.stopping,
            "start_time": scanner.start_time.isoformat()
        },
        "modules": {}
    }
    mod_stats = scanner.stats.module_stats
    for name, mod in ((n, m) for n, m in scanner.modules.items() if not n.startswith("_")):
        incoming: int = mod.status["events"]["incoming"]
        processing: int = mod.status["tasks"]
        outgoing: int = mod.status["events"]["outgoing"]
        mod_stat = {
            "running": mod.running,
            "errored": mod.errored,
            "incoming": incoming,
            "processing": processing,
            "outgoing": outgoing,
            "pipeline_total": incoming + processing + outgoing
        }

        if prod_cons := mod_stats.get(name):
            mod_stat.update({
                "consumed": {"total": prod_cons.consumed_total, "event_counts": prod_cons.consumed},
                "produced": {"total": prod_cons.produced_total, "event_counts": prod_cons.produced}
            })

        stats["modules"][name] = mod_stat
    return stats

    # return {
    #     "modules_status": scanner.modules_status(),
    #     "events_by_type": dict(scanner.stats.events_emitted_by_type),
    #     "speed": scanner.stats.speedometer.speed,
    #     "module_stats": {
    #         name: {
    #             "produced": stat.produced,
    #             "produced_total": stat.produced_total,
    #             "consumed": stat.consumed,
    #             "consumed_total": stat.consumed_total,
    #         }
    #         for name, stat in scanner.stats.module_stats.items()
    #     },
    #     "num_queued_events": scanner.num_queued_events,
    #     "log_level": getLevelName(scanner.log_level),
    #     "scan": scanner.json,
    #     "duration_seconds": scanner.duration_seconds,
    #     "duration_human": scanner.duration_human,
    #     "running": scanner.running,
    #     "modules_finished": scanner.modules_finished,
    #     "status": scanner.status,
    # }


def _handle_scope_check(scanner: Scanner, payload: dict[str, Any]) -> dict[str, Any]:
    host = payload.get("host", "")
    log.debug(f"{host=}")
    if not host:
        return {"error": "Missing 'host' parameter"}
    return {
        "host": host,
        "in_scope": scanner.in_scope(host),
        "in_target": scanner.in_target(host),
        "blacklisted": scanner.blacklisted(host),
    }


def _handle_scan_config(scanner: Scanner, payload: dict[str, Any]) -> dict[str, Any]:
    log.debug("gathering scan config")
    return {
        "modules": sorted(scanner.preset.modules),
        "target": scanner.target.json,
        "omitted_event_types": list(scanner.omitted_event_types),
        "web_config": dict(scanner.web_config) if scanner.web_config else {},
        "scope_search_distance": scanner.scope_search_distance,
        "scope_report_distance": scanner.scope_report_distance,
    }


def _handle_list_modules(scanner: Scanner, payload: dict[str, Any]) -> dict[str, Any]:
    log.debug("gathering list of modules")
    return {
        "modules": sorted(scanner.preset.modules)
    }


_COMMAND_HANDLERS = {
    "kill_module": _handle_kill_module,
    "set_log_level": _handle_set_log_level,
    "toggle_log_level": _handle_toggle_log_level,
    "scan_status": _handle_scan_status,
    "scope_check": _handle_scope_check,
    "scan_config": _handle_scan_config,
    "list_modules": _handle_list_modules
}


async def _dispatch_command(scanner: Scanner, cmd_dict: dict[str, Any]) -> None:
    """Dispatch a runtime command to the appropriate handler and write the result to stdout."""
    cmd = cmd_dict.get("cmd", "")
    request_id = cmd_dict.get("request_id")
    handler = _COMMAND_HANDLERS.get(cmd)

    if handler is None:
        log.warning(f"unknown command: {cmd=}")
        result = {"_type": "cmd_result", "cmd": cmd, "success": False, "error": f"Unknown command: {cmd}"}
    else:
        try:
            data = handler(scanner, cmd_dict)
            result = {"_type": "cmd_result", "cmd": cmd, **data}
        except Exception as exc:
            log.error(f"{cmd} failed: {type(exc).__name__}: {exc}")
            result = {"_type": "cmd_result", "cmd": cmd, "success": False, "error": str(exc)}

    if request_id is not None:
        result["request_id"] = request_id

    _write_stdout(result)


async def _stdin_command_reader(scanner: Scanner, reader: StreamReader) -> None:
    """Read JSON command lines from stdin and dispatch to the scanner."""
    log.info("listening for commands on stdin")
    while True:
        raw_line = await reader.readline()
        if not raw_line:
            log.debug("stdin closed (EOF)")
            break
        line = raw_line.strip()
        if not line:
            continue
        try:
            cmd_dict = loads(line)
        except JSONDecodeError as exc:
            log.warning(f"invalid JSON: {exc}")
            continue
        await _dispatch_command(scanner, cmd_dict)
    log.info("exiting")


async def _run_scan(scan_id: str, preset_dict: dict[str, Any], stdin_reader: StreamReader) -> int:
    """Run a bbot scan, stream results, and process commands.

    Args:
        scan_id: The scan's unique identifier.
        preset_dict: BBOT preset configuration dict.
        stdin_reader: Async reader for stdin command channel.

    Returns:
        Exit code: 0=finished, 1=failed, 2=aborted.
    """
    log.info(f"{scan_id=} — building scanner from preset")

    # Default to deps.behavior=disable since the bee image ships with all
    # module deps pre-installed (bbot --install-all-deps in Dockerfile).
    # Without this, concurrent scans race on the shared ansible artifact
    # directory and crash. The preset can override if needed.
    config = preset_dict.setdefault("config", {})
    config.setdefault("deps", {}).setdefault("behavior", "disable")
    log.debug(f"{scan_id=} — deps.behavior={config['deps']['behavior']}")

    try:
        preset_obj = Preset.from_dict(preset_dict)
    except Exception as exc:
        log.error(f"{scan_id=} — failed to create preset: {type(exc).__name__}: {exc}")
        return 1

    dispatcher = _SubprocessDispatcher()

    try:
        scanner = Scanner(preset=preset_obj, scan_id=scan_id, dispatcher=dispatcher)
    except Exception as exc:
        log.error(f"{scan_id=} — failed to create scanner: {type(exc).__name__}: {exc}")
        return 1

    log.info(f"{scan_id=}, name={scanner.name!r} — scanner created")

    loop = get_running_loop()
    loop.add_signal_handler(SIGTERM, scanner.stop)
    log.debug(f"{scan_id=} — SIGTERM handler installed")

    cmd_task = create_task(
        _stdin_command_reader(scanner, stdin_reader),
        name=f"stdin-cmd-reader-{scan_id}",
    )

    log.info(f"{scan_id=} — starting scan iteration")
    try:
        async for event in scanner.async_start():
            event_dict = event.json()
            event_dict["_type"] = "event"
            _write_stdout(event_dict)
    except Exception as exc:
        log.error(f"{scan_id=} — scan error: {type(exc).__name__}: {exc}")
        cmd_task.cancel()
        return 1

    cmd_task.cancel()
    with suppress(CancelledError):
        await cmd_task

    final_status = scanner.status
    log.info(f"{scan_id=} — scan complete, {final_status=}")

    if final_status == "FINISHED":
        return 0
    elif final_status == "ABORTED":
        return 2
    else:
        return 1


async def _async_main() -> None:
    """Async entry point — sets up stdin pipe, reads preset, runs scan."""
    loop = get_running_loop()
    reader = StreamReader()
    await loop.connect_read_pipe(lambda: StreamReaderProtocol(reader), stdin.buffer)

    raw_line = await reader.readline()
    if not raw_line:
        log.error("empty stdin, no preset received")
        sys_exit(1)

    try:
        config = loads(raw_line)
    except JSONDecodeError as exc:
        log.error(f"failed to parse preset JSON: {type(exc).__name__}: {exc}")
        sys_exit(1)

    scan_id = config.get("scan_id")
    preset = config.get("preset")

    if not scan_id:
        log.error("missing 'scan_id' in stdin JSON")
        sys_exit(1)
    if preset is None:
        log.error("missing 'preset' in stdin JSON")
        sys_exit(1)

    log.info(f"{scan_id=} — running scan")
    exit_code = await _run_scan(scan_id, preset, reader)
    log.info(f"{scan_id=} — exiting with {exit_code=}")
    sys_exit(exit_code)


def _self_enroll_in_cgroup() -> None:
    """Write our own PID to ``$BBOT_BEE_SCAN_CGROUP/cgroup.procs`` if set.

    Belt-and-suspenders for the Drone-side ``populate(pid)`` write — closes
    the spawn → populate race window completely. Runs at the very top of
    ``main()`` before any other code so that even an interpreter that
    eagerly forks during startup is guaranteed to fork inside the cgroup.

    Best-effort: any error is logged to stderr but does NOT crash the scan.
    The parent's ``populate(pid)`` is the authoritative enrollment; this is
    redundant safety.
    """
    cgroup_path = environ.get("BBOT_BEE_SCAN_CGROUP")
    if not cgroup_path:
        return
    try:
        (Path(cgroup_path) / "cgroup.procs").write_text(f"{getpid()}\n")
    except OSError as exc:
        print(
            f"scan_process: BBOT_BEE_SCAN_CGROUP self-enroll failed: {exc!r}",
            file=stderr,
            flush=True,
        )


def main() -> None:
    """Main entry point for the scan subprocess."""
    _self_enroll_in_cgroup()
    _configure_logging()
    log.info("scan_process starting")
    run(_async_main())


if __name__ == "__main__":
    main()
