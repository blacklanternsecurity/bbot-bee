"""Tests for bbot_bee.scan_process."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

_FAST_PRESET: dict[str, Any] = {
    "target": ["127.0.0.1"],
    "modules": [],
    "output_modules": [],
    "config": {
        "dns": {"minimal": True, "timeout": 1, "disable": True},
        "scope": {"search_distance": 0, "report_distance": 0},
        "speculate": False,
        "excavate": False,
        "aggregate": False,
        "cloudcheck": False,
    },
    "silent": True,
}


async def _run_scan_process(
    scan_id: str,
    preset: dict[str, Any],
    timeout_s: float = 30.0,
) -> tuple[list[dict[str, Any]], bytes, int]:
    """Run scan_process.py as a subprocess and collect output.

    Returns:
        Tuple of (stdout_lines_parsed, stderr_bytes, exit_code).
    """
    payload = json.dumps({"scan_id": scan_id, "preset": preset}).encode()

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "bbot_bee.scan_process",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert proc.stdin is not None
    proc.stdin.write(payload)
    proc.stdin.close()

    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[dict[str, Any]] = []
    try:
        stdout_data, stderr_data = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_s,
        )
    except TimeoutError:
        proc.kill()
        stdout_data, stderr_data = await proc.communicate()

    for line in stdout_data.decode().splitlines():
        line = line.strip()
        if line:
            stdout_lines.append(json.loads(line))

    returncode = proc.returncode
    assert returncode is not None
    return stdout_lines, stderr_data, returncode


class TestScanProcessProtocol:
    """Tests for the stdout JSON line protocol."""

    async def test_outputs_status_messages(self) -> None:
        """scan_process should emit status messages with _type=status on stdout."""
        lines, _, _ = await _run_scan_process("test-protocol", _FAST_PRESET)

        status_lines = [msg for msg in lines if msg.get("_type") == "status"]
        assert len(status_lines) >= 2, f"Expected at least 2 status messages, got {len(status_lines)}"

        for msg in status_lines:
            assert "status" in msg, f"Missing 'status' field in {msg}"
            assert "status_code" in msg, f"Missing 'status_code' field in {msg}"

    async def test_status_includes_starting(self) -> None:
        """scan_process should emit a STARTING status."""
        lines, _, _ = await _run_scan_process("test-starting", _FAST_PRESET)

        statuses = [msg["status"] for msg in lines if msg.get("_type") == "status"]
        assert "STARTING" in statuses, f"Expected STARTING in {statuses}"

    async def test_events_have_type_field(self) -> None:
        """Events on stdout should have _type=event."""
        lines, _, _ = await _run_scan_process("test-events", _FAST_PRESET)

        event_lines = [msg for msg in lines if msg.get("_type") == "event"]
        for event in event_lines:
            assert "type" in event, f"Missing bbot 'type' field in event: {event}"

    async def test_status_codes_advance_forward(self) -> None:
        """Status codes should only increase (forward-only progression)."""
        lines, _, _ = await _run_scan_process("test-forward", _FAST_PRESET)

        status_codes = [msg["status_code"] for msg in lines if msg.get("_type") == "status"]
        assert len(status_codes) >= 2, f"Expected at least 2 status codes, got {status_codes}"

        for i in range(1, len(status_codes)):
            assert status_codes[i] > status_codes[i - 1], (
                f"Status code regressed: {status_codes[i - 1]} -> {status_codes[i]}"
            )


class TestScanProcessExitCodes:
    """Tests for subprocess exit codes."""

    async def test_successful_scan_exits_zero(self) -> None:
        """A successful scan should exit with code 0."""
        _, _, returncode = await _run_scan_process("test-exit-0", _FAST_PRESET)
        assert returncode == 0, f"Expected exit code 0, got {returncode}"

    async def test_bad_preset_exits_nonzero(self) -> None:
        """A bad preset with nonexistent modules should exit with code 1."""
        bad_preset = {"modules": ["nonexistent_module_xyz_123"]}
        _, stderr, returncode = await _run_scan_process("test-bad-preset", bad_preset)
        assert returncode == 1, f"Expected exit code 1, got {returncode}. stderr: {stderr.decode()[-500:]}"


class TestScanProcessSigterm:
    """Tests for SIGTERM handling."""

    async def test_sigterm_exits_with_code_2(self) -> None:
        """Sending SIGTERM during a scan should result in exit code 2 (aborted)."""
        slow_preset = dict(_FAST_PRESET)
        slow_preset["modules"] = ["http"]
        slow_preset["target"] = ["127.0.0.1"]

        payload = json.dumps({"scan_id": "test-sigterm", "preset": slow_preset}).encode()

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "bbot_bee.scan_process",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert proc.stdin is not None
        proc.stdin.write(payload)
        proc.stdin.close()

        assert proc.stdout is not None
        started = False
        for _ in range(100):
            try:
                raw_line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
                if raw_line:
                    data = json.loads(raw_line)
                    if data.get("_type") == "status" and data.get("status") in ("STARTING", "RUNNING"):
                        started = True
                        break
            except (TimeoutError, json.JSONDecodeError):
                continue

        assert started, "Scan never reached STARTING/RUNNING before timeout"

        os.kill(proc.pid, signal.SIGTERM)

        try:
            await asyncio.wait_for(proc.wait(), timeout=30.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()

        returncode = proc.returncode
        assert returncode == 2, f"Expected exit code 2 (aborted), got {returncode}"


class TestScanProcessStdin:
    """Tests for stdin input handling."""

    async def test_missing_scan_id_exits_nonzero(self) -> None:
        """Missing scan_id in stdin JSON should exit with code 1."""
        payload = json.dumps({"preset": _FAST_PRESET}).encode()

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "bbot_bee.scan_process",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        proc.stdin.write(payload)
        proc.stdin.close()

        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()

        assert proc.returncode != 0, f"Expected non-zero exit, got {proc.returncode}"

    async def test_invalid_json_exits_nonzero(self) -> None:
        """Invalid JSON on stdin should exit with code 1."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "bbot_bee.scan_process",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        proc.stdin.write(b"not valid json{{{")
        proc.stdin.close()

        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()

        assert proc.returncode != 0, f"Expected non-zero exit, got {proc.returncode}"


class TestScanProcessCgroupSelfEnroll:
    """Closes the spawn→populate race window in Drone.start() by enrolling on the child side at the top of main()."""

    async def test_self_enroll_writes_pid_when_env_set(self, tmp_path: Path) -> None:
        """``BBOT_BEE_SCAN_CGROUP=<dir>`` → child writes its PID to ``cgroup.procs`` before exiting on bad stdin."""
        procs_file = tmp_path / "cgroup.procs"
        procs_file.write_text("")

        env = {**os.environ, "BBOT_BEE_SCAN_CGROUP": str(tmp_path)}
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "bbot_bee.scan_process",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        assert proc.stdin is not None
        # Immediate EOF — scan_process will exit 1, but only after self-enroll.
        proc.stdin.close()

        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()

        written = procs_file.read_text().strip()
        assert written == str(proc.pid), f"expected self-enroll to write PID {proc.pid}, got {written!r}"

    async def test_no_env_no_action(self) -> None:
        """Without the env var, scan_process must behave exactly as before — self-enroll is a no-op."""
        env = {k: v for k, v in os.environ.items() if k != "BBOT_BEE_SCAN_CGROUP"}
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "bbot_bee.scan_process",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        assert proc.stdin is not None
        proc.stdin.close()

        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()

        assert proc.returncode != 0

    async def test_self_enroll_failure_does_not_crash(self, tmp_path: Path) -> None:
        """If the cgroup path is invalid, the child logs to stderr but continues without crashing."""
        nonexistent = tmp_path / "does-not-exist"
        env = {**os.environ, "BBOT_BEE_SCAN_CGROUP": str(nonexistent)}
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "bbot_bee.scan_process",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        assert proc.stdin is not None
        proc.stdin.close()

        try:
            _, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            _, stderr_data = await proc.communicate()

        assert proc.returncode is not None
        assert b"BBOT_BEE_SCAN_CGROUP" in stderr_data or b"cgroup" in stderr_data.lower()
