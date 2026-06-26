"""Shared test fixtures for bbot_bee.

The ``_FakeScanCgroup`` autouse fixture replaces the real cgroup machinery
across all unit tests so they don't try to mkdir under ``/sys/fs/cgroup``
(which fails without root). ``detect_cgroup_kill_supported`` and
``recover_orphan_cgroups`` are also stubbed to no-op so the bee boots in
tests on hosts without a writable cgroup v2 tree.

Tests that need to inspect cgroup interactions can access the recorded
calls via ``_FakeScanCgroup.instances``.
"""

from __future__ import annotations

import os
import signal
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest


class _FakeScanCgroup:
    """In-memory fake of ``bbot_bee.cgroup.ScanCgroup``.

    Records an ordered event log per instance so tests can pin the
    lifecycle contract (``create`` → ``populate`` → ``kill`` →
    ``wait_empty`` → ``cleanup``). ``kill()`` actually SIGKILL's any
    populated PIDs so production code paths that wait on subprocess
    exit still terminate in tests.
    """

    instances: list[_FakeScanCgroup] = []

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def __init__(self, scan_id: str) -> None:
        self.scan_id = scan_id
        self.path = Path(f"/sys/fs/cgroup/bbot-scans/scan_{scan_id}")
        self.events: list[tuple[str, Any]] = []
        self._pids: list[int] = []
        type(self).instances.append(self)

    @property
    def env(self) -> dict[str, str]:
        return {"BBOT_BEE_SCAN_CGROUP": str(self.path)}

    def create(self) -> None:
        self.events.append(("create", None))

    def populate(self, pid: int) -> None:
        self.events.append(("populate", pid))
        self._pids.append(pid)

    def kill(self) -> int:
        self.events.append(("kill", None))
        for pid in self._pids:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        return len(self._pids)

    def wait_empty(self, timeout_s: float = 10.0) -> bool:
        self.events.append(("wait_empty", timeout_s))
        return True

    def cleanup(self) -> None:
        self.events.append(("cleanup", None))


@pytest.fixture(autouse=True)
def _stub_cgroup(monkeypatch: pytest.MonkeyPatch) -> type[_FakeScanCgroup]:
    """Replace ScanCgroup + detect/recover with no-op test doubles."""
    _FakeScanCgroup.reset()
    monkeypatch.setattr("bbot_bee.drone.ScanCgroup", _FakeScanCgroup)
    monkeypatch.setattr("bbot_bee.cgroup.detect_cgroup_kill_supported", lambda: True)
    monkeypatch.setattr("bbot_bee.cgroup.recover_orphan_cgroups", lambda: 0)
    return _FakeScanCgroup
