"""Tests for bbot_bee.cgroup — per-scan cgroup v2 isolation.

Replaces the brittle ``/proc/1/task/1/children`` sweep (see
``BUG_REPORT_kill_orphaned_children.md``, 2026-04-16, which SIGKILLed the
user's ``systemd --user`` on the host) with kernel-authoritative termination
via ``cgroup.kill``. Each scan lives in
``/sys/fs/cgroup/bbot-scans/scan_<scan_id>``; writing ``1`` to
``<cgroup>/cgroup.kill`` atomically SIGKILLs every process in the cgroup,
including grandchildren that escaped the process group, processes ignoring
signals, and processes being forked into the cgroup.

Tests use real ``tmp_path`` directories for happy paths and ``monkeypatch``
for failure injection (EROFS / EBUSY / ENOENT). The kernel's
auto-population of ``cgroup.procs`` / ``cgroup.kill`` inside a freshly
created cgroup is simulated by wrapping ``Path.mkdir``.
"""

from __future__ import annotations

from collections.abc import Callable
from errno import EBUSY, EROFS
from pathlib import Path

import pytest

from bbot_bee import cgroup as cgroup_module
from bbot_bee.cgroup import (
    ScanCgroup,
    detect_cgroup_kill_supported,
    reap_zombies,
    recover_orphan_cgroups,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_cgroup_root(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    kernel_kill: bool = True,
) -> Path:
    """Point the cgroup module at ``root`` and simulate kernel behavior on mkdir.

    In a real cgroup v2 tree, the kernel automatically creates ``cgroup.procs``,
    ``cgroup.events``, and (on Linux 5.14+) ``cgroup.kill`` inside every
    freshly-made cgroup directory. We replicate that here so the tests touch
    real directories on ``tmp_path`` rather than mocking every Path call.
    """
    parent = root / "bbot-scans"
    monkeypatch.setattr(cgroup_module, "_CGROUP_MOUNT", root)
    monkeypatch.setattr(cgroup_module, "_PARENT", parent)

    real_mkdir = Path.mkdir
    real_rmdir = Path.rmdir

    _interface_files = ("cgroup.procs", "cgroup.events", "cgroup.kill")

    def kernel_mkdir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        real_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)
        if str(self).startswith(str(root)) and self.is_dir():
            procs = self / "cgroup.procs"
            if not procs.exists():
                procs.write_text("")
            events = self / "cgroup.events"
            if not events.exists():
                events.write_text("populated 0\nfrozen 0\n")
            if kernel_kill:
                kill = self / "cgroup.kill"
                if not kill.exists():
                    kill.write_text("0")

    def kernel_rmdir(self: Path) -> None:
        """In a real cgroup tree, ``cgroup.procs``/``cgroup.kill``/etc. are
        virtual interface files — they don't block rmdir. Simulate that
        by clearing them before delegating to real rmdir, but only if
        no child *cgroups* (subdirectories) are present.
        """
        if str(self).startswith(str(root)) and self.is_dir():
            has_subdir = any(p.is_dir() for p in self.iterdir())
            if not has_subdir:
                for name in _interface_files:
                    iface = self / name
                    if iface.exists():
                        iface.unlink()
        real_rmdir(self)

    monkeypatch.setattr(Path, "mkdir", kernel_mkdir)
    monkeypatch.setattr(Path, "rmdir", kernel_rmdir)
    # Top-level cgroup.controllers — only the unified v2 hierarchy has this.
    (root / "cgroup.controllers").write_text("cpu memory pids\n")
    return parent


def _drain_procs_after(path: Path, *, after_calls: int) -> Callable[[], None]:
    """Returns a callable that, on the Nth invocation, empties ``cgroup.procs``.

    Used to test ``wait_empty`` — simulates the kernel reaping killed processes
    out of the cgroup after some number of poll iterations.
    """
    calls = {"n": 0}

    def reaper() -> None:
        calls["n"] += 1
        if calls["n"] >= after_calls:
            (path / "cgroup.procs").write_text("")

    return reaper


# ---------------------------------------------------------------------------
# detect_cgroup_kill_supported
# ---------------------------------------------------------------------------


class TestDetect:
    """Startup probe for cgroup v2 + cgroup.kill availability.

    Bee refuses to boot if this returns False — no override. Tests pin the
    detection logic so a regression cannot let the bee silently start in
    a degraded mode.
    """

    def test_returns_true_when_v2_and_cgroup_kill_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Happy path: writable v2 tree with cgroup.kill (kernel 5.14+)."""
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        assert detect_cgroup_kill_supported() is True

    def test_returns_false_when_cgroup_controllers_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """cgroup v1 host has no /sys/fs/cgroup/cgroup.controllers."""
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        (tmp_path / "cgroup.controllers").unlink()
        assert detect_cgroup_kill_supported() is False

    def test_returns_false_when_cgroup_kill_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Kernel <5.14 doesn't auto-create cgroup.kill on mkdir."""
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=False)
        assert detect_cgroup_kill_supported() is False

    def test_returns_false_when_mkdir_fails_erofs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Read-only cgroup mount (hardened cluster) → False."""
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        real_mkdir = Path.mkdir

        def rofs_mkdir(
            self: Path,
            mode: int = 0o777,
            parents: bool = False,
            exist_ok: bool = False,
        ) -> None:
            if str(self).startswith(str(tmp_path)) and "bbot-detect-" in self.name:
                raise OSError(EROFS, "Read-only file system")
            real_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "mkdir", rofs_mkdir)
        assert detect_cgroup_kill_supported() is False

    def test_probe_cleans_up_after_itself(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A successful detect must rmdir its probe cgroup."""
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        assert detect_cgroup_kill_supported() is True
        leftovers = [p for p in tmp_path.iterdir() if "bbot-detect-" in p.name]
        assert leftovers == [], f"detect left probe dirs behind: {leftovers}"


# ---------------------------------------------------------------------------
# recover_orphan_cgroups
# ---------------------------------------------------------------------------


class TestRecoverOrphanCgroups:
    """Run at Queen startup to reclaim cgroups + processes from a crashed bee.

    Without this, a `kill -9` of the bee mid-scan leaves stale cgroup dirs
    plus possibly-still-alive scan descendants in the pod. Over time the
    pod's cgroup tree accumulates dead state.
    """

    def test_returns_zero_when_parent_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """First-ever run: no parent dir, nothing to recover."""
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        assert recover_orphan_cgroups() == 0

    def test_returns_zero_when_parent_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clean restart: parent exists but holds no scan cgroups."""
        parent = _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        parent.mkdir()
        assert recover_orphan_cgroups() == 0

    def test_kills_and_rmdirs_each_orphan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Crash recovery: each leftover scan_* dir gets its procs killed + rmdir'd."""
        parent = _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        parent.mkdir()
        for sid in ("aaa", "bbb", "ccc"):
            (parent / f"scan_{sid}").mkdir()

        assert recover_orphan_cgroups() == 3
        # cgroup.kill in each orphan must have been written
        for sid in ("aaa", "bbb", "ccc"):
            assert not (parent / f"scan_{sid}").exists()

    def test_skips_non_scan_directories(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Defensive: only directories prefixed ``scan_`` are reclaimed.

        Avoids touching anything else that might be in /sys/fs/cgroup/bbot-scans/
        (e.g. a future sibling component using the same parent).
        """
        parent = _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        parent.mkdir()
        (parent / "scan_keep").mkdir()
        (parent / "other_thing").mkdir()

        assert recover_orphan_cgroups() == 1
        assert not (parent / "scan_keep").exists()
        assert (parent / "other_thing").exists()


# ---------------------------------------------------------------------------
# ScanCgroup — constructor / scan_id validation
# ---------------------------------------------------------------------------


class TestScanCgroupInit:
    """scan_id is opaque from the wire — validate strictly even though the
    hive currently always sends UUIDs. Defense in depth.
    """

    @pytest.mark.parametrize(
        "scan_id",
        [
            "abc123",
            "scan-with-dashes",
            "scan_with_underscores",
            "a" * 64,
            "deadbeef-1234-5678-9abc-def012345678",  # uuid4 shape
        ],
    )
    def test_accepts_safe_scan_ids(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        scan_id: str,
    ) -> None:
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup(scan_id)
        assert scan_id in str(sc.path)

    @pytest.mark.parametrize(
        "scan_id",
        [
            "../foo",
            "scan/sub",
            "with space",
            "with.dot",
            "",
            "a" * 65,  # too long
            "with$shell",
            "with;injection",
        ],
    )
    def test_rejects_unsafe_scan_ids(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        scan_id: str,
    ) -> None:
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        with pytest.raises(ValueError, match="scan_id"):
            ScanCgroup(scan_id)

    def test_path_is_under_parent_with_scan_prefix(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Visual distinction: every cgroup dir name starts with ``scan_``."""
        parent = _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("abc123")
        assert sc.path == parent / "scan_abc123"

    def test_env_exposes_path_for_self_enroll(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``BBOT_BEE_SCAN_CGROUP`` is read by scan_process.py to self-enroll."""
        parent = _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("abc123")
        assert sc.env == {"BBOT_BEE_SCAN_CGROUP": str(parent / "scan_abc123")}


# ---------------------------------------------------------------------------
# ScanCgroup — lifecycle: create / populate / kill / cleanup
# ---------------------------------------------------------------------------


class TestScanCgroupCreate:
    def test_create_makes_cgroup_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("alpha")
        sc.create()
        assert sc.path.is_dir()

    def test_create_makes_parent_if_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parent ``/sys/fs/cgroup/bbot-scans/`` is created on demand."""
        parent = _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        assert not parent.exists()
        sc = ScanCgroup("beta")
        sc.create()
        assert parent.is_dir()
        assert sc.path.is_dir()

    def test_create_collision_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """scan_id reuse is a bug worth surfacing loudly."""
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        ScanCgroup("dup").create()
        with pytest.raises(FileExistsError):
            ScanCgroup("dup").create()


class TestScanCgroupPopulate:
    def test_populate_writes_pid_to_cgroup_procs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("gamma")
        sc.create()
        sc.populate(12345)
        assert (sc.path / "cgroup.procs").read_text().strip() == "12345"


class TestScanCgroupKill:
    """``kill()`` writes "1" to cgroup.kill — kernel atomically SIGKILLs
    every member, including descendants and processes ignoring signals."""

    def test_kill_writes_one_to_cgroup_kill(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("delta")
        sc.create()
        (sc.path / "cgroup.procs").write_text("1001\n1002\n1003\n")
        sc.kill()
        assert (sc.path / "cgroup.kill").read_text().strip() == "1"

    def test_kill_returns_proc_count_seen_before_kill(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """For logging: how many procs did we just signal?"""
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("epsilon")
        sc.create()
        (sc.path / "cgroup.procs").write_text("100\n200\n300\n")
        assert sc.kill() == 3

    def test_kill_idempotent_on_missing_cgroup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """kill() after the cgroup is already gone must NOT raise — it is
        called from multiple paths (force stop, failsafe, monitor cleanup)
        any of which may run after the cgroup has been removed by another.
        """
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("zeta")
        # Never call create().
        sc.kill()  # must not raise
        # Also: kill on a populated-then-removed cgroup
        sc2 = ScanCgroup("eta")
        sc2.create()
        sc2.cleanup()
        sc2.kill()  # must not raise


class TestScanCgroupWaitEmpty:
    """wait_empty polls cgroup.procs until empty + reaps zombies each tick.

    Without the reap call, the bee's own zombie children (the scan_process
    parent) keep them listed in cgroup.procs even after SIGKILL — wait_empty
    times out spuriously.
    """

    def test_returns_true_when_already_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("theta")
        sc.create()
        assert sc.wait_empty(timeout_s=0.5) is True

    def test_returns_true_when_drains_within_timeout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("iota")
        sc.create()
        (sc.path / "cgroup.procs").write_text("999\n")

        drainer = _drain_procs_after(sc.path, after_calls=2)
        monkeypatch.setattr(cgroup_module, "reap_zombies", drainer)
        assert sc.wait_empty(timeout_s=0.5) is True

    def test_calls_reap_zombies_each_poll(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pins the contract: every iteration must drain zombies, or the
        cgroup can never empty for the bee's own children.
        """
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("kappa")
        sc.create()
        (sc.path / "cgroup.procs").write_text("999\n")
        calls = {"n": 0}

        def fake_reap() -> None:
            calls["n"] += 1
            if calls["n"] >= 3:
                (sc.path / "cgroup.procs").write_text("")

        monkeypatch.setattr(cgroup_module, "reap_zombies", fake_reap)
        sc.wait_empty(timeout_s=0.5)
        assert calls["n"] >= 3

    def test_returns_false_on_timeout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Never-draining cgroup: timeout → caller must leak the dir, not hang."""
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("lambda")
        sc.create()
        (sc.path / "cgroup.procs").write_text("999\n")
        monkeypatch.setattr(cgroup_module, "reap_zombies", lambda: None)
        assert sc.wait_empty(timeout_s=0.1) is False


class TestScanCgroupCleanup:
    """cleanup rmdirs the cgroup. Must tolerate EBUSY (leak + log) and
    ENOENT (already gone — no-op)."""

    def test_cleanup_removes_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("mu")
        sc.create()
        sc.cleanup()
        assert not sc.path.exists()

    def test_cleanup_silent_on_already_removed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("nu")
        # never created
        sc.cleanup()  # must not raise
        sc.create()
        sc.cleanup()
        sc.cleanup()  # second cleanup must also be a no-op

    def test_cleanup_leaks_on_ebusy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-empty cgroup → log WARNING, do NOT raise, leak the directory.
        The next bee startup will reclaim it via recover_orphan_cgroups.
        """
        _install_fake_cgroup_root(monkeypatch, tmp_path, kernel_kill=True)
        sc = ScanCgroup("xi")
        sc.create()
        real_rmdir = Path.rmdir

        def busy_rmdir(self: Path) -> None:
            if self == sc.path:
                raise OSError(EBUSY, "Directory not empty")
            real_rmdir(self)

        monkeypatch.setattr(Path, "rmdir", busy_rmdir)
        with caplog.at_level("WARNING", logger="bbot_bee.cgroup"):
            sc.cleanup()  # must not raise
        assert any("xi" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# reap_zombies — moved from drone.py; preserve behavior
# ---------------------------------------------------------------------------


class TestReapZombies:
    """reap_zombies must remain a safe-anywhere idempotent helper, used by
    both Drone._monitor_process and Queen._event_flush_loop."""

    def test_no_children_is_noop(self) -> None:
        """Calling with no zombie children must not raise."""
        reap_zombies()  # must not raise

    def test_can_be_called_repeatedly(self) -> None:
        """Idempotent — multiple consecutive calls."""
        for _ in range(3):
            reap_zombies()
