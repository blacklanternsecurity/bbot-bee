"""Per-scan cgroup v2 isolation for bbot scan subprocesses.

Each scan_process subprocess and all its descendants (httpx, nuclei,
massdns, multiprocessing helpers) live in a dedicated cgroup at
``/sys/fs/cgroup/bbot-scans/scan_<scan_id>/``. Termination uses
``cgroup.kill`` (Linux 5.14+) — writing ``1`` atomically delivers SIGKILL
to every process in the cgroup, including processes that ignore signals,
descendants that escaped the process group, and processes still being
forked into the cgroup.

This replaces the pre-cgroup ``/proc/1/task/1/children`` orphan sweep that
crashed the user's desktop session on 2026-04-16 (see
``BUG_REPORT_kill_orphaned_children.md``) and the cmdline-string filter
that occasionally killed peer drones running concurrent scans.

Deployment requirement: Linux kernel 5.14+ with a writable cgroup v2
hierarchy at ``/sys/fs/cgroup``. In Kubernetes 1.25+ with cgroupns=private
and the systemd cgroup driver, this works via pod cgroup delegation with
no extra capabilities. The bee refuses to boot if detection fails.
"""

from __future__ import annotations

from contextlib import suppress
from errno import EBUSY, ENOENT
from logging import getLogger
from os import WNOHANG, waitpid
from pathlib import Path
from re import compile as re_compile
from time import monotonic, sleep
from uuid import uuid4

log = getLogger(__name__)

# Cgroup v2 unified-hierarchy mount point. Overridable in tests via
# ``monkeypatch.setattr(cgroup_module, "_CGROUP_MOUNT", ...)``.
_CGROUP_MOUNT = Path("/sys/fs/cgroup")

# Parent under which every per-scan cgroup is created.
_PARENT = _CGROUP_MOUNT / "bbot-scans"

# scan_id must be filesystem-safe and short. UUIDs (36 chars, hex+dashes)
# pass; anything with ``..``, ``/``, whitespace, shell metachars, or empty
# is rejected at ScanCgroup construction.
_SCAN_ID_RE = re_compile(r"^[A-Za-z0-9_-]{1,64}$")

# wait_empty polling: 10s total, 50ms between checks. The 10s window is
# generous — killing dozens of httpx workers + bbot helpers typically
# completes in well under 1s.
_WAIT_EMPTY_TIMEOUT_S = 10.0
_WAIT_EMPTY_POLL_INTERVAL_S = 0.05


# ---------------------------------------------------------------------------
# Zombie reaping (moved here from drone.py so wait_empty can call it
# without a circular import; drone.py and queen.py import from here).
# ---------------------------------------------------------------------------


def reap_zombies() -> None:
    """Reap any zombie children with ``waitpid(-1, WNOHANG)``.

    Safe to call from any process — only reaps actual children of the
    caller. After ``cgroup.kill`` fires, descendants exit and become
    zombies until the parent (the bee, typically PID 1 in its pod)
    reaps them. Zombies still count as members of ``cgroup.procs`` until
    reaped, so ``wait_empty`` calls this on every poll iteration.
    """
    reaped = 0
    while True:
        try:
            pid, _ = waitpid(-1, WNOHANG)
            if pid == 0:
                break
            reaped += 1
        except ChildProcessError:
            break
    if reaped:
        log.debug(f"reap_zombies: reaped {reaped=} zombie processes")


# ---------------------------------------------------------------------------
# Detection — run once at Queen startup. Bee refuses to boot if False.
# ---------------------------------------------------------------------------


def detect_cgroup_kill_supported() -> bool:
    """Return True iff the bee can use ``cgroup.kill`` in this environment.

    Three things have to be true: (1) the unified cgroup v2 hierarchy is
    mounted at ``_CGROUP_MOUNT`` (signaled by ``cgroup.controllers`` at
    the mount root), (2) the bee can ``mkdir`` a child cgroup (writable
    delegation), (3) the kernel auto-populates ``cgroup.kill`` inside
    new cgroups (Linux 5.14+).

    Probes by creating a uniquely-named temp child cgroup under the mount
    root, checking for the kernel-managed ``cgroup.kill`` file, then
    rmdir'ing. Any failure → False; no exception escapes.
    """
    log.debug(f"detect_cgroup_kill_supported: probing {_CGROUP_MOUNT=}")

    controllers = _CGROUP_MOUNT / "cgroup.controllers"
    if not controllers.exists():
        log.warning(f"detect_cgroup_kill_supported: {controllers=} missing — not a cgroup v2 mount")
        return False

    probe = _CGROUP_MOUNT / f"bbot-detect-{uuid4().hex[:8]}"
    try:
        try:
            probe.mkdir()
        except OSError as exc:
            log.warning(f"detect_cgroup_kill_supported: cannot mkdir {probe=}: {exc!r}")
            return False

        if not (probe / "cgroup.kill").exists():
            log.warning(f"detect_cgroup_kill_supported: {probe / 'cgroup.kill'} missing — kernel <5.14")
            return False

        log.info(f"detect_cgroup_kill_supported: cgroup v2 cgroup.kill available at {_CGROUP_MOUNT}")
        return True
    finally:
        with suppress(OSError):
            probe.rmdir()


# ---------------------------------------------------------------------------
# Crash recovery — reclaim cgroups left over from a previous bee process.
# ---------------------------------------------------------------------------


def recover_orphan_cgroups() -> int:
    """Reclaim any ``scan_*`` cgroups left under ``_PARENT`` by a crashed bee.

    For each leftover directory: write ``1`` to ``cgroup.kill`` (atomically
    SIGKILL any processes still inside), wait briefly, then rmdir. Only
    touches directories prefixed ``scan_`` — defensive against unrelated
    siblings appearing under the parent.

    Returns the number of orphan cgroups recovered. Returns 0 if the
    parent directory doesn't exist (clean first run).
    """
    log.debug(f"recover_orphan_cgroups: scanning {_PARENT=}")
    if not _PARENT.exists():
        log.debug("recover_orphan_cgroups: parent absent, nothing to recover")
        return 0

    recovered = 0
    for child in _PARENT.iterdir():
        if not child.is_dir() or not child.name.startswith("scan_"):
            log.debug(f"recover_orphan_cgroups: skipping {child.name=} (not a scan cgroup)")
            continue

        log.warning(f"recover_orphan_cgroups: recovering orphan {child}")
        kill_file = child / "cgroup.kill"
        if kill_file.exists():
            with suppress(OSError):
                kill_file.write_text("1")
        _wait_cgroup_drained(child, timeout_s=2.0)
        try:
            child.rmdir()
            recovered += 1
        except OSError as exc:
            log.warning(f"recover_orphan_cgroups: rmdir {child=} failed: {exc!r} — leaking")

    if recovered:
        log.warning(f"recover_orphan_cgroups: recovered {recovered=} orphan cgroups")
    return recovered


def _wait_cgroup_drained(cgroup_dir: Path, *, timeout_s: float) -> bool:
    """Poll ``cgroup.procs`` until empty or timeout, reaping zombies each tick.

    Internal helper shared by ``recover_orphan_cgroups`` and
    ``ScanCgroup.wait_empty`` — same loop, same semantics.
    """
    procs = cgroup_dir / "cgroup.procs"
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        reap_zombies()
        try:
            contents = procs.read_text()
        except FileNotFoundError:
            return True
        if not contents.strip():
            return True
        sleep(_WAIT_EMPTY_POLL_INTERVAL_S)
    return False


# ---------------------------------------------------------------------------
# ScanCgroup — one per Drone, drives the per-scan cgroup lifecycle.
# ---------------------------------------------------------------------------


class ScanCgroup:
    """Owns the cgroup for a single scan.

    Lifecycle (called from ``Drone``):

    1. ``create()`` — mkdir the cgroup directory.
    2. ``populate(pid)`` — move the scan_process PID into the cgroup
       immediately after subprocess spawn, before any forking.
    3. ``kill()`` — write ``1`` to ``cgroup.kill``. Kernel atomically
       delivers SIGKILL to every process in the cgroup, including
       descendants of descendants. Idempotent.
    4. ``wait_empty()`` — block until ``cgroup.procs`` empties (reaping
       zombies on each poll).
    5. ``cleanup()`` — rmdir. Tolerant of EBUSY (leak) and ENOENT
       (already gone).

    All methods are safe to call from synchronous contexts including
    ``loop.call_later`` callbacks — there is no async work.
    """

    def __init__(self, scan_id: str) -> None:
        """Construct a ScanCgroup for ``scan_id``.

        Raises ValueError if ``scan_id`` contains anything outside the
        ``[A-Za-z0-9_-]{1,64}`` allowlist — defensive against path
        traversal even though the hive's scan_id is currently a UUID.
        """
        if not _SCAN_ID_RE.match(scan_id):
            raise ValueError(
                f"Invalid scan_id={scan_id!r}: must match [A-Za-z0-9_-]{{1,64}}",
            )
        self._scan_id = scan_id
        self._path = _PARENT / f"scan_{scan_id}"
        log.debug(f"ScanCgroup.__init__: {scan_id=}, path={self._path}")

    @property
    def path(self) -> Path:
        """Full path to this scan's cgroup directory."""
        return self._path

    @property
    def env(self) -> dict[str, str]:
        """Env vars to hand to the scan subprocess so it can self-enroll.

        ``scan_process.main()`` reads ``BBOT_BEE_SCAN_CGROUP`` and writes
        its own PID to ``<path>/cgroup.procs`` before any other startup,
        closing the spawn→populate race window.
        """
        return {"BBOT_BEE_SCAN_CGROUP": str(self._path)}

    def create(self) -> None:
        """``mkdir`` the cgroup directory (and parent on first call).

        Raises ``FileExistsError`` on collision — scan_id reuse is a bug,
        not something to silently paper over.
        """
        log.info(f"ScanCgroup.create: {self._scan_id=}, path={self._path}")
        _PARENT.mkdir(parents=True, exist_ok=True)
        self._path.mkdir(parents=False, exist_ok=False)

    def populate(self, pid: int) -> None:
        """Move ``pid`` into this cgroup by writing it to ``cgroup.procs``.

        Must be called between subprocess spawn and any other work the
        subprocess may do (e.g. forking helpers). Idempotent with the
        child's own self-enroll on ``BBOT_BEE_SCAN_CGROUP``.
        """
        log.info(f"ScanCgroup.populate: {self._scan_id=}, {pid=}")
        (self._path / "cgroup.procs").write_text(f"{pid}\n")

    def kill(self) -> int:
        """Write ``1`` to ``cgroup.kill``. Atomic kernel SIGKILL to all
        processes in the cgroup (including descendants and processes
        being forked in).

        Returns the count of processes in ``cgroup.procs`` at call time
        (read just before the kill, for logging). Returns 0 if the
        cgroup is already gone — idempotent on the kill side; safe to
        call from any of ``stop(force=True)``, ``_force_kill_runaway``,
        or ``_monitor_process`` cleanup, in any order, any number of
        times.

        Single ``write(2)`` syscall on the user side — non-blocking,
        safe inside synchronous event-loop callbacks.
        """
        kill_file = self._path / "cgroup.kill"
        if not kill_file.exists():
            log.debug(f"ScanCgroup.kill: {self._scan_id=}, cgroup already gone — no-op")
            return 0

        try:
            count = len([line for line in (self._path / "cgroup.procs").read_text().splitlines() if line.strip()])
        except OSError:
            count = 0

        log.warning(
            f"ScanCgroup.kill: {self._scan_id=}, atomic SIGKILL to {count=} procs in cgroup",
        )
        try:
            kill_file.write_text("1")
        except OSError as exc:
            log.warning(f"ScanCgroup.kill: {self._scan_id=}, write failed: {exc!r}")
            return 0
        return count

    def wait_empty(self, timeout_s: float = _WAIT_EMPTY_TIMEOUT_S) -> bool:
        """Block until ``cgroup.procs`` is empty (or timeout).

        Reaps zombies on each poll iteration via ``reap_zombies()`` —
        zombies count as members of ``cgroup.procs`` until reaped, so
        without this the cgroup never empties for descendants of the
        bee itself.

        Returns True if the cgroup drained, False on timeout. On
        timeout the caller should leak the cgroup directory and let
        the next ``recover_orphan_cgroups()`` reclaim it.
        """
        log.debug(f"ScanCgroup.wait_empty: {self._scan_id=}, {timeout_s=}")
        drained = _wait_cgroup_drained(self._path, timeout_s=timeout_s)
        if not drained:
            log.warning(
                f"ScanCgroup.wait_empty: {self._scan_id=} did not drain within "
                f"{timeout_s=}s — leaking directory; next bee startup will reclaim",
            )
        return drained

    def cleanup(self) -> None:
        """``rmdir`` the cgroup directory. Tolerant of:

        - ENOENT (already gone): silent no-op.
        - EBUSY (non-empty): WARNING log + leak; recovered on next bee startup.

        Never raises — callers in cleanup paths shouldn't have to guard.
        """
        log.debug(f"ScanCgroup.cleanup: {self._scan_id=}, path={self._path}")
        try:
            self._path.rmdir()
        except OSError as exc:
            if exc.errno == ENOENT:
                log.debug(f"ScanCgroup.cleanup: {self._scan_id=}, already gone")
                return
            if exc.errno == EBUSY:
                log.warning(
                    f"ScanCgroup.cleanup: {self._scan_id=}, cgroup not empty — leaking "
                    f"directory {self._path}; will be reclaimed on next bee startup",
                )
                return
            log.warning(f"ScanCgroup.cleanup: {self._scan_id=}, unexpected rmdir error: {exc!r}")
