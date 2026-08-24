"""Per-scan cgroup v2 isolation for bbot scan subprocesses."""

from __future__ import annotations

from contextlib import suppress
from errno import EBUSY, ENOENT
from logging import getLogger
from os import ST_RDONLY, WNOHANG, statvfs, waitpid
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
# generous — killing the scan's process tree (nuclei, massdns, bbot
# multiprocessing helpers) typically completes in well under 1s.
_WAIT_EMPTY_TIMEOUT_S = 10.0
_WAIT_EMPTY_POLL_INTERVAL_S = 0.05
# Reason returned alongside False from detect_cgroup_kill_supported when the
# /sys/fs/cgroup mount is read-only — the dominant failure mode in plain
# Docker containers without ``--security-opt writable-cgroups=true`` and in
# K8s pods without a ``securityContext`` that delegates the cgroup. Queen.py
# uses this to emit a deployment-specific fix message instead of a generic
# "cannot mkdir" log.
DETECT_REASON_RO_CGROUPFS = "ro_cgroupfs"
DETECT_REASON_NOT_CGROUP_V2 = "not_cgroup_v2"
DETECT_REASON_NO_DELEGATION = "no_delegation"
DETECT_REASON_KERNEL_TOO_OLD = "kernel_too_old"


def reap_zombies() -> None:
    """Reap any zombie children with `waitpid(-1, WNOHANG)`.

    After `cgroup.kill` fires, descendants exit and become zombies until the
    parent (the bee, typically PID 1 in its pod) reaps them. Zombies still
    count as members of `cgroup.procs` until reaped, so `wait_empty` calls
    this on every poll iteration.
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
        log.debug(f"reaped {reaped=} zombie processes")


def detect_cgroup_kill_supported() -> tuple[bool, str | None]:
    """Probe whether the bee can use `cgroup.kill` in this environment.

    Three things have to be true: (1) the unified cgroup v2 hierarchy is
    mounted at `_CGROUP_MOUNT` (signaled by `cgroup.controllers` at the mount
    root), (2) the bee can `mkdir` a child cgroup (writable delegation),
    (3) the kernel auto-populates `cgroup.kill` inside new cgroups (Linux 5.14+).

    Probes by creating a uniquely-named temp child cgroup under the mount root,
    checking for the kernel-managed `cgroup.kill` file, then rmdir'ing.
    Distinguishes the read-only-cgroupfs case (common in plain Docker / default
    K8s pods) via `os.statvfs(..).f_flag & ST_RDONLY` so the caller can emit a
    deployment-specific fix message.

    Returns:
        `(True, None)` if supported; `(False, reason)` otherwise, where reason
        is one of the `DETECT_REASON_*` constants.
    """
    log.debug(f"probing {_CGROUP_MOUNT=}")

    controllers = _CGROUP_MOUNT / "cgroup.controllers"
    if not controllers.exists():
        log.warning(f"{controllers=} missing — not a cgroup v2 mount")
        return False, DETECT_REASON_NOT_CGROUP_V2

    # Cheaper + more specific than waiting for mkdir to fail with EROFS.
    try:
        if statvfs(_CGROUP_MOUNT).f_flag & ST_RDONLY:
            log.warning(
                f"{_CGROUP_MOUNT} mounted read-only — "
                f"Docker needs --security-opt writable-cgroups=true (28.0+) or "
                f"--privileged; K8s pods need securityContext.privileged or "
                f"capabilities.add: [SYS_ADMIN] + hostPath /sys/fs/cgroup",
            )
            return False, DETECT_REASON_RO_CGROUPFS
    except OSError as exc:
        log.warning(f"statvfs({_CGROUP_MOUNT}) failed: {exc!r}")

    probe = _CGROUP_MOUNT / f"bbot-detect-{uuid4().hex[:8]}"
    try:
        try:
            probe.mkdir()
        except OSError as exc:
            log.warning(f"cannot mkdir {probe=}: {exc!r}")
            return False, DETECT_REASON_NO_DELEGATION

        if not (probe / "cgroup.kill").exists():
            log.warning(f"{probe / 'cgroup.kill'} missing — kernel <5.14")
            return False, DETECT_REASON_KERNEL_TOO_OLD

        log.info(f"cgroup v2 cgroup.kill available at {_CGROUP_MOUNT}")
        return True, None
    finally:
        with suppress(OSError):
            probe.rmdir()


def recover_orphan_cgroups() -> int:
    """Reclaim any `scan_*` cgroups left under `_PARENT` by a crashed bee.

    For each leftover directory: write `1` to `cgroup.kill` (atomically SIGKILL
    any processes still inside), wait briefly, then rmdir. Only touches
    directories prefixed `scan_` — defensive against unrelated siblings
    appearing under the parent.

    Returns:
        Number of orphan cgroups recovered. 0 if the parent directory doesn't
        exist (clean first run).
    """
    log.debug(f"scanning {_PARENT=}")
    if not _PARENT.exists():
        log.debug("parent absent, nothing to recover")
        return 0

    recovered = 0
    for child in _PARENT.iterdir():
        if not child.is_dir() or not child.name.startswith("scan_"):
            log.debug(f"skipping {child.name=} (not a scan cgroup)")
            continue

        log.warning(f"recovering orphan {child}")
        kill_file = child / "cgroup.kill"
        if kill_file.exists():
            with suppress(OSError):
                kill_file.write_text("1")
        _wait_cgroup_drained(child, timeout_s=2.0)
        try:
            child.rmdir()
            recovered += 1
        except OSError as exc:
            log.warning(f"rmdir {child=} failed: {exc!r} — leaking")

    if recovered:
        log.warning(f"recovered {recovered=} orphan cgroups")
    return recovered


def _wait_cgroup_drained(cgroup_dir: Path, *, timeout_s: float) -> bool:
    """Poll `cgroup.events` `populated` until 0 or timeout, reaping zombies each tick.

    `cgroup.events` exposes a kernel-maintained `populated` flag — the canonical
    "is this cgroup empty" signal (what containerd's `isCgroupEmpty` and runc
    both use). Falls back to reading `cgroup.procs` if `cgroup.events` is
    missing (defensive — bee requires 5.14+ which always has it). Zombies
    still count as populated until reaped, so reap on every tick.
    """
    events = cgroup_dir / "cgroup.events"
    procs = cgroup_dir / "cgroup.procs"
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        reap_zombies()
        try:
            for line in events.read_text().splitlines():
                key, _, value = line.partition(" ")
                if key == "populated":
                    if value.strip() == "0":
                        return True
                    break
            else:
                # No 'populated' key found — fall through to procs fallback.
                try:
                    if not procs.read_text().strip():
                        return True
                except FileNotFoundError:
                    return True
        except FileNotFoundError:
            return True
        sleep(_WAIT_EMPTY_POLL_INTERVAL_S)
    return False


class ScanCgroup:
    """Owns the cgroup for a single scan.

    Lifecycle (called from `Drone`):

    1. `create()` — mkdir the cgroup directory.
    2. `populate(pid)` — move the scan_process PID into the cgroup
       immediately after subprocess spawn, before any forking.
    3. `kill()` — write `1` to `cgroup.kill`. Kernel atomically delivers
       SIGKILL to every process in the cgroup, including descendants of
       descendants. Idempotent.
    4. `wait_empty()` — block until the cgroup drains (reaping zombies
       on each poll).
    5. `cleanup()` — rmdir. Tolerant of EBUSY (leak) and ENOENT (already
       gone).

    All methods are safe to call from synchronous contexts including
    `loop.call_later` callbacks — there is no async work.
    """

    def __init__(self, scan_id: str) -> None:
        """Construct a ScanCgroup for `scan_id`.

        Defensive against path traversal even though the hive's `scan_id`
        is currently a UUID.

        Args:
            scan_id: The scan identifier — must match `[A-Za-z0-9_-]{1,64}`.

        Raises:
            ValueError: If `scan_id` contains anything outside the allowlist.
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

        `scan_process.main()` reads `BBOT_BEE_SCAN_CGROUP` and writes its own
        PID to `<path>/cgroup.procs` before any other startup, closing the
        spawn->populate race window.
        """
        return {"BBOT_BEE_SCAN_CGROUP": str(self._path)}

    def create(self) -> None:
        """`mkdir` the cgroup directory (and parent on first call).

        Raises:
            FileExistsError: If the cgroup directory already exists. `scan_id`
                reuse is a bug, not something to silently paper over.
        """
        log.info(f"ScanCgroup.create: {self._scan_id=}, path={self._path}")
        _PARENT.mkdir(parents=True, exist_ok=True)
        self._path.mkdir(parents=False, exist_ok=False)

    def populate(self, pid: int) -> None:
        """Move `pid` into this cgroup by writing it to `cgroup.procs`.

        Must be called between subprocess spawn and any other work the
        subprocess may do (e.g. forking helpers). Idempotent with the child's
        own self-enroll on `BBOT_BEE_SCAN_CGROUP`.
        """
        log.info(f"ScanCgroup.populate: {self._scan_id=}, {pid=}")
        (self._path / "cgroup.procs").write_text(f"{pid}\n")

    def kill(self) -> int:
        """Write `1` to `cgroup.kill` — atomic kernel SIGKILL to all processes in the cgroup.

        Single `write(2)` syscall on the user side — non-blocking, safe inside
        synchronous event-loop callbacks. Idempotent on the kill side; safe to
        call any number of times.

        Returns:
            Count of processes in `cgroup.procs` at call time (read just before
            the kill, for logging). 0 if the cgroup is already gone.
        """
        kill_file = self._path / "cgroup.kill"
        if not kill_file.exists():
            log.debug(f"ScanCgroup.kill: {self._scan_id=}, cgroup already gone — no-op")
            return 0

        try:
            count = len([line for line in (self._path / "cgroup.procs").read_text().splitlines() if line.strip()])
        except OSError:
            count = 0

        log.warning(f"ScanCgroup.kill: {self._scan_id=}, atomic SIGKILL to {count=} procs in cgroup")
        try:
            kill_file.write_text("1")
        except OSError as exc:
            log.warning(f"ScanCgroup.kill: {self._scan_id=}, write failed: {exc!r}")
            return 0
        return count

    def wait_empty(self, timeout_s: float = _WAIT_EMPTY_TIMEOUT_S) -> bool:
        """Block until the cgroup drains (or timeout).

        Reaps zombies on each poll iteration via `reap_zombies()` — unreaped
        zombies stay accounted in the cgroup, so without this the cgroup never
        empties for descendants of the bee itself.

        Args:
            timeout_s: Maximum seconds to wait.

        Returns:
            True if the cgroup drained, False on timeout. On timeout the caller
            should leak the cgroup directory and let the next
            `recover_orphan_cgroups()` reclaim it.
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
        """`rmdir` the cgroup directory.

        Tolerant of ENOENT (silent no-op) and EBUSY (WARNING + leak,
        recovered on next bee startup). Never raises — callers in cleanup
        paths shouldn't have to guard.
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
