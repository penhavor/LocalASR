"""Self-update: GitHub releases check + differential file sync.

How it works
------------
1. Connectivity: a short request to the GitHub API. Offline -> the check
   ends silently (auto mode) or with an explicit message (manual mode).
2. Version: the latest release tag (`releases/latest`, fallback: `tags`)
   is compared with APP_VERSION. Same or older -> nothing to do.
3. Diff: the git tree of the new tag (`git/trees/<tag>?recursive=1`) holds
   the blob SHA of every repository file. The local SHA is computed with
   the same formula git uses — sha1("blob <len>\\0" + bytes) — so exactly
   the missing/changed files are downloaded from raw.githubusercontent.com.
4. Removals: files that existed in the CURRENT version's tag tree but are
   gone from the new tree are deleted locally. Nothing else is ever
   touched: models/, outputs/, logs/, settings/, runtime/, .venv/ and any
   other file outside the repository tree stay exactly as they are.

The module imports no Qt: it is used by GUI workers (ui/workers.py) and is
fully unit-testable (tests/test_updater.py) with stubbed network calls.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from .. import paths
from ..constants import (
    APP_VERSION,
    GITHUB_API_BASE,
    GITHUB_OWNER,
    GITHUB_REPO,
    REPO_URL,
    UPDATE_HTTP_TIMEOUT,
)
from ..i18n import _
from ..logutils import get_logger

log = get_logger("updater")

ProgressCallback = Callable[[float, str], None]   # fraction, status text
CancelToken = Callable[[], bool]

#: Top-level folders that must NEVER be deleted or modified by the updater:
#: user data, downloaded models, results, logs and Python environments.
PROTECTED_DIRS = frozenset({
    "models", "outputs", "logs", "settings", "runtime",
    ".venv", ".git", "__pycache__",
})

#: GitHub raw content base (release downloads are pinned to the tag).
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}"

#: REST endpoints of the project repository.
API_REPO = f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPO}"


class UpdateError(RuntimeError):
    """A failed check/apply with a human-readable (translated) message."""


class OfflineError(UpdateError):
    """GitHub could not be reached at all (no internet, DNS, proxy, block)."""


# --------------------------------------------------------------------------- #
#  Result containers
# --------------------------------------------------------------------------- #


@dataclass
class RemoteFile:
    path: str      # repository-relative, POSIX separators
    sha: str       # git blob sha1
    size: int = 0


@dataclass
class UpdateInfo:
    """Outcome of one update check (never raises — see `ok`/`error`)."""

    manual: bool = False             # True: started by the user (report everything)
    ok: bool = False                 # the check itself completed
    offline: bool = False            # GitHub unreachable (no internet etc.)
    update_available: bool = False
    current_version: str = APP_VERSION
    latest_version: str = ""
    tag: str = ""                    # remote tag to install, e.g. "v1.2.0"
    release_url: str = ""
    release_notes: str = ""
    download: List[RemoteFile] = field(default_factory=list)   # changed/new
    delete: List[str] = field(default_factory=list)            # removed upstream
    added: List[str] = field(default_factory=list)             # paths NEW in this release
    tree: Dict[str, str] = field(default_factory=dict)         # path -> blob sha (new release)
    deps_changed: bool = False        # requirements/base.txt touched -> env re-install
    repair: bool = False             # forced re-check of an already-"current" version
    unchanged: int = 0               # files already identical locally
    total_bytes: int = 0             # download volume
    error: str = ""

    @property
    def file_count(self) -> int:
        return len(self.download) + len(self.delete)


# --------------------------------------------------------------------------- #
#  Versions
# --------------------------------------------------------------------------- #

_VERSION_RE = re.compile(r"(\d+(?:\.\d+)*)")


def parse_version(text: str) -> Tuple[int, ...]:
    """"v1.10.2-beta" -> (1, 10, 2). Unparseable -> () (never 'newer')."""
    match = _VERSION_RE.search(str(text or ""))
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def is_newer(candidate: str, current: str) -> bool:
    """Tuple compare: 1.10.0 > 1.9.9; equal or unparseable -> False."""
    new, old = parse_version(candidate), parse_version(current)
    if not new or not old:
        return False
    return new > old


def current_tag() -> str:
    return f"v{APP_VERSION}"


# --------------------------------------------------------------------------- #
#  GitHub API (tiny, tolerant)
# --------------------------------------------------------------------------- #


def _session():
    from . import net  # local import: requests lives in the project env

    return net.session()


def _classify_request_error(exc: Exception) -> UpdateError:
    """requests exception -> Offline- or UpdateError with a readable text."""
    name = exc.__class__.__name__
    text = str(exc)
    import requests  # noqa: PLC0415 - already imported by net.session()

    if isinstance(exc, (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout)):
        return OfflineError(_(
            "No access to GitHub: the connection failed ({err}).\n"
            "Check the internet connection, proxy/antivirus or try again later."
        ).format(err=f"{name}: {text}"[:200]))
    return UpdateError(_(
        "Failed to contact GitHub: {err}"
    ).format(err=f"{name}: {text}"[:200]))


def _get_json(url: str, params: Optional[Dict] = None) -> object:
    """GET url -> parsed JSON. Network problems become UpdateError/OfflineError."""
    sess = _session()
    try:
        resp = sess.get(url, params=params or {}, timeout=UPDATE_HTTP_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - requests raises a wide family
        raise _classify_request_error(exc) from exc
    if resp.status_code == 403 and "rate limit" in (resp.text or "").lower():
        raise UpdateError(_(
            "GitHub API request limit exceeded — try again in an hour."
        ))
    if resp.status_code == 404:
        raise UpdateError(_("Not found on GitHub: {url}").format(url=url))
    if resp.status_code != 200:
        raise UpdateError(_(
            "GitHub returned HTTP {code} for {url}"
        ).format(code=resp.status_code, url=url))
    try:
        return resp.json()
    except ValueError as exc:
        raise UpdateError(_("GitHub returned a non-JSON response: {exc}").format(exc=exc)) from exc


def fetch_latest_release() -> Dict[str, str]:
    """{tag, url, notes}: the newest GitHub release; falls back to tags.

    `releases/latest` already skips drafts and prereleases. Without releases
    the newest tag wins (compared as a version tuple, not alphabetically).
    """
    try:
        payload = _get_json(f"{API_REPO}/releases/latest")
        if isinstance(payload, dict) and payload.get("tag_name"):
            return {
                "tag": str(payload["tag_name"]),
                "url": str(payload.get("html_url") or ""),
                "notes": str(payload.get("body") or ""),
            }
    except UpdateError as exc:
        if isinstance(exc, OfflineError):
            raise
        log.debug("releases/latest unavailable (%s) — trying tags", exc)
    payload = _get_json(f"{API_REPO}/tags")
    if not isinstance(payload, list) or not payload:
        raise UpdateError(_("The repository has no tags/releases yet."))
    names = [str(item.get("name") or "") for item in payload if isinstance(item, dict)]
    names = [name for name in names if parse_version(name)]
    if not names:
        raise UpdateError(_("Unexpected GitHub API response (no version tags)."))
    tag = max(names, key=parse_version)
    return {
        "tag": tag,
        "url": f"{REPO_URL}/releases/tag/{quote(tag)}",
        "notes": "",
    }


def fetch_tree(tag: str) -> Dict[str, RemoteFile]:
    """All files of a tag: path -> RemoteFile(sha, size)."""
    payload = _get_json(f"{API_REPO}/git/trees/{quote(tag)}", {"recursive": "1"})
    if not isinstance(payload, dict):
        raise UpdateError(_("Unexpected GitHub API response (tree)."))
    if payload.get("truncated"):
        raise UpdateError(_(
            "The repository tree is too large for one API response — "
            "a differential update is impossible."
        ))
    tree: Dict[str, RemoteFile] = {}
    for item in payload.get("tree") or []:
        if not isinstance(item, dict) or item.get("type") != "blob":
            continue
        path = str(item.get("path") or "")
        sha = str(item.get("sha") or "")
        if not path or not sha or path.startswith("/") or ".." in path.split("/"):
            continue
        tree[path] = RemoteFile(path, sha, int(item.get("size") or 0))
    if not tree:
        raise UpdateError(_("The repository tree is empty — refusing to update."))
    return tree


# --------------------------------------------------------------------------- #
#  Local side: blob hashes and safe paths
# --------------------------------------------------------------------------- #


def local_blob_sha(file_path: Path) -> Optional[str]:
    """git-compatible blob SHA1 of a local file (None when unreadable)."""
    try:
        with open(file_path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git format


def _safe_local(root: Path, rel: str) -> Optional[Path]:
    """rel -> absolute path strictly inside root; None on anything suspicious."""
    rel = (rel or "").replace("\\", "/").strip()
    if not rel or rel.startswith("/") or ":" in rel.split("/")[0]:
        return None
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    try:
        candidate = (root / Path(*parts)).resolve()
        candidate.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def is_protected(rel: str) -> bool:
    """True for user-data folders the updater must never touch."""
    first = str(rel or "").replace("\\", "/").split("/", 1)[0].lower()
    return first in {name.lower() for name in PROTECTED_DIRS}


# --------------------------------------------------------------------------- #
#  Diff / plan
# --------------------------------------------------------------------------- #


#: the dependency fingerprint file: when it changes, the next launch through
#: bootstrap re-installs the environment (see requirements/base.txt header)
DEPS_FILE = "requirements/base.txt"


def build_plan(
    root: Path,
    new_tree: Dict[str, RemoteFile],
    old_tree: Optional[Dict[str, RemoteFile]] = None,
) -> Tuple[List[RemoteFile], List[str], int, int, List[str]]:
    """(download, delete, unchanged_count, total_bytes, added_paths).

    download: files of the new tree that are missing locally or differ by
              blob SHA (byte-exact; an identical file is never re-fetched).
    delete:   files present in the CURRENT version's tree but removed in the
              new one AND still lying on disk. Without `old_tree` nothing is
              deleted (we cannot know what belonged to the old version).
              Protected folders and everything outside the trees is untouched.
    added_paths: subset of `download` that does not exist locally at all —
              genuinely new files of the release (for the impact summary).
    """
    download: List[RemoteFile] = []
    added: List[str] = []
    unchanged = 0
    total_bytes = 0
    for rel, remote in sorted(new_tree.items()):
        if is_protected(rel):        # paranoia: a repo would never have these
            continue
        local = _safe_local(root, rel)
        if local is None:
            log.warning("Skipping a suspicious remote path: %s", rel)
            continue
        if local.is_file() and local_blob_sha(local) == remote.sha:
            unchanged += 1
            continue
        if not local.is_file():
            added.append(rel)
        download.append(remote)
        total_bytes += remote.size or 0

    delete: List[str] = []
    if old_tree:
        removed = sorted(set(old_tree) - set(new_tree))
        for rel in removed:
            if is_protected(rel):
                continue
            local = _safe_local(root, rel)
            if local is not None and local.is_file():
                delete.append(rel)
    return download, delete, unchanged, total_bytes, added


# --------------------------------------------------------------------------- #
#  Full check
# --------------------------------------------------------------------------- #


def check_for_update(
    manual: bool = False,
    root: Optional[Path] = None,
    force: bool = False,
) -> UpdateInfo:
    """One complete check. Never raises: problems land in info.error.

    offline=True  -> GitHub unreachable (silent in auto mode, warning in manual).
    ok=True       -> the check completed; look at update_available.
    force=True    -> REPAIR mode: compare the installation with the tree of
          its OWN version tag even when no newer release exists. Catches the
          aftermath of an interrupted update (version file already new, some
          other file missing/broken). update_available then means "there is
          something to restore", and info.repair is True.
    """
    info = UpdateInfo(manual=manual, current_version=APP_VERSION)
    info.repair = force
    base = Path(root) if root else paths.ROOT_DIR
    try:
        release = fetch_latest_release()
        tag = release["tag"]
        info.tag = tag
        info.release_url = release["url"]
        info.release_notes = (release["notes"] or "").strip()
        info.latest_version = tag.lstrip("vV") if parse_version(tag) else tag
        if not force and not is_newer(tag, APP_VERSION):
            info.ok = True
            log.debug("Up to date: local %s, remote %s", APP_VERSION, tag)
            return info
        if force:
            # repair compares against OUR OWN version tag when the remote
            # release is not newer (the damage belongs to the installed version)
            tag = current_tag() if not is_newer(release["tag"], APP_VERSION) else release["tag"]
            info.tag = tag

        new_tree = fetch_tree(tag)
        old_tree: Optional[Dict[str, RemoteFile]] = None
        try:
            old_tree = fetch_tree(current_tag())
        except UpdateError as exc:
            # no tree for the local version (dev build, tag removed): downloads
            # still work, we just cannot know which files to delete
            log.debug("Old tree (%s) unavailable: %s — deletions skipped",
                      current_tag(), exc)

        download, delete, unchanged, total_bytes, added = build_plan(base, new_tree, old_tree)
        info.ok = True
        info.download = download
        info.delete = delete
        info.added = added
        info.tree = {rel: remote.sha for rel, remote in new_tree.items()}
        touched = {remote.path for remote in download} | set(delete)
        info.deps_changed = DEPS_FILE in touched
        info.unchanged = unchanged
        info.total_bytes = total_bytes
        # in repair mode an empty plan means the installation is healthy
        info.update_available = bool(download or delete) if force else True
        log.info(
            "Update available: %s -> %s (download %d file(s), %s; delete %d; same %d; deps=%s)",
            APP_VERSION, tag, len(download), paths.human_size(total_bytes),
            len(delete), unchanged, info.deps_changed,
        )
        return info
    except OfflineError as exc:
        info.ok = False
        info.offline = True
        info.error = str(exc)
        log.info("Update check: offline (%s)", exc)
        return info
    except UpdateError as exc:
        info.ok = False
        info.error = str(exc)
        log.warning("Update check failed: %s", exc)
        return info
    except Exception as exc:  # noqa: BLE001 - an update check must never crash
        info.ok = False
        info.error = _("Unexpected error while checking for updates: {exc}").format(
            exc=f"{type(exc).__name__}: {exc}"
        )
        log.warning("Update check crashed: %s", exc, exc_info=True)
        return info


# --------------------------------------------------------------------------- #
#  Impact assessment and post-update verification
# --------------------------------------------------------------------------- #


def classify_update(info: UpdateInfo) -> str:
    """"major" | "minor": how much the update shakes the installation.

    major — a new MAJOR version (2.0.0 after 1.x), a dependency change
    (the environment will be re-installed at the next launch), or the release
    replaces half of the program files or more. Everything else is minor.
    Used only for the human-readable warning BEFORE the update.
    """
    new, old = parse_version(info.latest_version or info.tag), parse_version(info.current_version)
    if new and old and new[0] > old[0]:
        return "major"
    if info.deps_changed:
        return "major"
    total = len(info.tree) or 1
    touched = len(info.download) + len(info.delete)
    if touched * 2 >= total:      # >= 50% of the release files
        return "major"
    return "minor"


def verify_update(info: UpdateInfo, root: Optional[Path] = None) -> Tuple[List[str], List[str]]:
    """Byte-exact post-update audit against the release tree (offline).

    Returns (bad_files, not_deleted):
      bad_files   — files of the new release that are missing locally or whose
                    blob SHA differs (interrupted download, disk error…);
      not_deleted — files that should have been removed but are still there.
    Empty lists == the installation matches the release exactly.
    """
    base = Path(root) if root else paths.ROOT_DIR
    bad: List[str] = []
    for rel, sha in sorted(info.tree.items()):
        if is_protected(rel):
            continue
        local = _safe_local(base, rel)
        if local is None or not local.is_file() or local_blob_sha(local) != sha:
            bad.append(rel)
    not_deleted: List[str] = []
    for rel in info.delete:
        local = _safe_local(base, rel)
        if local is not None and local.exists():
            not_deleted.append(rel)
    return bad, not_deleted


# --------------------------------------------------------------------------- #
#  Apply
# --------------------------------------------------------------------------- #


def raw_url(tag: str, rel: str) -> str:
    return f"{RAW_BASE}/{quote(tag)}/{quote(rel)}"


def _prune_empty_dirs(root: Path, file_path: Path) -> None:
    """Remove parent folders that became empty (never above root)."""
    parent = file_path.parent
    try:
        root_resolved = root.resolve()
    except OSError:
        return
    while True:
        try:
            if parent.resolve() == root_resolved:
                return
            if any(parent.iterdir()):
                return
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def apply_update(
    info: UpdateInfo,
    root: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[CancelToken] = None,
) -> Tuple[List[str], List[str], List[str], bool]:
    """Download the changed files and remove the deleted ones.

    Returns (updated, deleted, failed, cancelled). Every file is written
    atomically (.part -> rename), so an interrupted update leaves no broken
    files; repeating it finishes the job (the diff is recomputed anyway).
    """
    from . import net  # local import: requests lives in the project env

    base = Path(root) if root else paths.ROOT_DIR
    tag = info.tag or current_tag()
    updated: List[str] = []
    deleted: List[str] = []
    failed: List[str] = []
    cancelled = False

    steps = len(info.download) + len(info.delete)
    step = 0

    def report(status: str, inner: float = 1.0) -> None:
        if progress is not None and steps:
            fraction = (step + max(0.0, min(1.0, inner))) / steps
            progress(fraction, status)

    for remote in info.download:
        if cancel is not None and cancel():
            cancelled = True
            break
        dest = _safe_local(base, remote.path)
        if dest is None or is_protected(remote.path):
            failed.append("{rel}: {why}".format(rel=remote.path, why=_("unsafe path — skipped")))
            step += 1
            continue
        # a stale .part from an older attempt would resume into a mixed file
        try:
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
        except OSError:
            pass
        report(_("Downloading {rel}…").format(rel=remote.path), 0.0)

        def on_bytes(done: int, total: int, speed: float, name: str, _r=remote) -> None:
            inner = (done / total) if total else 0.0
            report(_("Downloading {rel}…").format(rel=_r.path), inner)

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            net.download_file(
                raw_url(tag, remote.path), dest,
                progress=on_bytes, cancel=cancel, resume=False,
            )
            updated.append(remote.path)
            log.info("Updated file: %s (%d bytes)", remote.path, remote.size)
        except net.DownloadCancelled:
            cancelled = True
            break
        except Exception as exc:  # noqa: BLE001 - collect and keep going
            failed.append(f"{remote.path}: {exc}")
            log.warning("Failed to update %s: %s", remote.path, exc)
        step += 1
        # the file is done: show step/steps (inner=0 — the next one is not started)
        report(_("Downloading {rel}…").format(rel=remote.path), 0.0)

    if not cancelled:
        for rel in info.delete:
            if cancel is not None and cancel():
                cancelled = True
                break
            local = _safe_local(base, rel)
            if local is None or is_protected(rel):
                step += 1
                continue
            report(_("Removing {rel}…").format(rel=rel), 0.5)
            try:
                if local.is_file():
                    local.unlink()
                    deleted.append(rel)
                    _prune_empty_dirs(base, local)
                    log.info("Removed file: %s", rel)
            except OSError as exc:
                failed.append(f"{rel}: {exc}")
                log.warning("Failed to remove %s: %s", rel, exc)
            step += 1

    if progress is not None:
        progress(1.0 if not cancelled else (step / max(1, steps)), _("Done"))
    return updated, deleted, failed, cancelled
