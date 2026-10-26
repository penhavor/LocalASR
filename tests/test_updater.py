# -*- coding: utf-8 -*-
"""Self-update system tests (app/core/updater.py) — NO network access.

  * version parsing/compare ("v1.10.0" > "v1.9.9", junk -> never newer);
  * the local git blob SHA1 matches git's formula byte-for-byte;
  * build_plan: unchanged files are NOT re-downloaded, changed/missing are,
    files removed upstream are deleted — but ONLY files that belonged to the
    old repository tree; models/, settings/, outputs/, logs/ and any other
    user data are never touched; path traversal is rejected;
  * check_for_update: offline is silent-safe (ok=False, offline=True),
    up-to-date reports nothing to do, a newer release builds a real plan,
    API errors never raise;
  * apply_update: downloads (stubbed), deletes, prunes empty folders,
    reports failures per file and stops on cancel.

Run:  cd <project root> && python tests/test_updater.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("LOCALASR_QUIET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.core import net, updater  # noqa: E402
from app.core.updater import RemoteFile  # noqa: E402

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def write(root: Path, rel: str, data: bytes) -> RemoteFile:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return RemoteFile(rel.replace("\\", "/"), blob_sha(data), len(data))


# --------------------------------------------------------------------------- #


def test_versions() -> None:
    print("== version parsing and comparison ==")
    check(updater.parse_version("1.2.3") == (1, 2, 3), "parse 1.2.3")
    check(updater.parse_version("v1.10.0") == (1, 10, 0), "parse v1.10.0 (leading v)")
    check(updater.parse_version("v2.0.0-rc1") == (2, 0, 0), "parse strips suffixes")
    check(updater.parse_version("banana") == (), "unparseable -> ()")
    check(updater.is_newer("v1.10.0", "v1.9.9"), "1.10.0 > 1.9.9 (numeric, not lexicographic)")
    check(not updater.is_newer("v1.0.0", "v1.0.0"), "equal versions are not newer")
    check(not updater.is_newer("v1.0.0", "v1.1.0"), "older is not newer")
    check(not updater.is_newer("banana", "1.0.0"), "junk remote tag -> never newer")
    check(not updater.is_newer("v1.1.0", "banana"), "junk local version -> never newer")
    check(updater.current_tag() == f"v{updater.APP_VERSION}", "current_tag matches APP_VERSION")


def test_blob_sha() -> None:
    print("== local blob sha (git-compatible) ==")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "x.txt"
        path.write_bytes(b"hello\n")
        # the value git hash-object would print for the same content
        check(updater.local_blob_sha(path) == blob_sha(b"hello\n"), "sha matches the git formula")
        check(updater.local_blob_sha(path) == "ce013625030ba8dba906f756967f9e9ca394464a",
              "sha of 'hello\\n' is the known git value")
        check(updater.local_blob_sha(Path(td) / "missing.bin") is None, "missing file -> None")


def test_build_plan() -> None:
    print("== diff plan: download / delete / untouched ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        same = write(root, "app/same.py", b"SAME")            # in both trees, identical
        changed = write(root, "app/changed.py", b"OLD")       # in both trees, content differs
        gone = write(root, "tools/gone.py", b"GONE")          # old tree only -> delete
        gone_missing = root / "tools" / "already_gone.py"     # old tree only, absent locally
        write(root, "models/big.bin", b"USER MODEL")          # user data
        write(root, "settings/settings.json", b"{}")          # user data
        write(root, "outputs/result.txt", b"text")            # user data
        (root / "logs").mkdir(exist_ok=True)

        new_changed_sha = blob_sha(b"NEW")
        new_tree = {
            "app/same.py": same,
            "app/changed.py": RemoteFile("app/changed.py", new_changed_sha, 3),
            "app/added.py": RemoteFile("app/added.py", blob_sha(b"ADDED"), 5),
            "CHANGELOG.md": RemoteFile("CHANGELOG.md", blob_sha(b"# log"), 5),
        }
        old_tree = {
            "app/same.py": same,
            "app/changed.py": RemoteFile("app/changed.py", blob_sha(b"OLD"), 3),
            "tools/gone.py": RemoteFile("tools/gone.py", gone.sha, 4),
            "tools/already_gone.py": RemoteFile("tools/already_gone.py", blob_sha(b"X"), 1),
            # paranoia: even if a protected path somehow appears in a tree,
            # it must never be scheduled for deletion
            "models/keep.bin": RemoteFile("models/keep.bin", blob_sha(b"USER MODEL"), 10),
        }

        download, delete, unchanged, total, added = updater.build_plan(root, new_tree, old_tree)
        download_paths = [item.path for item in download]
        check(download_paths == ["CHANGELOG.md", "app/added.py", "app/changed.py"],
              f"download = missing + changed, sorted: {download_paths}")
        check("app/same.py" not in download_paths and unchanged == 1,
              "identical file is NOT re-downloaded (unchanged=1)")
        check(total == 5 + 3 + 5, f"total_bytes sums the download sizes: {total}")
        check(delete == ["tools/gone.py"],
              f"delete = removed upstream AND present locally: {delete}")
        check(sorted(added) == ["CHANGELOG.md", "app/added.py"],
              f"added = files that do not exist locally at all: {added}")

        # without the old tree nothing may be deleted (we cannot know better)
        download2, delete2, _, _, added2 = updater.build_plan(root, new_tree, None)
        check(delete2 == [] and len(download2) == len(download) and added2 == added,
              "no old tree -> no deletions, same downloads/added")

        # hostile/malformed remote paths are rejected
        evil = {
            "../escape.py": RemoteFile("../escape.py", blob_sha(b"E"), 1),
            "/abs.py": RemoteFile("/abs.py", blob_sha(b"E"), 1),
            "a/../../b.py": RemoteFile("a/../../b.py", blob_sha(b"E"), 1),
            "models/implant.bin": RemoteFile("models/implant.bin", blob_sha(b"E"), 1),
        }
        download3, delete3, _, _, added3 = updater.build_plan(root, evil, evil)
        check(download3 == [] and delete3 == [] and added3 == [],
              "path traversal / absolute / protected paths are skipped")

        check(updater.is_protected("models/x") and updater.is_protected("settings/x")
              and updater.is_protected("outputs/x") and updater.is_protected("logs/x")
              and updater.is_protected("runtime/x") and updater.is_protected(".venv/x")
              and updater.is_protected(".git/x"),
              "user-data folders are protected")
        check(not updater.is_protected("app/x.py") and not updater.is_protected("tools/x.py"),
              "program folders are not protected")
        check(gone_missing.name == "already_gone.py" and not gone_missing.exists(),
              "sanity: the locally absent old file stayed absent")


def _stub_get_json(release_payload, trees):
    """Replace updater._get_json with an offline dispatcher; returns restore()."""
    original = updater._get_json

    def fake(url, params=None):
        if url.endswith("/releases/latest"):
            if release_payload is None:
                raise updater.UpdateError("HTTP 404")
            return release_payload
        if "/git/trees/" in url:
            tag = url.rsplit("/git/trees/", 1)[1]
            if tag in trees:
                return trees[tag]
            raise updater.UpdateError(f"no tree for {tag}")
        if url.endswith("/tags"):
            return [{"name": tag} for tag in trees]
        raise AssertionError(f"unexpected URL: {url}")

    updater._get_json = fake
    return lambda: setattr(updater, "_get_json", original)


def _tree_payload(files: list[RemoteFile]) -> dict:
    return {
        "truncated": False,
        "tree": [{"type": "blob", "path": f.path, "sha": f.sha, "size": f.size} for f in files],
    }


def test_check_for_update() -> None:
    print("== check_for_update (stubbed network) ==")

    # --- offline: silent-safe -------------------------------------------------
    original_get = updater._get_json

    def offline(url, params=None):
        raise updater.OfflineError("no internet")

    updater._get_json = offline
    try:
        info = updater.check_for_update(manual=False)
    finally:
        updater._get_json = original_get
    check(not info.ok and info.offline and not info.update_available,
          "offline -> ok=False, offline=True, no update")
    check(info.error == "no internet", "offline error text is passed through")

    # --- up to date ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        same = write(root, "app/a.py", b"A")
        tag_now = updater.current_tag()
        restore = _stub_get_json(
            {"tag_name": tag_now, "html_url": "u", "body": "notes"},
            {tag_now: _tree_payload([same])},
        )
        try:
            info = updater.check_for_update(manual=True, root=root)
        finally:
            restore()
        check(info.ok and not info.update_available and info.manual,
              "same version -> ok, nothing to update")
        check(info.latest_version == tag_now.lstrip("v"),
              f"latest_version reported: {info.latest_version}")

    # --- update available ------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        same = write(root, "app/a.py", b"A")
        old_b = write(root, "app/b.py", b"OLD-B")
        gone = write(root, "tools/g.py", b"GONE")
        write(root, "models/model.bin", b"USER")          # must stay untouched
        write(root, "settings/settings.json", b"{}")      # must stay untouched

        new_b_sha = blob_sha(b"NEW-B")
        added = RemoteFile("app/new.py", blob_sha(b"NEW FILE"), 8)
        new_tree = {
            "app/a.py": same,
            "app/b.py": RemoteFile("app/b.py", new_b_sha, 6),
            "app/new.py": added,
        }
        old_tree = {
            "app/a.py": same,
            "app/b.py": RemoteFile("app/b.py", old_b.sha, 5),
            "tools/g.py": RemoteFile("tools/g.py", gone.sha, 4),
        }
        tag_new = "v999.1.0"
        restore = _stub_get_json(
            {"tag_name": tag_new, "html_url": "https://x/release", "body": "## Changes\n- stuff"},
            {
                updater.current_tag(): _tree_payload(list(old_tree.values())),
                tag_new: _tree_payload(list(new_tree.values())),
            },
        )
        try:
            info = updater.check_for_update(manual=False, root=root)
        finally:
            restore()
        check(info.ok and info.update_available, "newer release -> update_available")
        check(info.tag == tag_new and info.latest_version == "999.1.0",
              f"tag/latest_version: {info.tag} / {info.latest_version}")
        check([f.path for f in info.download] == ["app/b.py", "app/new.py"],
              "plan downloads only changed+added files")
        check(info.delete == ["tools/g.py"], "plan deletes only the removed program file")
        check(info.unchanged == 1, "identical file counted as unchanged")
        check(info.total_bytes == 6 + 8, f"total_bytes = {info.total_bytes}")
        check(info.file_count == 3, "file_count = downloads + deletions")
        check(info.release_notes.startswith("## Changes"), "release notes carried along")

    # --- repair (force) mode ---------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        same = write(root, "app/a.py", b"A")
        tag_now = updater.current_tag()
        restore = _stub_get_json(
            {"tag_name": tag_now, "html_url": "u", "body": ""},
            {tag_now: _tree_payload([same])},
        )
        try:
            info = updater.check_for_update(manual=True, force=True, root=root)
            check(info.ok and not info.update_available and info.repair,
                  "force on a healthy install: nothing to restore")
            (root / "app" / "a.py").write_bytes(b"DAMAGED")
            info = updater.check_for_update(manual=True, force=True, root=root)
            check(info.ok and info.update_available and info.repair
                  and [f.path for f in info.download] == ["app/a.py"],
                  "force finds the damaged file even though the version is current")
        finally:
            restore()

    # --- API error: never raises ------------------------------------------------
    def broken(url, params=None):
        raise updater.UpdateError("GitHub returned HTTP 500")

    updater._get_json = broken
    try:
        info = updater.check_for_update(manual=True)
    finally:
        updater._get_json = original_get
    check(not info.ok and not info.offline and "500" in info.error,
          "API error -> ok=False, offline=False, error text kept")

    # --- unexpected exception: still no raise ------------------------------------
    def boom(url, params=None):
        raise ValueError("surprise")

    updater._get_json = boom
    try:
        info = updater.check_for_update(manual=True)
    finally:
        updater._get_json = original_get
    check(not info.ok and "surprise" in info.error, "unexpected exceptions are contained")


def test_apply_update() -> None:
    print("== apply_update (stubbed downloads) ==")
    payloads = {
        "app/b.py": b"NEW-B",
        "app/new.py": b"NEW FILE",
        "app/broken.py": b"",       # the fake download will "fail" on this one
    }
    original_download = net.download_file

    def fake_download(url, dest, progress=None, cancel=None, **kwargs):
        # derive the repo path from .../{tag}/{path...}
        marker = "/v999.1.0/"
        path = url.split(marker, 1)[1] if marker in url else url.rsplit("/", 1)[-1]
        data = payloads.get(path)
        if data is None or path == "app/broken.py":
            raise net.NetworkError(f"HTTP 404 for {path}")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(data)
        if progress is not None:
            progress(len(data), len(data), 0.0, Path(dest).name)
        return Path(dest)

    net.download_file = fake_download
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write(root, "app/b.py", b"OLD-B")
            gone_dir = root / "tools" / "sub"
            gone_dir.mkdir(parents=True)
            (gone_dir / "g.py").write_bytes(b"GONE")

            info = updater.UpdateInfo(
                ok=True, update_available=True, tag="v999.1.0", latest_version="999.1.0",
                download=[
                    RemoteFile("app/b.py", blob_sha(b"NEW-B"), 5),
                    RemoteFile("app/new.py", blob_sha(b"NEW FILE"), 8),
                    RemoteFile("app/broken.py", blob_sha(b"X"), 1),
                ],
                delete=["tools/sub/g.py"],
            )
            seen: list[tuple[float, str]] = []
            updated, deleted, failed, cancelled = updater.apply_update(
                info, root=root,
                progress=lambda f, s: seen.append((f, s)),
                cancel=lambda: False,
            )
            fractions = [f for f, _ in seen]
            check(updated == ["app/b.py", "app/new.py"], f"updated: {updated}")
            check((root / "app" / "b.py").read_bytes() == b"NEW-B", "changed file has the new content")
            check((root / "app" / "new.py").is_file(), "new file appeared")
            check(failed and failed[0].startswith("app/broken.py"),
                  f"the failing file is reported, not fatal: {failed}")
            check(cancelled is False, "not cancelled")
            check(seen and seen[-1][0] == 1.0, "progress reached 1.0")
            check(all(0.0 <= f <= 1.0 for f in fractions)
                  and fractions == sorted(fractions),
                  f"progress is monotonic within 0..1: {fractions[:6]}…")
            # a failing download must not block the rest of the plan:
            # the deletion pass still runs
            check(deleted == ["tools/sub/g.py"], f"deletions still applied: {deleted}")

        # --- successful run performs deletions and prunes empty folders ----------
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write(root, "app/b.py", b"OLD-B")
            gone = root / "tools" / "sub" / "g.py"
            gone.parent.mkdir(parents=True)
            gone.write_bytes(b"GONE")
            keep = write(root, "tools/keep.py", b"KEEP")

            info = updater.UpdateInfo(
                ok=True, tag="v999.1.0", latest_version="999.1.0",
                download=[RemoteFile("app/b.py", blob_sha(b"NEW-B"), 5)],
                delete=["tools/sub/g.py"],
            )
            updated, deleted, failed, cancelled = updater.apply_update(info, root=root)
            check(updated == ["app/b.py"] and not failed and not cancelled, "clean update")
            check(deleted == ["tools/sub/g.py"], "removed-upstream file deleted")
            check(not gone.exists() and not (root / "tools" / "sub").exists(),
                  "the folder that became empty was pruned")
            check((root / "tools" / "keep.py").is_file(),
                  "a sibling file keeps its (non-empty) folder alive")

        # --- cancellation stops the queue ---------------------------------------
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = {"n": 0}

            def cancel_after_first():
                state["n"] += 1
                # the token is consulted once per queued step: the first
                # download passes, the second step is cancelled
                return state["n"] > 1

            info = updater.UpdateInfo(
                ok=True, tag="v999.1.0", latest_version="999.1.0",
                download=[
                    RemoteFile("app/b.py", blob_sha(b"NEW-B"), 5),
                    RemoteFile("app/new.py", blob_sha(b"NEW FILE"), 8),
                ],
                delete=["tools/sub/g.py"],
            )
            updated, deleted, failed, cancelled = updater.apply_update(
                info, root=root, cancel=cancel_after_first)
            check(cancelled and updated == ["app/b.py"] and deleted == [],
                  f"cancel mid-way: updated={updated}, deleted={deleted}")
    finally:
        net.download_file = original_download


def test_verify_and_classify() -> None:
    print("== post-update verification and impact classification ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = write(root, "app/a.py", b"A")
        b = write(root, "app/b.py", b"B")
        gone = root / "tools" / "old.py"
        gone.parent.mkdir(parents=True)
        gone.write_bytes(b"OLD")

        info = updater.UpdateInfo(
            ok=True, update_available=True, tag="v2.0.0", latest_version="2.0.0",
            current_version="1.1.0",
            tree={"app/a.py": a.sha, "app/b.py": b.sha},
            delete=["tools/old.py"],
        )
        # verification runs AFTER apply: the removed-upstream file is gone by then
        gone.unlink()
        bad, not_deleted = updater.verify_update(info, root=root)
        check(bad == [] and not_deleted == [], f"clean installation verifies: {bad} {not_deleted}")

        (root / "app" / "b.py").write_bytes(b"CORRUPTED")
        bad, not_deleted = updater.verify_update(info, root=root)
        check(bad == ["app/b.py"] and not_deleted == [], f"corrupted file detected: {bad}")
        (root / "app" / "b.py").write_bytes(b"B")
        (root / "app" / "a.py").unlink()
        bad, not_deleted = updater.verify_update(info, root=root)
        check(bad == ["app/a.py"], f"missing file detected: {bad}")
        write(root, "app/a.py", b"A")
        gone.parent.mkdir(parents=True, exist_ok=True)
        gone.write_bytes(b"OLD")
        check(updater.verify_update(info, root=root) == ([], ["tools/old.py"]),
              "a file that should be gone is reported")

    # --- classification -------------------------------------------------------
    def mk(latest, tree_n, touched, deps=False):
        info = updater.UpdateInfo(
            ok=True, update_available=True, tag=f"v{latest}", latest_version=latest,
            current_version="1.1.0",
            tree={f"f{i}.py": "x" for i in range(tree_n)},
            download=[RemoteFile(f"f{i}.py", "y", 1) for i in range(touched)],
            deps_changed=deps,
        )
        return info

    check(updater.classify_update(mk("2.0.0", 71, 3)) == "major",
          "a new major version is always a major update")
    check(updater.classify_update(mk("1.1.1", 71, 2)) == "minor",
          "a couple of files in a patch release -> minor")
    check(updater.classify_update(mk("1.2.0", 71, 2, deps=True)) == "major",
          "dependency change upgrades the update to major")
    check(updater.classify_update(mk("1.2.0", 70, 40)) == "major",
          "half of the program replaced -> major")
    check(updater.classify_update(mk("1.2.0", 70, 20)) == "minor",
          "a third of the files -> minor")


def test_raw_url() -> None:
    print("== raw download urls ==")
    url = updater.raw_url("v1.2.3", "Create Shortcut.bat")
    check(url == "https://raw.githubusercontent.com/penhavor/LocalASR/v1.2.3/Create%20Shortcut.bat",
          f"spaces are quoted: {url}")
    url2 = updater.raw_url("v1.2.3", "app/core/updater.py")
    check(url2.endswith("/v1.2.3/app/core/updater.py"), f"plain path: {url2}")


def main() -> int:
    test_versions()
    test_blob_sha()
    test_build_plan()
    test_check_for_update()
    test_apply_update()
    test_verify_and_classify()
    test_raw_url()
    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("UPDATER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
