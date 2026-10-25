# -*- coding: utf-8 -*-
"""Model installation integrity tests (local HTTP server, no internet).

  * successful install: installed.json present, the partial marker removed,
    the catalog sees the model as installed, the engine is detected (qwen-asr);
  * a "truncated" file (size differs from the plan) -> InstallError, the
    folder keeps .localasr-partial.json and the catalog does NOT treat the
    model as installed;
  * cancelling an install -> the marker stays, the model is not "installed";
  * reinstalling skips already complete files (resume) and finishes fine;
  * uninstall_model removes the partial folder as well.

Run:  cd <project root> && python tests/test_download_integrity.py
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("LOCALASR_QUIET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import i18n  # noqa: E402

i18n.init("ru")  # stable language for message assertions

from app.constants import ENGINE_QWEN_ASR  # noqa: E402
from app.core import net  # noqa: E402
from app.core.downloader import (  # noqa: E402
    PARTIAL_MARKER,
    InstallError,
    install_model,
    uninstall_model,
)
from app.core.models_catalog import META_FILENAME, ModelCatalog, ModelEntry  # noqa: E402

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


# --------------------------------------------------------------------------- #
#  Local "Hugging Face"
# --------------------------------------------------------------------------- #

FILES: dict[str, bytes] = {
    "config.json": json.dumps(
        {"model_type": "qwen3_asr", "architectures": ["Qwen3ASRForConditionalGeneration"]}
    ).encode(),
    "preprocessor_config.json": json.dumps(
        {"feature_extractor_type": "Qwen3ASRFeatureExtractor", "sampling_rate": 16000}
    ).encode(),
    "tokenizer.json": b'{"model": {"vocab": {}}}' + b"T" * 500,
    "model.safetensors": b"\x00" * 4096,
}

#: files the server deliberately serves TRUNCATED (simulating a network drop)
TRUNCATED: set[str] = set()
#: per-file request counter (for the resume check)
HITS: dict[str, int] = {}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        name = self.path.lstrip("/").split("?")[0]
        HITS[name] = HITS.get(name, 0) + 1
        if name not in FILES:
            self.send_error(404)
            return
        body = FILES[name]
        if name in TRUNCATED:
            body = body[: max(1, len(body) // 2)]
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def log_message(self, *args):  # console silence
        pass


def start_server() -> tuple[http.server.ThreadingHTTPServer, str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def make_entry() -> ModelEntry:
    return ModelEntry(
        id="qwen3-asr-test",
        name="Qwen3-ASR (test)",
        engine=ENGINE_QWEN_ASR,
        source="test/qwen3-asr",
        languages=["ru", "en"],
    )


def patch_net(base_url: str):
    """Replace the HF API with the local server."""

    def fake_hf_tree(repo_id, subfolder="", revision="main", recursive=True):
        return [net.RemoteFile(name, len(data), False) for name, data in FILES.items()]

    def fake_resolve(repo_id, filename, revision="main"):
        return f"{base_url}/{filename}"

    original = (net.hf_tree, net.hf_resolve_url)
    net.hf_tree = fake_hf_tree
    net.hf_resolve_url = fake_resolve
    return original


# --------------------------------------------------------------------------- #


def main() -> int:
    server, base_url = start_server()
    original = patch_net(base_url)
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            print("== successful installation ==")
            entry = make_entry()
            result = install_model(entry, root=root)
            model_dir = root / "models" / "qwen3-asr-test"
            check(model_dir.is_dir(), f"model folder created: {model_dir.name}")
            check((model_dir / META_FILENAME).is_file(), "installed.json written")
            check(not (model_dir / PARTIAL_MARKER).exists(), "partial-install marker removed")
            meta = json.loads((model_dir / META_FILENAME).read_text(encoding="utf-8"))
            check(meta["engine"] == ENGINE_QWEN_ASR, f"engine detected: {meta['engine']}")
            check(sorted(meta.get("files") or []) == sorted(FILES), "file list in metadata")
            check(result.downloaded_bytes == sum(len(v) for v in FILES.values()),
                  "downloaded volume matches the plan")

            catalog = ModelCatalog(root=root)
            installed = catalog.get("qwen3-asr-test")
            check(installed is not None and installed.installed, "catalog sees the model as installed")

            print("== truncated file -> install fails, model is NOT 'installed' ==")
            uninstall_model(installed, root=root)
            check(not model_dir.exists(), "old folder removed")
            TRUNCATED.add("model.safetensors")
            entry2 = make_entry()
            try:
                install_model(entry2, root=root)
                check(False, "a truncated file must raise InstallError")
            except InstallError as exc:
                check("повреждены или недокачаны" in str(exc), f"clear error: {str(exc)[:80]}…")
            check((model_dir / PARTIAL_MARKER).is_file(), "partial marker kept")
            check(not (model_dir / META_FILENAME).exists(), "installed.json NOT created")
            catalog.refresh_installed()
            entry_after = catalog.get("qwen3-asr-test")
            check(
                entry_after is None or not entry_after.installed,
                "the catalog does NOT treat the incomplete model as installed",
            )

            print("== reinstall: resume of complete files ==")
            TRUNCATED.clear()
            for name in list(HITS):
                HITS[name] = 0
            entry3 = make_entry()
            install_model(entry3, root=root)
            check((model_dir / META_FILENAME).is_file(), "installed.json present after resume")
            check(not (model_dir / PARTIAL_MARKER).exists(), "marker removed")
            check(HITS.get("model.safetensors", 0) >= 1, "the truncated file was re-downloaded")
            check(HITS.get("config.json", 0) == 0 and HITS.get("tokenizer.json", 0) == 0,
                  "complete files were NOT re-downloaded (resume works)")

            print("== cancelling an install ==")
            uninstall_model(catalog.get("qwen3-asr-test") or entry3, root=root)
            cancel_flag = {"cancel": False}

            def cancel_after_first():
                # cancel right after the first server hit (with a deadline)
                deadline = time.time() + 10
                while sum(HITS.values()) < 1 and time.time() < deadline:
                    time.sleep(0.005)
                cancel_flag["cancel"] = True

            watcher = threading.Thread(target=cancel_after_first, daemon=True)
            watcher.start()
            entry4 = make_entry()
            try:
                install_model(entry4, root=root, cancel=lambda: cancel_flag["cancel"])
                check(False, "cancellation must interrupt the install")
            except net.DownloadCancelled:
                check(True, "cancellation interrupted the install (DownloadCancelled)")
            check((model_dir / PARTIAL_MARKER).is_file(), "marker kept after cancellation")
            catalog.refresh_installed()
            after_cancel = catalog.get("qwen3-asr-test")
            check(after_cancel is None or not after_cancel.installed,
                  "a cancelled install is not seen as 'installed'")
            check(not any(p.suffix == ".part" for p in model_dir.rglob("*.part")),
                  "incomplete .part files cleaned up")

            print("== uninstall removes the partial folder ==")
            check(uninstall_model(entry4, root=root), "uninstall_model deleted the partial folder")
            check(not model_dir.exists(), "folder is gone")
    finally:
        net.hf_tree, net.hf_resolve_url = original
        server.shutdown()

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("INSTALLATION INTEGRITY TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
