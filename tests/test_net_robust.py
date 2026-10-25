# -*- coding: utf-8 -*-
"""net.download_file robustness tests.

Local HTTP server, no internet needed:
  * plain download: file intact, .part renamed;
  * resume (Range): an existing .part is continued, not re-downloaded;
  * a "silently truncated" response (Content-Length > body) -> retry and an
    error; the broken file is NOT saved as ready;
  * cancellation during a retry pause fires FAST (a thread sleeping in the
    pause must hear the cancel immediately);
  * HTTP 404 -> NetworkError on the first attempt.

Run:  cd <project root> && python tests/test_net_robust.py
"""
from __future__ import annotations

import http.server
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

from app.core import net  # noqa: E402

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


PAYLOAD = bytes(range(256)) * 400          # 102 400 bytes
MODE = {"mode": "normal", "flaky_left": 0}


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802
        name = self.path.lstrip("/").split("?")[0]
        if name == "missing.bin":
            self.send_error(404)
            return
        if name == "drop.bin":
            # connection drop right after the headers: the client instantly
            # gets a ConnectionError and enters the retry pause — exactly
            # the pause cancel must be able to interrupt
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.close_connection = True
            return
        if name == "flaky.bin":
            # the first N requests drop mid-body (Content-Length lies, the
            # connection closes — the client sees IncompleteRead immediately)
            if MODE["flaky_left"] > 0:
                MODE["flaky_left"] -= 1
                body = PAYLOAD[: len(PAYLOAD) // 3]
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
                return
            body = PAYLOAD
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # plain file with Range support
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                start = int(rng.split("=")[1].split("-")[0])
            except ValueError:
                start = 0
        body = PAYLOAD[start:]
        status = 206 if start else 200
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_server() -> http.server.ThreadingHTTPServer:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    server = start_server()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)

            print("== plain download ==")
            dest = net.download_file(f"{base}/file.bin", out / "file.bin")
            check(dest.read_bytes() == PAYLOAD, "file downloaded in full")
            check(not (out / "file.bin.part").exists(), ".part renamed")

            print("== resume (Range) ==")
            part = out / "resume.bin.part"
            part.write_bytes(PAYLOAD[: len(PAYLOAD) // 2])
            dest2 = net.download_file(f"{base}/file.bin", out / "resume.bin")
            check(dest2.read_bytes() == PAYLOAD, "incomplete .part resumed correctly")

            print("== truncated response: retry and verification ==")
            MODE["mode"] = "flaky"
            MODE["flaky_left"] = 1     # first response truncated, second complete
            dest3 = net.download_file(f"{base}/flaky.bin", out / "flaky.bin", backoff_cap=1)
            check(dest3.read_bytes() == PAYLOAD, "file intact after the retry")

            MODE["flaky_left"] = 99    # server always truncates -> attempts exhausted
            try:
                net.download_file(f"{base}/flaky.bin", out / "hopeless.bin", timeout=5, max_attempts=3, backoff_cap=1)
                check(False, "a permanently truncated file must raise an error")
            except net.NetworkError as exc:
                check("попыт" in str(exc), f"clear error after retries: {str(exc)[:70]}…")
            check(not (out / "hopeless.bin").exists(), "broken file NOT saved as ready")
            MODE["flaky_left"] = 0

            print("== 404 -> NetworkError ==")
            try:
                net.download_file(f"{base}/missing.bin", out / "missing.bin")
                check(False, "404 must raise NetworkError")
            except net.NetworkError as exc:
                check("404" in str(exc), f"the 404 error is clear: {str(exc)[:60]}")

            print("== cancellation during a retry pause — instant ==")
            cancel_flag = {"cancel": False}

            def cancel_soon():
                time.sleep(0.7)
                cancel_flag["cancel"] = True

            threading.Thread(target=cancel_soon, daemon=True).start()
            started = time.time()
            try:
                net.download_file(
                    f"{base}/drop.bin", out / "cancel.bin",
                    cancel=lambda: cancel_flag["cancel"], timeout=5, backoff_cap=8,
                )
                check(False, "cancellation must interrupt the download")
            except net.DownloadCancelled:
                elapsed = time.time() - started
                check(elapsed < 2.5, f"cancel fired fast, in {elapsed:.2f} s (would sleep 2-16 s)")
    finally:
        server.shutdown()

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("NETWORK TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
