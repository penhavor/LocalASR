# -*- coding: utf-8 -*-
"""Тесты устойчивости net.download_file (задачи 5c + верификация размера).

Локальный HTTP-сервер, интернет не нужен:
  * обычное скачивание: файл цел, .part переименован;
  * доозагрузка (Range): существующий .part докачивается, а не качается заново;
  * «тихо обрезанный» ответ (Content-Length больше тела) -> ретрай и ошибка,
    битый файл НЕ сохраняется как готовый;
  * отмена во время паузы ретрая срабатывает БЫСТРО (регрессия
    «QThread: Destroyed while thread is still running» — поток спал по 16 с
    и не слышал отмену);
  * HTTP 404 -> NetworkError с первого раза.

Запуск:  cd <корень проекта> && python tests/test_net_robust.py
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

from app.core import net  # noqa: E402

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


PAYLOAD = bytes(range(256)) * 400          # 102 400 байт
MODE = {"mode": "normal", "flaky_left": 0}


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802
        name = self.path.lstrip("/").split("?")[0]
        if name == "missing.bin":
            self.send_error(404)
            return
        if name == "drop.bin":
            # обрыв соединения сразу после заголовков: клиент мгновенно
            # получает ConnectionError и уходит в паузу ретрая — именно её
            # должен уметь прерывать cancel
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.close_connection = True
            return
        if name == "flaky.bin":
            # первые N обращений обрываются на середине (Content-Length врёт,
            # соединение закрывается — клиент видит IncompleteRead сразу)
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

        # обычный файл с поддержкой Range
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

            print("== обычное скачивание ==")
            dest = net.download_file(f"{base}/file.bin", out / "file.bin")
            check(dest.read_bytes() == PAYLOAD, "файл скачан целиком")
            check(not (out / "file.bin.part").exists(), ".part переименован")

            print("== доозагрузка (Range) ==")
            part = out / "resume.bin.part"
            part.write_bytes(PAYLOAD[: len(PAYLOAD) // 2])
            dest2 = net.download_file(f"{base}/file.bin", out / "resume.bin")
            check(dest2.read_bytes() == PAYLOAD, "недокачанный .part докачан корректно")

            print("== обрезанный ответ: ретрай и верификация ==")
            MODE["mode"] = "flaky"
            MODE["flaky_left"] = 1     # первый ответ обрезан, второй — целый
            dest3 = net.download_file(f"{base}/flaky.bin", out / "flaky.bin", backoff_cap=1)
            check(dest3.read_bytes() == PAYLOAD, "после ретрая файл цел")

            MODE["flaky_left"] = 99    # сервер всегда обрезает -> исчерпание попыток
            try:
                net.download_file(f"{base}/flaky.bin", out / "hopeless.bin", timeout=5, max_attempts=3, backoff_cap=1)
                check(False, "постоянно обрезанный файл должен приводить к ошибке")
            except net.NetworkError as exc:
                check("попыт" in str(exc), f"понятная ошибка после ретраев: {str(exc)[:70]}…")
            check(not (out / "hopeless.bin").exists(), "битый файл НЕ сохранён как готовый")
            MODE["flaky_left"] = 0

            print("== 404 -> NetworkError ==")
            try:
                net.download_file(f"{base}/missing.bin", out / "missing.bin")
                check(False, "404 должен приводить к NetworkError")
            except net.NetworkError as exc:
                check("404" in str(exc), f"ошибка 404 понятна: {str(exc)[:60]}")

            print("== отмена во время паузы ретрая — мгновенная ==")
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
                check(False, "отмена должна прерывать скачивание")
            except net.DownloadCancelled:
                elapsed = time.time() - started
                check(elapsed < 2.5, f"отмена сработала быстро, за {elapsed:.2f} с (спал бы 2-16 с)")
    finally:
        server.shutdown()

    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("ТЕСТ СЕТИ ПРОЙДЕН")
    return 0


if __name__ == "__main__":
    sys.exit(main())
