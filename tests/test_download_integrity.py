# -*- coding: utf-8 -*-
"""Тесты целостности установки моделей (задача 5b).

Всё работает на ЛОКАЛЬНОМ http-сервере — интернет не нужен. Проверяется:
  * успешная установка: installed.json на месте, маркер частичной установки
    снят, каталог видит модель как установленную, движок определён (qwen-asr);
  * «недокачанный» файл (размер не совпал с планом) -> InstallError, папка
    помечена .localasr-partial.json, каталог НЕ считает модель установленной
    (регрессия бага «Qwen3-ASR без preprocessor_config.json выглядел готовым»);
  * отмена установки -> маркер остаётся, модель не «установлена»;
  * повторная установка пропускает уже скачанные целые файлы (resume) и
    успешно завершается;
  * uninstall_model убирает и частичную папку тоже.

Запуск:  cd <корень проекта> && python tests/test_download_integrity.py
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
#  Локальный «Hugging Face»
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

#: файлы, которые сервер намеренно отдаёт ОБРЕЗАННЫМИ (симуляция обрыва сети)
TRUNCATED: set[str] = set()
#: счётчик обращений к каждому файлу (для проверки resume)
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

    def log_message(self, *args):  # тишина в консоли
        pass


def start_server() -> tuple[http.server.ThreadingHTTPServer, str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def make_entry() -> ModelEntry:
    return ModelEntry(
        id="qwen3-asr-test",
        name="Qwen3-ASR (тестовая)",
        engine=ENGINE_QWEN_ASR,
        source="test/qwen3-asr",
        languages=["ru", "en"],
    )


def patch_net(base_url: str):
    """Подменяет HF-API на локальный сервер."""

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

            print("== успешная установка ==")
            entry = make_entry()
            result = install_model(entry, root=root)
            model_dir = root / "models" / "qwen3-asr-test"
            check(model_dir.is_dir(), f"папка модели создана: {model_dir.name}")
            check((model_dir / META_FILENAME).is_file(), "installed.json записан")
            check(not (model_dir / PARTIAL_MARKER).exists(), "маркер частичной установки снят")
            meta = json.loads((model_dir / META_FILENAME).read_text(encoding="utf-8"))
            check(meta["engine"] == ENGINE_QWEN_ASR, f"движок определён: {meta['engine']}")
            check(sorted(meta.get("files") or []) == sorted(FILES), "в метаданных список файлов")
            check(result.downloaded_bytes == sum(len(v) for v in FILES.values()),
                  "скачанный объём совпал с планом")

            catalog = ModelCatalog(root=root)
            installed = catalog.get("qwen3-asr-test")
            check(installed is not None and installed.installed, "каталог видит модель установленной")

            print("== недокачанный файл -> установка падает, модель НЕ «установлена» ==")
            uninstall_model(installed, root=root)
            check(not model_dir.exists(), "старая папка удалена")
            TRUNCATED.add("model.safetensors")
            entry2 = make_entry()
            try:
                install_model(entry2, root=root)
                check(False, "обрезанный файл должен приводить к InstallError")
            except InstallError as exc:
                check("повреждены или недокачаны" in str(exc), f"понятная ошибка: {str(exc)[:80]}…")
            check((model_dir / PARTIAL_MARKER).is_file(), "маркер частичной установки остался")
            check(not (model_dir / META_FILENAME).exists(), "installed.json НЕ создан")
            catalog.refresh_installed()
            entry_after = catalog.get("qwen3-asr-test")
            check(
                entry_after is None or not entry_after.installed,
                "каталог НЕ считает недокачанную модель установленной",
            )

            print("== повторная установка: resume целых файлов ==")
            TRUNCATED.clear()
            for name in list(HITS):
                HITS[name] = 0
            entry3 = make_entry()
            install_model(entry3, root=root)
            check((model_dir / META_FILENAME).is_file(), "после докачки installed.json на месте")
            check(not (model_dir / PARTIAL_MARKER).exists(), "маркер снят")
            check(HITS.get("model.safetensors", 0) >= 1, "недокачанный файл перекачан заново")
            check(HITS.get("config.json", 0) == 0 and HITS.get("tokenizer.json", 0) == 0,
                  "целые файлы НЕ перекачивались (resume работает)")

            print("== отмена установки ==")
            uninstall_model(catalog.get("qwen3-asr-test") or entry3, root=root)
            cancel_flag = {"cancel": False}

            def cancel_after_first():
                # отменяем после первого же обращения к серверу (с таймаутом)
                deadline = time.time() + 10
                while sum(HITS.values()) < 1 and time.time() < deadline:
                    time.sleep(0.005)
                cancel_flag["cancel"] = True

            watcher = threading.Thread(target=cancel_after_first, daemon=True)
            watcher.start()
            entry4 = make_entry()
            try:
                install_model(entry4, root=root, cancel=lambda: cancel_flag["cancel"])
                check(False, "отмена должна прерывать установку")
            except net.DownloadCancelled:
                check(True, "отмена прервала установку (DownloadCancelled)")
            check((model_dir / PARTIAL_MARKER).is_file(), "после отмены маркер остался")
            catalog.refresh_installed()
            after_cancel = catalog.get("qwen3-asr-test")
            check(after_cancel is None or not after_cancel.installed,
                  "отменённая установка не видна как «установленная»")
            check(not any(p.suffix == ".part" for p in model_dir.rglob("*.part")),
                  "недокачанные .part-файлы убраны")

            print("== uninstall убирает частичную папку ==")
            check(uninstall_model(entry4, root=root), "uninstall_model удалил частичную папку")
            check(not model_dir.exists(), "папки больше нет")
    finally:
        net.hf_tree, net.hf_resolve_url = original
        server.shutdown()

    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("ТЕСТ ЦЕЛОСТНОСТИ УСТАНОВКИ ПРОЙДЕН")
    return 0


if __name__ == "__main__":
    sys.exit(main())
