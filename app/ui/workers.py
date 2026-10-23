"""Фоновые потоки Qt: установка пакетов, скачивание моделей, распознавание.

Всё тяжёлое выполняется ЗДЕСЬ, а не в GUI-потоке: интерфейс не подвисает,
прогресс приходит сигналами, отмена — через флаг.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PySide6.QtCore import QThread, Signal

from .. import paths
from ..asr.base import EngineNotInstalled
from ..asr.registry import registry, release_thread_engines, thread_engine
from ..asr.transcriber import FileResult, JobOptions, transcribe_file
from ..core import downloader
from ..core.downloader import InstallError
from ..core.ffmpeg import decode_audio, find_ffmpeg
from ..core.models_catalog import ModelCatalog, ModelEntry
from ..constants import MAX_PREFETCH, SAMPLE_RATE
from ..logutils import get_logger
from ..settings import Settings

log = get_logger("workers")


def venv_python() -> str:
    """Интерпретатор, в который ставим пакеты (мы сами внутри .venv)."""
    return sys.executable or str(paths.venv_dir() / "Scripts" / "python.exe")


# --------------------------------------------------------------------------- #
#  Установка пакетов в .venv
# --------------------------------------------------------------------------- #


class PackageInstaller(QThread):
    """Ставит pip-пакеты строго в виртуальное окружение проекта."""

    outputLine = Signal(str)
    progress = Signal(float, str)
    finishedWith = Signal(bool, str)

    def __init__(self, packages: Sequence[str], title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.packages = list(packages)
        self.title = title or "Установка пакетов"
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    # ------------------------------------------------------------------ #
    def run(self) -> None:  # noqa: D102
        started = time.time()
        python = venv_python()
        self.progress.emit(0.02, f"Проверяю окружение: {python}")
        self.outputLine.emit(f"python: {python}")

        if not self.packages:
            self.finishedWith.emit(True, "Нечего устанавливать")
            return

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        requirements_file = next((p for p in self.packages if p.startswith("-r ")), "")
        if requirements_file and not any(p.startswith("-") is False for p in self.packages):
            cmd = [python, "-m", "pip", "install", *self.packages]
        else:
            uv = self._find_uv(python)
            if uv and not requirements_file:
                cmd = [uv, "pip", "install", "--python", python, *self.packages]
            else:
                cmd = [
                    python, "-m", "pip", "install",
                    "--disable-pip-version-check", "--no-input", "--prefer-binary",
                    "--timeout", "60", "--retries", "3", *self.packages,
                ]

        self.progress.emit(0.05, "Запускаю установку…")
        self.outputLine.emit("$ " + " ".join(cmd))

        code = self._stream(cmd, env)
        if self._cancel:
            self.finishedWith.emit(False, "Установка отменена")
            return
        if code != 0:
            # uv иногда не может разрешить окружение — пробуем обычный pip
            if cmd[0] != python:
                self.outputLine.emit("Первый способ не сработал, пробую pip…")
                code = self._stream(
                    [python, "-m", "pip", "install", "--disable-pip-version-check",
                     "--no-input", "--prefer-binary", *self.packages],
                    env,
                )
            if code != 0:
                self.finishedWith.emit(
                    False,
                    f"Установка завершилась с ошибкой (код {code}).\n"
                    f"Лог: {paths.logs_dir() / 'localasr.log'}",
                )
                return

        self.progress.emit(1.0, "Готово")
        self.finishedWith.emit(True, f"Установлено за {time.time() - started:.0f} с")

    # ------------------------------------------------------------------ #
    def _stream(self, cmd: List[str], env: Dict[str, str]) -> int:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(paths.ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.outputLine.emit(f"!! не удалось запустить: {exc}")
            return 127

        assert proc.stdout is not None
        for line in proc.stdout:
            if self._cancel:
                proc.kill()
                break
            text = line.rstrip()
            if text:
                self.outputLine.emit(text)
                low = text.lower()
                if "downloading" in low or "collecting" in low:
                    self.progress.emit(-1.0, text[:160])
        proc.wait()
        return proc.returncode

    @staticmethod
    def _find_uv(python: str) -> Optional[str]:
        import shutil

        found = shutil.which("uv")
        if found:
            return found
        candidate = Path(python).parent / ("uv.exe" if os.name == "nt" else "uv")
        return str(candidate) if candidate.is_file() else None


# --------------------------------------------------------------------------- #
#  Скачивание модели
# --------------------------------------------------------------------------- #


class ModelDownloadWorker(QThread):
    outputLine = Signal(str)
    progress = Signal(float, str)
    finishedWith = Signal(bool, str, object)   # ok, message, ModelEntry|None

    def __init__(self, entry: ModelEntry, root: Optional[Path] = None, parent=None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.root = Path(root) if root else None
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        name = self.entry.name
        self.progress.emit(0.0, f"Подготовка к скачиванию: {name}")
        self.outputLine.emit(f"Модель: {name} ({self.entry.id})")
        self.outputLine.emit(f"Источник: {self.entry.source}")

        try:
            result = downloader.install_model(
                self.entry,
                root=self.root,
                progress=lambda fraction, status: (
                    self.progress.emit(fraction, status),
                    self.outputLine.emit(status),
                )[0],
                cancel=lambda: self._cancel,
            )
        except downloader.net.DownloadCancelled:  # type: ignore[attr-defined]
            self.finishedWith.emit(False, "Скачивание отменено. Недокачанные файлы можно продолжить позже.", None)
            return
        except InstallError as exc:
            self.finishedWith.emit(False, str(exc), None)
            return
        except Exception as exc:  # noqa: BLE001
            log.error("Сбой установки модели: %s\n%s", exc, traceback.format_exc())
            self.finishedWith.emit(False, f"{type(exc).__name__}: {exc}", None)
            return

        if self._cancel:
            self.finishedWith.emit(False, "Скачивание отменено", None)
            return

        size_text = paths.human_size(result.downloaded_bytes)
        self.progress.emit(1.0, "Готово")
        self.outputLine.emit(
            f"Установлено в {result.directory} ({size_text} за {result.duration_seconds:.0f} с)"
        )
        self.finishedWith.emit(
            True,
            f"«{name}» установлена в {result.directory.name}/ ({size_text})",
            result.entry,
        )


# --------------------------------------------------------------------------- #
#  Обновление каталога из сети
# --------------------------------------------------------------------------- #


class CatalogRefreshWorker(QThread):
    finishedWith = Signal(bool, str)

    def run(self) -> None:  # noqa: D102
        try:
            catalog = ModelCatalog()
            updated = catalog.refresh_remote(force=True)
            count = len(catalog.entries)
            if updated:
                self.finishedWith.emit(True, f"Каталог обновлён: {count} моделей")
            else:
                self.finishedWith.emit(False, f"Обновлений нет (использую локальный каталог, {count} моделей)")
        except Exception as exc:  # noqa: BLE001
            self.finishedWith.emit(False, f"Не удалось обновить каталог: {exc}")


# --------------------------------------------------------------------------- #
#  Распознавание очереди файлов
# --------------------------------------------------------------------------- #


class TranscribeWorker(QThread):
    """Обрабатывает очередь файлов.

    Сигналы:
      overallProgress(done, total)      — общий прогресс по файлам
      fileProgress(index, fraction, text) — прогресс текущего файла
      segmentReady(index, text, start, end) — «живая» лента текста
      fileFinished(index, FileResult)
      finishedWith(ok_count, fail_count, seconds)
      engineMissing(engine, packages)   — движок не установлен
    """

    overallProgress = Signal(int, int)
    fileProgress = Signal(int, float, str)
    segmentReady = Signal(int, str, float, float)
    fileFinished = Signal(int, object)
    logLine = Signal(str)
    finishedWith = Signal(int, int, float)
    failed = Signal(str)
    engineMissing = Signal(str, object)

    def __init__(
        self,
        files: Sequence[Path],
        options: JobOptions,
        settings: Settings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.files = [Path(f) for f in files]
        self.options = options
        self.settings = settings
        self._cancel = threading.Event()

    # ------------------------------------------------------------------ #
    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------------ #
    def run(self) -> None:  # noqa: D102
        started = time.time()
        try:
            if not self.files:
                self.finishedWith.emit(0, 0, 0.0)
                return

            entry = self.options.model
            if not entry.installed or not entry.install_dir:
                self.failed.emit(f"Модель «{entry.name}» не установлена")
                return

            try:
                registry.ensure_engine_installed(entry.engine)
            except EngineNotInstalled as exc:
                self.engineMissing.emit(exc.engine, list(exc.packages))
                return

            hardware = registry.hardware()
            self.logLine.emit(hardware.get("summary", "").replace("\n", " | "))
            ffmpeg = find_ffmpeg()
            self.logLine.emit(f"ffmpeg: {ffmpeg or 'НЕ НАЙДЕН (будет использован PyAV)'}")

            parallel = max(1, min(4, int(self.settings.parallel_files or 1)))
            if parallel == 1:
                ok, fail = self._run_sequential()
            else:
                ok, fail = self._run_parallel(parallel)

            self.finishedWith.emit(ok, fail, time.time() - started)
        except Exception as exc:  # noqa: BLE001
            log.error("Сбой воркера: %s\n%s", exc, traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=6)}")
        finally:
            try:
                release_thread_engines()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    def _run_sequential(self) -> tuple[int, int]:
        ok = fail = 0
        total = len(self.files)

        # Префетч декодирования: пока модель считает файл N, соседний поток уже
        # вытаскивает звук из файла N+1. Включаем только если RAM хватает —
        # декодированная 3-часовая лекция сама по себе весит ~700 МБ.
        prefetch_depth = self._prefetch_depth()
        self.logLine.emit(
            f"Префетч декодирования: {prefetch_depth} файл(ов) вперёд"
            if prefetch_depth else "Префетч декодирования выключен (мало свободной памяти)"
        )

        executor: Optional[ThreadPoolExecutor] = None
        futures: Dict[int, Future] = {}
        if prefetch_depth:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode")
            for ahead in range(1, prefetch_depth + 1):
                if ahead < total:
                    futures[ahead] = executor.submit(self._decode_one, self.files[ahead])

        for index, source in enumerate(self.files):
            if self.cancelled:
                self.logLine.emit("Отменено пользователем")
                break
            self.overallProgress.emit(index, total)

            audio: Any = None
            duration = 0.0
            decode_error = ""
            if executor is not None:
                audio, duration, decode_error = self._take_future(futures.pop(index, None))
                next_index = index + prefetch_depth
                if next_index < total:
                    futures[next_index] = executor.submit(self._decode_one, self.files[next_index])

            if decode_error:
                result = FileResult(source=source, error=decode_error)
                fail += 1
                self.fileFinished.emit(index, result)
                self.logLine.emit(f"✗ {source.name}: {decode_error}")
                continue

            engine = registry.get(
                self.options.model,
                {
                    "device": self.options.device,
                    "compute_type": self.options.compute_type,
                    "cpu_threads": self.options.cpu_threads,
                    "gpu_index": self.options.gpu_index,
                },
            )
            result = self._process_one(index, source, audio, duration, engine)
            del audio
            ok += int(result.ok)
            fail += int(not result.ok)

        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self.overallProgress.emit(total, total)
        return ok, fail

    def _prefetch_depth(self) -> int:
        """Сколько файлов декодировать заранее (0 = выключить)."""
        if self.cancelled:
            return 0
        try:
            hardware = registry.hardware()
            available = int(hardware.get("ram_available_mb") or 0)
            model_ram = int(self.options.model.ram_mb or 0)
        except Exception:  # noqa: BLE001
            return MAX_PREFETCH
        if not available:
            return MAX_PREFETCH
        # нужен запас: модель + один декодируемый файл (~800 МБ на час аудио)
        if available - model_ram < 2400:
            return 0
        if available - model_ram < 4800:
            return 1
        return MAX_PREFETCH

    def _run_parallel(self, workers: int) -> tuple[int, int]:
        total = len(self.files)
        ok = fail = 0
        lock = threading.Lock()
        self.overallProgress.emit(0, total)

        def handle(index: int, source: Path) -> FileResult:
            if self.cancelled:
                return FileResult(source=source, error="отменено")
            engine = thread_engine(
                self.options.model,
                {
                    "device": self.options.device,
                    "compute_type": self.options.compute_type,
                    "cpu_threads": self.options.cpu_threads,
                    "gpu_index": self.options.gpu_index,
                },
            )
            return self._process_one(index, source, None, 0.0, engine)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asr") as pool:
            futures = {pool.submit(handle, index, source): index for index, source in enumerate(self.files)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = FileResult(source=self.files[index], error=f"{type(exc).__name__}: {exc}")
                with lock:
                    ok += int(result.ok)
                    fail += int(not result.ok)
                    self.fileFinished.emit(index, result)
                    self.overallProgress.emit(ok + fail, total)

        self.overallProgress.emit(total, total)
        return ok, fail

    # ------------------------------------------------------------------ #
    def _process_one(
        self,
        index: int,
        source: Path,
        audio: Any,
        duration: float,
        engine: Any,
    ) -> FileResult:
        def progress(fraction: float, stage: str) -> None:
            self.fileProgress.emit(index, fraction, stage)

        def on_segment(segment) -> None:
            self.segmentReady.emit(index, segment.text, segment.start, segment.end)

        self.logLine.emit(f"→ {source.name}")
        try:
            result = transcribe_file(
                source,
                self.options,
                progress=progress,
                on_segment=on_segment,
                cancel=lambda: self.cancelled,
                audio=audio,
                audio_duration=duration,
                engine=engine,
            )
        except EngineNotInstalled as exc:
            self.engineMissing.emit(exc.engine, list(exc.packages))
            result = FileResult(source=source, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            log.error("Ошибка обработки %s: %s\n%s", source.name, exc, traceback.format_exc())
            result = FileResult(source=source, error=f"{type(exc).__name__}: {exc}")

        if result.ok:
            outputs = ", ".join(p.name for p in result.outputs)
            self.logLine.emit(
                f"✓ {source.name} → {outputs} · {result.words} слов · {result.speed_x:.1f}× · "
                f"{result.elapsed_seconds:.0f} с"
                + (f" · отброшено сегментов: {result.dropped}" if result.dropped else "")
            )
        else:
            self.logLine.emit(f"✗ {source.name}: {result.error}")
        self.fileFinished.emit(index, result)
        return result

    def _decode_one(self, source: Path) -> tuple[Any, float, str]:
        """Префетч: декодируем звук, пока модель занята предыдущим файлом."""
        try:
            audio = decode_audio(source, SAMPLE_RATE, cancel=lambda: self.cancelled)
            duration = len(audio) / SAMPLE_RATE
            return audio, duration, ""
        except Exception as exc:  # noqa: BLE001
            return None, 0.0, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _take_future(future: Optional[Future]) -> tuple[Any, float, str]:
        if future is None:
            return None, 0.0, ""
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001
            return None, 0.0, f"{type(exc).__name__}: {exc}"
