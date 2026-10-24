"""Фоновые потоки Qt: установка пакетов, скачивание моделей, распознавание.

Всё тяжёлое выполняется ЗДЕСЬ, а не в GUI-потоке: интерфейс не подвисает,
прогресс приходит сигналами, отмена — через флаг.
"""

from __future__ import annotations

import json
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


def stop_worker(worker: Optional[QThread], wait_ms: int = 4000, label: str = "") -> None:
    """Гарантированно останавливает QThread (используется при закрытии окна).

    Порядок: cancel() (если есть) → wait → terminate. Без terminate() поток,
    застрявший в сетевом ретрае или subprocess, доживал до «QThread: Destroyed
    while thread is still running» в момент удаления объектов Qt.
    """
    if worker is None:
        return
    try:
        if not worker.isRunning():
            return
        cancel = getattr(worker, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # noqa: BLE001
                pass
        if worker.wait(wait_ms):
            return
        log.warning("Воркер %s не остановился за %d мс — завершаю принудительно",
                    label or type(worker).__name__, wait_ms)
        worker.terminate()
        worker.wait(2000)
    except RuntimeError:
        pass    # C++-объект уже удалён


# --------------------------------------------------------------------------- #
#  Установка пакетов в .venv
# --------------------------------------------------------------------------- #


class PackageInstaller(QThread):
    """Ставит (или удаляет — mode="uninstall") pip-пакеты строго в .venv проекта."""

    outputLine = Signal(str)
    progress = Signal(float, str)
    finishedWith = Signal(bool, str)

    def __init__(
        self,
        packages: Sequence[str],
        title: str = "",
        parent=None,
        pre_uninstall: Optional[Sequence[str]] = None,
        mode: str = "install",
    ) -> None:
        super().__init__(parent)
        self.packages = list(packages)
        self.title = title or ("Удаление пакетов" if mode == "uninstall" else "Установка пакетов")
        self.mode = mode
        self._cancel = False
        self._proc: Optional[subprocess.Popen] = None
        # что удалить ПЕРЕД установкой (например, CPU-сборку torch перед
        # CUDA-сборкой: иначе pip скажет «already satisfied» и ничего не поставит)
        self.pre_uninstall = list(pre_uninstall or [])

    def cancel(self) -> None:
        self._cancel = True
        # убиваем дочерний pip/uv сразу: иначе поток висит на чтении stdout,
        # пока пакетник молчит (резолвит зависимости), и окно не закрывается
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

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

        if self.mode == "uninstall":
            cmd = [python, "-m", "pip", "uninstall", "-y", *self.packages]
            self.progress.emit(0.05, "Удаляю пакеты…")
            self.outputLine.emit("$ " + " ".join(cmd))
            code = self._stream(cmd, env)
            if self._cancel:
                self.finishedWith.emit(False, "Удаление отменено")
                return
            if code != 0:
                self.finishedWith.emit(False, f"pip uninstall завершился с ошибкой (код {code})")
                return
            self.progress.emit(1.0, "Готово")
            self.finishedWith.emit(True, f"Удалено за {time.time() - started:.0f} с")
            return

        requirements_file = next((p for p in self.packages if p.startswith("-r ")), "")
        if requirements_file and not any(p.startswith("-") is False for p in self.packages):
            cmd = [python, "-m", "pip", "install", *self.packages]
        else:
            uv = self._find_uv(python)
            if uv and not requirements_file:
                cmd = [uv, "pip", "install", "--python", python, *self.packages]
                # uv по умолчанию берёт пакет из ПЕРВОГО индекса, где тот есть
                # (first-index): с дополнительным CUDA-индексом PyTorch он взял
                # бы CPU-сборку torch с PyPI. unsafe-best-match заставляет
                # сравнить версии во всех индексах — и выбрать +cu126, которая
                # по PEP 440 старше чистой версии. pip в этом не нуждается.
                if "--extra-index-url" in self.packages:
                    cmd += ["--index-strategy", "unsafe-best-match"]
            else:
                cmd = [
                    python, "-m", "pip", "install",
                    "--disable-pip-version-check", "--no-input", "--prefer-binary",
                    "--timeout", "60", "--retries", "3", *self.packages,
                ]

        if self.pre_uninstall:
            self.progress.emit(0.03, "Удаляю старую версию: " + ", ".join(self.pre_uninstall))
            uninstall_cmd = [python, "-m", "pip", "uninstall", "-y", *self.pre_uninstall]
            self.outputLine.emit("$ " + " ".join(uninstall_cmd))
            self._stream(uninstall_cmd, env)   # код возврата не важен: пакета могло и не быть

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
                from .. import logutils

                current = logutils.current_log_file()
                log_hint = str(current) if current else str(paths.logs_dir())
                self.finishedWith.emit(
                    False,
                    f"Установка завершилась с ошибкой (код {code}).\n"
                    f"Подробности: {log_hint}\n"
                    "(подробный лог включается в Сервис → Настройки → «Журнал и диагностика»)",
                )
                return

        self.progress.emit(1.0, "Готово")
        # если transformers уже импортирован в этом процессе (например, до этого
        # грузили GigaAM), его lru-кеши доступности устарели: только что
        # установленные пакеты (accelerate и т.п.) ими «не видны». Сбрасываем —
        # следующий запуск модели подхватит свежие пакеты БЕЗ перезапуска.
        try:
            if "transformers" in sys.modules:
                from ..asr.torch_engines import reset_transformers_caches

                cleared = reset_transformers_caches()
                log.debug("После установки пакетов сброшено lru-кэшей transformers: %d", cleared)
        except Exception as exc:  # noqa: BLE001
            log.debug("Не удалось сбросить кэши transformers после установки: %s", exc)
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

        self._proc = proc
        log.debug("PackageInstaller subprocess: %s", " ".join(cmd))
        try:
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
        finally:
            self._proc = None

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
        log.debug("ModelDownloadWorker запущен: %s (source=%s, root=%s)",
                  self.entry.id, self.entry.source, self.root)
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
#  Диагностика целостности (файлы проекта + аудит .venv)
# --------------------------------------------------------------------------- #


class EnvAuditWorker(QThread):
    """Проверяет файлы проекта, папки и пакеты .venv (см. core/integrity.py)."""

    finishedWith = Signal(object)   # dict-отчёт

    def run(self) -> None:  # noqa: D102
        from ..core import integrity

        try:
            report = integrity.full_audit()
        except Exception as exc:  # noqa: BLE001
            log.error("Сбой диагностики: %s\n%s", exc, traceback.format_exc())
            report = {
                "python": "",
                "missing_files": [],
                "folders": [],
                "base_missing": [],
                "extra_packages": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        self.finishedWith.emit(report)


class FileRestoreWorker(QThread):
    """Скачивает отсутствующие файлы программы с GitHub."""

    outputLine = Signal(str)
    progress = Signal(float, str)
    finishedWith = Signal(bool, str)

    def __init__(self, files: Sequence[str], parent=None) -> None:
        super().__init__(parent)
        self.files = list(files)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        from ..core import integrity

        if not self.files:
            self.finishedWith.emit(True, "Восстанавливать нечего — все файлы на месте")
            return
        self.progress.emit(0.0, f"Восстановление {len(self.files)} файл(ов) с GitHub…")
        restored, failed = integrity.restore_files(
            self.files,
            progress=lambda fraction, status: (
                self.progress.emit(fraction, status),
                self.outputLine.emit(status),
            )[0],
            cancel=lambda: self._cancel,
        )
        if failed:
            self.finishedWith.emit(
                False,
                f"Восстановлено: {len(restored)}, с ошибкой: {len(failed)}\n" + "\n".join(failed[:8]),
            )
        else:
            self.progress.emit(1.0, "Готово")
            self.finishedWith.emit(True, f"Восстановлено файлов: {len(restored)}")


# --------------------------------------------------------------------------- #
#  Зонд: видит ли torch из .venv CUDA (в отдельном процессе!)
# --------------------------------------------------------------------------- #


class TorchCudaProbeWorker(QThread):
    """Проверяет subprocess'ом, стоит ли в .venv torch и видит ли он CUDA.

    Почему не в процессе GUI:
      * import torch блокирует на секунды — нельзя в GUI-потоке;
      * импортированный torch «фиксирует» сборку (CPU/CUDA) на всю сессию,
        а нам нужно знать состояние диска, не трогая текущий процесс.
    """

    probed = Signal(bool, bool, str)   # torch_installed, cuda_available, detail

    _SCRIPT = (
        "import importlib.util as iu, json\n"
        "if iu.find_spec('torch') is None:\n"
        "    print('PROBE' + json.dumps({'installed': False}))\n"
        "else:\n"
        "    import torch\n"
        "    print('PROBE' + json.dumps({"
        "'installed': True,"
        "'cuda': bool(torch.cuda.is_available()),"
        "'version': torch.__version__,"
        "'cuda_ver': torch.version.cuda or ''}))\n"
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        proc: Optional[subprocess.Popen] = None
        try:
            proc = subprocess.Popen(
                [venv_python(), "-c", self._SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            # ждём кусочками по 0.5 с: закрытие окна не должно ждать весь таймаут
            deadline = time.time() + 180
            out_text, err_text = "", ""
            while time.time() < deadline:
                if self._cancel:
                    proc.kill()
                    return
                try:
                    out_text, err_text = proc.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
            for line in reversed((out_text or "").splitlines()):
                if line.startswith("PROBE"):
                    data = json.loads(line[len("PROBE"):])
                    detail = str(data.get("version") or "")
                    if data.get("cuda_ver"):
                        detail += f" + CUDA {data['cuda_ver']}"
                    self.probed.emit(
                        bool(data.get("installed")), bool(data.get("cuda")), detail
                    )
                    return
            log.debug("Зонд torch не дал результата: rc=%s stderr=%s",
                      proc.returncode if proc else "?", (err_text or "")[-300:])
        except Exception as exc:  # noqa: BLE001
            log.debug("Зонд torch не запустился: %s", exc)
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
        if not self._cancel:
            self.probed.emit(False, False, "")


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
        #: движку не хватило библиотек — GUI покажет диалог установки и повторит
        self.engine_missing = False

    # ------------------------------------------------------------------ #
    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------------ #
    def run(self) -> None:  # noqa: D102
        started = time.time()
        log.debug(
            "TranscribeWorker запущен: файлов=%d, модель=%s, движок=%s, parallel=%s",
            len(self.files), self.options.model.name, self.options.model.engine,
            self.settings.parallel_files,
        )
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

            # Пакеты движка могли установить ПРЯМО ВО ВРЕМЯ работы программы
            # (штатный сценарий «установка по требованию»). transformers кэширует
            # проверки доступности (lru_cache): без сброса он продолжит считать
            # свежеустановленный accelerate отсутствующим, и загрузка модели
            # упадёт с «requires accelerate». Сброс безопасен и дёшев.
            try:
                if "transformers" in sys.modules:
                    from ..asr.torch_engines import reset_transformers_caches

                    cleared = reset_transformers_caches()
                    if cleared:
                        log.debug("Перед загрузкой движка сброшено lru-кэшей transformers: %d", cleared)
            except Exception as exc:  # noqa: BLE001
                log.debug("Сброс кэшей transformers не выполнен: %s", exc)

            hardware = registry.hardware()
            log.debug("Железо для задачи: %s", hardware.get("summary", "").replace("\n", " | "))
            self.logLine.emit(hardware.get("summary", "").replace("\n", " | "))
            ffmpeg = find_ffmpeg()
            self.logLine.emit(f"ffmpeg: {ffmpeg or 'НЕ НАЙДЕН (будет использован PyAV)'}")

            parallel = max(1, min(4, int(self.settings.parallel_files or 1)))
            if parallel == 1:
                ok, fail = self._run_sequential()
            else:
                ok, fail = self._run_parallel(parallel)

            log.debug(
                "TranscribeWorker завершён за %.1f с: ok=%d fail=%d cancel=%s",
                time.time() - started, ok, fail, self.cancelled,
            )
            self.finishedWith.emit(ok, fail, time.time() - started)
        except Exception as exc:  # noqa: BLE001
            # полный стектрейс — в лог; пользователю показываем только суть
            log.error("Сбой воркера: %s\n%s", exc, traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")
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

            try:
                engine = registry.get(
                    self.options.model,
                    {
                        "device": self.options.device,
                        "compute_type": self.options.compute_type,
                        "cpu_threads": self.options.cpu_threads,
                        "gpu_index": self.options.gpu_index,
                    },
                )
            except EngineNotInstalled as exc:
                # Коду модели не хватило библиотеки: не стектрейс, а диалог
                # установки с автопродолжением задачи после установки.
                self.engine_missing = True
                self.engineMissing.emit(exc.engine, list(exc.packages))
                break
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
            try:
                engine = thread_engine(
                    self.options.model,
                    {
                        "device": self.options.device,
                        "compute_type": self.options.compute_type,
                        "cpu_threads": self.options.cpu_threads,
                        "gpu_index": self.options.gpu_index,
                    },
                )
            except EngineNotInstalled as exc:
                self.engine_missing = True
                self.engineMissing.emit(exc.engine, list(exc.packages))
                return FileResult(source=source, error="не хватает компонентов движка")
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
            # выгружаем модели ВНУТРИ рабочих потоков: кэш движков лежит в их
            # thread-local, и вызов release из нашего потока не доставал его —
            # память (в т.ч. VRAM) висела до смерти потоков
            for _ in range(workers):
                pool.submit(release_thread_engines)

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
