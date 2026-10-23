"""Точка входа приложения: GUI по умолчанию + headless-режим (CLI)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

from . import paths
from .constants import APP_NAME, APP_VERSION, MEDIA_EXTENSIONS, OUTPUT_FORMATS
from .logutils import get_logger, setup_logging

log = get_logger("cli")


# --------------------------------------------------------------------------- #
#  Аргументы
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="LocalASR",
        description="Локальное распознавание речи: аудио/видео → текст (без облака).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python run_app.py                                  # запустить GUI\n"
            '  python run_app.py list-models                      # что доступно\n'
            '  python run_app.py transcribe "лекция 1.mp4"        # распознать без GUI\n'
            '  python run_app.py transcribe ./lectures -m whisper-large-v3-turbo -f srt\n'
            "  python run_app.py doctor                           # диагностика окружения\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    transcribe = sub.add_parser("transcribe", help="распознать файлы без GUI")
    transcribe.add_argument("inputs", nargs="+", help="файлы и/или папки")
    transcribe.add_argument("-m", "--model", default="", help="ID или имя модели из каталога")
    transcribe.add_argument("-o", "--output", default="", help="папка результата (по умолчанию outputs/)")
    transcribe.add_argument("-f", "--format", default="", choices=list(OUTPUT_FORMATS), help="формат результата")
    transcribe.add_argument("-l", "--language", default=None, help="код языка (ru, en, …) или 'auto'")
    transcribe.add_argument("--beam", type=int, default=None, help="beam size (1–12)")
    transcribe.add_argument("--no-vad", action="store_true", help="не отрезать тишину")
    transcribe.add_argument("--no-clean", action="store_true", help="не чистить галлюцинации")
    transcribe.add_argument("--timestamps", action="store_true", help="таймкоды в txt")
    transcribe.add_argument("--words", action="store_true", help="таймкоды каждого слова (для субтитров)")
    transcribe.add_argument("--device", default=None, choices=[None, "auto", "cpu", "cuda"], help="устройство")
    transcribe.add_argument("--compute-type", default=None, help="int8 / float16 / int8_float16 / default")
    transcribe.add_argument("--threads", type=int, default=None, help="потоки CPU")
    transcribe.add_argument("--batch-size", type=int, default=None, help="батч для batched-режима")
    transcribe.add_argument("--no-batched", action="store_true", help="выключить batched-режим")
    transcribe.add_argument("--prompt", default=None, help="initial_prompt (стиль/термины)")
    transcribe.add_argument("--hotwords", default=None, help="слова-подсказки через запятую")
    transcribe.add_argument("--overwrite", action="store_true", help="перезаписывать существующие файлы")
    transcribe.add_argument("--header", action="store_true", help="добавить шапку с метаинформацией")
    transcribe.add_argument("--all-formats", action="store_true", help="сохранить сразу txt+srt+vtt+json")

    models = sub.add_parser("list-models", aliases=["models"], help="список моделей")
    models.add_argument("--json", action="store_true", help="вывести в JSON")
    models.add_argument("--installed", action="store_true", help="только установленные")

    sub.add_parser("doctor", help="диагностика: Python, venv, ffmpeg, движки, железо")
    sub.add_parser("version", help="версия приложения")

    parser.add_argument("--no-gui", action="store_true", help="не открывать окно (для отладки)")
    return parser


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #


def _pyside_import_hint(exc: BaseException) -> str:
    """Подсказка под конкретную причину отказа импорта PySide6."""
    text = str(exc).lower()
    base = f"{type(exc).__name__}: {exc}\n\n"

    if "cannot open shared object" in text or ".so" in text or "dll" in text or "libglib" in text:
        return base + (
            "PySide6 установлен, но не хватает СИСТЕМНЫХ библиотек Qt.\n\n"
            "Linux (Debian/Ubuntu):\n"
            "  sudo apt install libglib2.0-0 libgl1 libegl1 libxkbcommon0 libdbus-1-3 \\\n"
            "      libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 \\\n"
            "      libxcb-xinerama0 libxcb-randr0 libxcb-render-util0 libfontconfig1\n\n"
            "Linux (Fedora):  sudo dnf install glib2 mesa-libEGL libxkbcommon xcb-util-*\n"
            "macOS:           обычно не требуется; при проблеме — переустанови PySide6\n"
            "                 в .venv (удали .venv и запусти main.py).\n\n"
            "Временный обход (программный рендер):\n"
            "  QT_OPENGL=software QT_QPA_PLATFORM=xcb python main.py\n"
        )

    if "no module named" in text:
        return base + (
            "PySide6 не установлен в .venv.\n"
            "Удали папку .venv и запусти main.py ещё раз — окружение пересоздастся,\n"
            "либо выполни: «Сервис → Проверить и починить окружение (.venv)».\n"
        )

    return base + (
        "Не удалось загрузить PySide6.\n"
        "1) Удали папку .venv и запусти main.py ещё раз.\n"
        "2) Обнови драйвер видеокарты (Qt использует её для отрисовки).\n"
        "3) Попробуй QT_OPENGL=software python main.py\n"
    )


def run_gui(argv: Sequence[str]) -> int:
    """Запускает Qt-приложение. Все импорты PySide6 — внутри функции."""
    try:
        from PySide6.QtCore import qInstallMessageHandler
        from PySide6.QtGui import QSurfaceFormat
        from PySide6.QtWidgets import QApplication, QMessageBox
    except BaseException as exc:  # noqa: BLE001 - ImportError/ImportError-подобные от Qt
        _print_fatal("Не удалось импортировать PySide6", _pyside_import_hint(exc))
        return 1

    # Qt-сообщения -> наш лог (иначе в pythonw они теряются)
    def _qt_message_handler(mode, context, message):
        level = {0: 10, 1: 20, 2: 30, 3: 40}.get(int(mode), 20)
        location = f"{context.file}:{context.line}" if context and context.file else ""
        log.log(level, "Qt: %s %s", message, location)

    try:
        qInstallMessageHandler(_qt_message_handler)
    except Exception:  # noqa: BLE001
        pass

    # HighDPI и стабильный рендер
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "xcb"))

    from .ui import compat

    # атрибуты ставим через compat: их имена менялись между версиями PySide6
    for attribute in (compat.AA_SHARE_OPENGL_CONTEXTS, compat.AA_DONT_CREATE_NATIVE_WIDGET_SIBLINGS):
        if attribute is not None:
            try:
                QApplication.setAttribute(attribute, True)
            except (AttributeError, TypeError, RuntimeError) as exc:
                log.debug("Не удалось выставить атрибут Qt %s: %s", attribute, exc)
    try:
        QSurfaceFormat.setDefaultFormat(QSurfaceFormat())
    except Exception:  # noqa: BLE001
        pass

    app = QApplication.instance() or QApplication([APP_NAME, *argv])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_NAME)

    from .ui import theme
    from .ui.main_window import MainWindow, app_icon
    from .settings import Settings

    app.setStyleSheet(theme.stylesheet())
    app.setWindowIcon(app_icon())

    settings = Settings.load()

    window: Optional[MainWindow] = None
    try:
        window = MainWindow(settings=settings)
    except Exception as exc:  # noqa: BLE001
        log.critical("Не удалось создать главное окно:\n%s", traceback.format_exc())
        QMessageBox.critical(
            None,
            f"{APP_NAME} — ошибка интерфейса",
            f"Не удалось создать главное окно:\n\n{type(exc).__name__}: {exc}\n\n"
            f"Подробности: {paths.logs_dir() / 'localasr.log'}\n\n"
            "Что можно попробовать:\n"
            "  1. Сервис → «Проверить и починить окружение».\n"
            "  2. Удалить папку settings/ и запустить заново.\n"
            "  3. Обновить драйвер видеокарты (Qt использует её для отрисовки).\n"
            "  4. Запустить с переменной окружения QT_OPENGL=software.",
        )
        return 1

    window.show()

    def _global_exception(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.critical("Необработанное исключение в GUI:\n%s", text)
        try:
            QMessageBox.critical(window, "Сбой", f"Произошла ошибка:\n\n{text[-2500:]}")
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = _global_exception

    exit_code = app.exec()
    log.info("GUI закрыт, код %s", exit_code)
    return int(exit_code)


def _print_fatal(title: str, message: str) -> None:
    text = f"[{APP_NAME}] {title}\n{message}"
    print(text, file=sys.stderr, flush=True)
    log.critical(text)
    try:
        log_path = paths.logs_dir() / "localasr.log"
        print(f"\nЛог: {log_path}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
#  Команды CLI
# --------------------------------------------------------------------------- #


def cmd_list_models(args) -> int:
    from .core.models_catalog import ModelCatalog

    catalog = ModelCatalog()
    entries = catalog.installed() if args.installed else catalog.sorted_for_display()
    if args.json:
        print(catalog.export_json())
        return 0

    print(f"{APP_NAME} {APP_VERSION} · моделей в каталоге: {len(catalog.all())} · установлено: {len(catalog.installed())}\n")
    for entry in entries:
        mark = "✓" if entry.installed else " "
        star = "★" if entry.recommended else " "
        ru = "RU" if entry.supports_russian else "  "
        print(
            f"{mark}{star} {entry.id:<34} {entry.engine:<15} {ru} "
            f"{entry.size_text:>9}  RAM {entry.ram_mb:>5} МБ  Q{entry.quality}/S{entry.speed}  {entry.name}"
        )
    print("\nУстановка: python run_app.py transcribe <файл> -m <id>  (модель скачается при необходимости — или через GUI)")
    return 0


def cmd_doctor(_args) -> int:
    import platform

    from .asr.registry import engine_available, registry
    from .core.ffmpeg import ffmpeg_version, find_ffmpeg

    print(f"{APP_NAME} {APP_VERSION} · doctor")
    print("-" * 64)
    print(f"Проект      : {paths.ROOT_DIR}")
    print(f"Python      : {sys.executable}")
    print(f"Версия      : {sys.version.split()[0]} ({platform.platform()})")
    print(f"venv        : {'да' if _inside_venv() else 'НЕТ (запусти main.py)'}")
    print(f"models/     : {paths.models_dir()}")
    print(f"outputs/    : {paths.outputs_dir()}")
    print(f"ffmpeg      : {find_ffmpeg() or 'не найден (будет PyAV)'}")
    print(f"ffmpeg ver  : {ffmpeg_version()}")
    print()
    for engine in ("faster-whisper", "whispercpp", "vosk"):
        print(f"движок {engine:<16}: {'установлен' if engine_available(engine) else 'не установлен'}")
    print()
    try:
        hardware = registry.hardware()
        print(hardware.get("summary", "—"))
        for note in hardware.get("notes", []) or []:
            print("  ! " + note)
    except Exception as exc:  # noqa: BLE001
        print(f"Железо: не удалось определить ({exc})")

    print()
    from .core.models_catalog import ModelCatalog

    catalog = ModelCatalog()
    installed = catalog.installed()
    print(f"Моделей установлено: {len(installed)}")
    for entry in installed:
        print(f"  ✓ {entry.id} -> {entry.install_dir}")
    if not installed:
        print("  (нет — скачай модель через GUI или укажи -m в transcribe)")
    return 0


def cmd_version(_args) -> int:
    print(f"{APP_NAME} {APP_VERSION}")
    print(f"Python {sys.version.split()[0]} · {sys.executable}")
    print(f"Проект: {paths.ROOT_DIR}")
    return 0


def cmd_transcribe(args) -> int:
    from .asr.registry import registry
    from .asr.transcriber import JobOptions, transcribe_file
    from .core.downloader import install_model
    from .core.models_catalog import ModelCatalog
    from .settings import Settings

    settings = Settings.load()
    if args.format:
        settings.output_format = args.format
    if args.language is not None:
        settings.language = "" if args.language.lower() in ("auto", "") else args.language
    if args.beam is not None:
        settings.beam_size = args.beam
    if args.no_vad:
        settings.use_vad = False
    if args.no_clean:
        settings.suppress_hallucinations = False
    if args.timestamps:
        settings.add_timestamps_to_txt = True
    if args.words:
        settings.word_timestamps = True
    if args.device:
        settings.device = args.device
    if args.compute_type:
        settings.compute_type = args.compute_type
    if args.threads is not None:
        settings.cpu_threads = args.threads
    if args.batch_size is not None:
        settings.batch_size = args.batch_size
    if args.no_batched:
        settings.batched = False
    if args.prompt is not None:
        settings.initial_prompt = args.prompt
    if args.hotwords is not None:
        settings.hotwords = args.hotwords
    if args.overwrite:
        settings.overwrite_existing = True

    files = paths.expand_media_inputs(args.inputs)
    if not files:
        print(
            "Не найдено медиафайлов по указанным путям.\n"
            f"Поддерживаются расширения: {', '.join(sorted(MEDIA_EXTENSIONS)[:10])}…",
            file=sys.stderr,
        )
        return 2

    catalog = ModelCatalog()
    entry = _resolve_model(catalog, args.model)
    if entry is None:
        installed = catalog.installed()
        if not installed:
            print(
                "Ни одна модель не установлена.\n"
                "Запусти GUI (python main.py) и скачай модель, либо укажи -m <id>:\n"
                "  python run_app.py list-models",
                file=sys.stderr,
            )
            return 2
        entry = installed[0]
        print(f"Модель не указана — беру установленную: {entry.id}")

    if not entry.installed:
        print(f"Модель «{entry.id}» не установлена — скачиваю ({entry.size_text})…")
        try:
            install_model(
                entry,
                progress=lambda fraction, status: print(
                    f"\r  {int(fraction * 100):3d}%  {status[:70]:<70}", end="", flush=True
                ),
            )
            print("\r  готово" + " " * 70)
        except Exception as exc:  # noqa: BLE001
            print(f"\nНе удалось скачать модель: {exc}", file=sys.stderr)
            return 3
        catalog.refresh_installed()
        entry = catalog.get(entry.id) or entry

    output_dir = Path(args.output).expanduser() if args.output else settings.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = list(OUTPUT_FORMATS) if args.all_formats else [settings.output_format]
    options = JobOptions.from_settings(entry, settings, output_dir)
    options.formats = formats
    options.write_header = bool(args.header)

    print(f"Файлов: {len(files)} · модель: {entry.name} · движок: {entry.engine}")
    print(f"Результаты: {output_dir} ({', '.join(formats)})\n")

    ok = fail = 0
    started_all = time.time()
    for index, source in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {source.name}")
        started = time.time()

        def _progress(fraction: float, stage: str, _started=started) -> None:
            elapsed = time.time() - _started
            print(f"\r  {int(fraction * 100):3d}%  {stage[:60]:<60} {elapsed:6.1f} с", end="", flush=True)

        try:
            result = transcribe_file(source, options, progress=_progress)
        except KeyboardInterrupt:
            print("\nПрервано пользователем.")
            return 130
        except Exception as exc:  # noqa: BLE001
            print(f"\n  ✗ ошибка: {type(exc).__name__}: {exc}")
            fail += 1
            continue

        print()
        if result.ok:
            ok += 1
            for path in result.outputs:
                print(f"  ✓ {path}")
            print(
                f"  {result.words} слов · {result.segments} сегментов · язык {result.language} · "
                f"{result.speed_x:.1f}× · {result.elapsed_seconds:.0f} с"
                + (f" · отброшено {result.dropped} мусорных сегментов" if result.dropped else "")
            )
        else:
            fail += 1
            print(f"  ✗ {result.error}")
        for warning in result.warnings:
            print(f"  ! {warning}")

    total_time = time.time() - started_all
    print(f"\nИтого: {ok} готово, {fail} с ошибкой, {total_time:.0f} с")
    registry.unload()
    return 0 if fail == 0 else 1


def _resolve_model(catalog, wanted: str):
    if not wanted:
        return None
    wanted = wanted.strip()
    entry = catalog.get(wanted)
    if entry is not None:
        return entry
    low = wanted.lower()
    for candidate in catalog.all():
        if candidate.id.lower() == low or candidate.name.lower() == low:
            return candidate
    for candidate in catalog.all():
        if low in candidate.id.lower() or low in candidate.name.lower():
            return candidate
    return None


def _inside_venv() -> bool:
    """Мы внутри виртуального окружения (любого — проверяем sys.prefix)."""
    if getattr(sys, "real_prefix", None):
        return True
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True
    exe = Path(sys.executable).resolve()
    try:
        exe.relative_to(paths.venv_dir().resolve())
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
#  Точка входа
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    setup_logging()

    # argparse не знает про наши «глобальные» флаги до разбора
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    command = getattr(args, "command", None)
    try:
        if command == "transcribe":
            return cmd_transcribe(args)
        if command in ("list-models", "models"):
            return cmd_list_models(args)
        if command == "doctor":
            return cmd_doctor(args)
        if command == "version":
            return cmd_version(args)
        if getattr(args, "no_gui", False):
            print("GUI отключён флагом --no-gui. Используй подкоманду (transcribe/list-models/doctor).")
            return 0
        return run_gui(argv)
    except KeyboardInterrupt:
        print("\nПрервано.")
        return 130
    except Exception as exc:  # noqa: BLE001
        log.critical("Критическая ошибка:\n%s", traceback.format_exc())
        _print_fatal("Критическая ошибка", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
