"""Application entry point: GUI by default + headless mode (CLI)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

from . import i18n, paths
from .constants import APP_NAME, APP_VERSION, MEDIA_EXTENSIONS, OUTPUT_FORMATS
from .i18n import _
from .logutils import get_logger, setup_logging

log = get_logger("cli")


# --------------------------------------------------------------------------- #
#  Arguments
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="LocalASR",
        description=_("Local speech recognition: audio/video to text (no cloud)."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_(
            "Examples:\n"
            "  python launcher/run_app.py                             # start the GUI\n"
            "  python launcher/run_app.py list-models                 # what is available\n"
            "  python launcher/run_app.py transcribe \"lecture 1.mp4\"  # recognize without GUI\n"
            "  python launcher/run_app.py transcribe ./lectures -m whisper-large-v3-turbo -f srt\n"
            "  python launcher/run_app.py doctor                      # environment diagnostics\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    transcribe = sub.add_parser("transcribe", help=_("recognize files without GUI"))
    transcribe.add_argument("inputs", nargs="+", help=_("files and/or folders"))
    transcribe.add_argument("-m", "--model", default="", help=_("model ID or name from the catalog"))
    transcribe.add_argument("-o", "--output", default="", help=_("result folder (default: outputs/)"))
    transcribe.add_argument("-f", "--format", default="", choices=list(OUTPUT_FORMATS), help=_("result format"))
    transcribe.add_argument("-l", "--language", default=None, help=_("language code (ru, en, …) or 'auto'"))
    transcribe.add_argument("--beam", type=int, default=None, help=_("beam size (1–12)"))
    transcribe.add_argument("--no-vad", action="store_true", help=_("do not cut silence"))
    transcribe.add_argument("--no-clean", action="store_true", help=_("do not clean hallucinations"))
    transcribe.add_argument("--timestamps", action="store_true", help=_("timecodes in txt"))
    transcribe.add_argument("--words", action="store_true", help=_("per-word timestamps (for subtitles)"))
    transcribe.add_argument("--device", default=None, choices=[None, "auto", "cpu", "cuda"], help=_("device"))
    transcribe.add_argument("--compute-type", default=None, help=_("int8 / float16 / int8_float16 / default"))
    transcribe.add_argument("--threads", type=int, default=None, help=_("CPU threads"))
    transcribe.add_argument("--batch-size", type=int, default=None, help=_("batch for the batched mode"))
    transcribe.add_argument("--no-batched", action="store_true", help=_("disable the batched mode"))
    transcribe.add_argument("--prompt", default=None, help=_("initial_prompt (style/terms)"))
    transcribe.add_argument("--hotwords", default=None, help=_("comma-separated hint words"))
    transcribe.add_argument("--overwrite", action="store_true", help=_("overwrite existing files"))
    transcribe.add_argument("--header", action="store_true", help=_("add a metadata header"))
    transcribe.add_argument("--all-formats", action="store_true", help=_("save txt+srt+vtt+json at once"))

    models = sub.add_parser("list-models", aliases=["models"], help=_("list models"))
    models.add_argument("--json", action="store_true", help=_("output as JSON"))
    models.add_argument("--installed", action="store_true", help=_("installed only"))

    sub.add_parser("doctor", help=_("diagnostics: Python, env, ffmpeg, engines, hardware"))
    sub.add_parser("version", help=_("application version"))

    parser.add_argument("--no-gui", action="store_true", help=_("do not open the window (debugging)"))
    return parser


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #


def _pyside_import_hint(exc: BaseException) -> str:
    """Hint tailored to the concrete PySide6 import failure."""
    text = str(exc).lower()
    base = f"{type(exc).__name__}: {exc}\n\n"

    if "cannot open shared object" in text or ".so" in text or "dll" in text or "libglib" in text:
        return base + _(
            "PySide6 is installed but SYSTEM Qt libraries are missing.\n\n"
            "Linux (Debian/Ubuntu):\n"
            "  sudo apt install libglib2.0-0 libgl1 libegl1 libxkbcommon0 libdbus-1-3 \\\n"
            "      libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 \\\n"
            "      libxcb-xinerama0 libxcb-randr0 libxcb-render-util0 libfontconfig1\n\n"
            "Linux (Fedora):  sudo dnf install glib2 mesa-libEGL libxkbcommon xcb-util-*\n"
            "macOS:           usually not needed; if broken, reinstall PySide6\n"
            "                 (delete the environment folder and run launcher/main.py again).\n\n"
            "Temporary workaround (software rendering):\n"
            "  QT_OPENGL=software QT_QPA_PLATFORM=xcb python launcher/main.py\n"
        )

    if "no module named" in text:
        return base + _(
            "PySide6 is not installed in the project environment.\n"
            "Delete the environment folder and run launcher/main.py again — it will be recreated,\n"
            "or use Tools → Check and repair the environment.\n"
        )

    return base + _(
        "Failed to load PySide6.\n"
        "1) Delete the environment folder and run launcher/main.py again.\n"
        "2) Update the GPU driver (Qt uses it for rendering).\n"
        "3) Try QT_OPENGL=software python launcher/main.py\n"
    )


def _install_qt_translations(app) -> None:
    """Localize Qt's own strings (standard buttons, menus) for the UI language."""
    try:
        from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator

        path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        for name in i18n.qt_translator_names():
            translator = QTranslator(app)
            if translator.load(name, path) or translator.load(name):
                app.installTranslator(translator)
        QLocale.setDefault(QLocale(i18n.current_language()))
    except Exception as exc:  # noqa: BLE001
        log.debug("Qt translations not installed: %s", exc)


def run_gui(argv: Sequence[str]) -> int:
    """Start the Qt application. All PySide6 imports are inside the function."""
    # Windows: without an explicit AppUserModelID the taskbar groups the window
    # under "python.exe" and shows the Python icon instead of ours.
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "LocalASR.App.1.0"
            )
        except Exception:  # noqa: BLE001
            pass

    # Third-party libraries (e.g. modeling_gigaam.py) run ffmpeg via subprocess
    # without CREATE_NO_WINDOW — from pythonw that means flashing consoles.
    try:
        from .core.winutil import suppress_child_consoles

        suppress_child_consoles()
    except Exception:  # noqa: BLE001
        pass

    try:
        from PySide6.QtCore import qInstallMessageHandler
        from PySide6.QtGui import QSurfaceFormat
        from PySide6.QtWidgets import QApplication, QMessageBox
    except BaseException as exc:  # noqa: BLE001
        _print_fatal(_("Failed to import PySide6"), _pyside_import_hint(exc))
        return 1

    # Qt messages -> our log (otherwise pythonw loses them)
    def _qt_message_handler(mode, context, message):
        # harmless Qt warning about px->pt font conversion in stylesheets
        # (QTBUG-102278): showing it only annoys the user
        if "point size <= 0" in message.lower():
            return
        level = {0: 10, 1: 20, 2: 30, 3: 40}.get(int(mode), 20)
        location = f"{context.file}:{context.line}" if context and context.file else ""
        log.log(level, "Qt: %s %s", message, location)

    try:
        qInstallMessageHandler(_qt_message_handler)
    except Exception:  # noqa: BLE001
        pass

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "xcb"))

    from .ui import compat

    # attributes are set via compat: their names changed across PySide6 versions
    for attribute in (compat.AA_SHARE_OPENGL_CONTEXTS, compat.AA_DONT_CREATE_NATIVE_WIDGET_SIBLINGS):
        if attribute is not None:
            try:
                QApplication.setAttribute(attribute, True)
            except (AttributeError, TypeError, RuntimeError) as exc:
                log.debug("Failed to set Qt attribute %s: %s", attribute, exc)
    try:
        QSurfaceFormat.setDefaultFormat(QSurfaceFormat())
    except Exception:  # noqa: BLE001
        pass

    app = QApplication.instance() or QApplication([APP_NAME, *argv])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_NAME)

    from .settings import Settings

    settings = Settings.load()
    i18n.init(settings.ui_language)
    _install_qt_translations(app)

    # the mouse wheel over combo/spin no longer changes values — it scrolls
    # the page instead (accidental value changes while scrolling were annoying)
    try:
        from .ui.wheelguard import install_wheel_guard

        install_wheel_guard(app)
    except Exception as exc:  # noqa: BLE001
        log.debug("Wheel guard not installed: %s", exc)

    from .ui import theme
    from .ui.main_window import MainWindow, app_icon

    app.setStyleSheet(theme.stylesheet())
    app.setWindowIcon(app_icon())

    window: Optional[MainWindow] = None
    try:
        window = MainWindow(settings=settings)
    except Exception as exc:  # noqa: BLE001
        log.critical(_("Failed to create the main window:\n%s"), traceback.format_exc())
        QMessageBox.critical(
            None,
            _("{APP_NAME} — interface error").format(APP_NAME=APP_NAME),
            _("Failed to create the main window:\n\n{exc}\n\nLogs folder: {logs}\n"
              "(the verbose log file is enabled in Settings → \"Logging & diagnostics\" "
              "or via LOCALASR_LOG=1)\n\n"
              "Things to try:\n"
              "  1. Tools → \"Check and repair the environment\".\n"
              "  2. Delete the settings/ folder and start again.\n"
              "  3. Update the GPU driver (Qt uses it for rendering).\n"
              "  4. Run with the QT_OPENGL=software environment variable."
              ).format(exc=f"{type(exc).__name__}: {exc}", logs=paths.logs_dir()),
        )
        return 1

    # Open maximized by default: the interface is dense. Saved geometry wins.
    if settings.window_geometry:
        window.show()
    else:
        window.showMaximized()

    # The app started fine — the bootstrap install log is no longer needed.
    cleanup_bootstrap_log()

    def _global_exception(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.critical(_("Unhandled exception in GUI:\n%s"), text)
        try:
            QMessageBox.critical(window, _("Crash"), _("An error occurred:\n\n{text}").format(text=text[-2500:]))
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = _global_exception

    exit_code = app.exec()
    log.info(_("GUI closed, code %s"), exit_code)
    return int(exit_code)


def _print_fatal(title: str, message: str) -> None:
    text = f"[{APP_NAME}] {title}\n{message}"
    print(text, file=sys.stderr, flush=True)
    log.critical(text)
    try:
        from . import logutils

        current = logutils.current_log_file()
        log_path = current or paths.logs_dir()
        print(_("Log: {}").format(log_path), file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
#  CLI commands
# --------------------------------------------------------------------------- #


def cmd_list_models(args) -> int:
    from .core.models_catalog import ModelCatalog

    catalog = ModelCatalog()
    entries = catalog.installed() if args.installed else catalog.sorted_for_display()
    if args.json:
        print(catalog.export_json())
        return 0

    print(
        _("{app} {version} · models in catalog: {total} · installed: {installed}\n").format(
            app=APP_NAME, version=APP_VERSION,
            total=len(catalog.all()), installed=len(catalog.installed()),
        )
    )
    for entry in entries:
        mark = "✓" if entry.installed else " "
        ru = "RU" if entry.supports_russian else "  "
        year = str(entry.released_year) if entry.released_year else "  ?"
        quant = entry.quantization or "?"
        params = entry.params_text
        print(
            f"{mark} {entry.id:<34} {entry.engine:<15} {ru} "
            f"{entry.size_text:>9}  RAM {entry.ram_mb:>5} {_('MB')}  "
            f"{year}  {params:>10}  {quant:<13}  Q{entry.quality}/S{entry.speed}  {entry.display_name}"
        )
    print(_("\nInstall: python launcher/run_app.py transcribe <file> -m <id>  "
            "(the model is downloaded on demand — or use the GUI)"))
    return 0


def cmd_doctor(_args) -> int:
    import platform

    from .asr.registry import engine_available, registry
    from .core.ffmpeg import ffmpeg_version, find_ffmpeg

    print(f"{APP_NAME} {APP_VERSION} · doctor")
    print("-" * 64)
    print(_("Project     : {}").format(paths.ROOT_DIR))
    print(f"Python      : {sys.executable}")
    print(_("Version     : {ver} ({plat})").format(ver=sys.version.split()[0], plat=platform.platform()))
    print(_("Environment : {}").format(
        _("project env") if _inside_project_env() else _("NOT FOUND (run launcher/main.py)")))
    try:
        from .core import integrity

        missing_files = integrity.missing_project_files()
        print(
            _("App files   : ")
            + (_("all present") if not missing_files
               else _("MISSING: {}").format(", ".join(missing_files[:8])))
        )
    except Exception as exc:  # noqa: BLE001
        print(_("App files   : check failed ({})").format(exc))
    print(f"models/     : {paths.models_dir()}")
    print(f"outputs/    : {paths.outputs_dir()}")
    print(_("ffmpeg      : {}").format(find_ffmpeg() or _("not found (PyAV will be used)")))
    print(f"ffmpeg ver  : {ffmpeg_version()}")
    print()
    from .constants import ENGINES

    for engine in ENGINES:
        print(_("engine {name:<18}: {state}").format(
            name=engine,
            state=_("installed") if engine_available(engine) else _("not installed"),
        ))
    print()
    try:
        hardware = registry.hardware()
        print(hardware.get("summary", "—"))
        for note in hardware.get("notes", []) or []:
            print("  ! " + note)
    except Exception as exc:  # noqa: BLE001
        print(_("Hardware: detection failed ({})").format(exc))

    print()
    from .core.models_catalog import ModelCatalog

    catalog = ModelCatalog()
    installed = catalog.installed()
    print(_("Models installed: {}").format(len(installed)))
    for entry in installed:
        print(f"  ✓ {entry.id} -> {entry.install_dir}")
    if not installed:
        print(_("  (none — download a model via the GUI or pass -m to transcribe)"))
    return 0


def cmd_version(_args) -> int:
    print(f"{APP_NAME} {APP_VERSION}")
    print(f"Python {sys.version.split()[0]} · {sys.executable}")
    print(_("Project: {}").format(paths.ROOT_DIR))
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
            _("No media files found at the given paths.\nSupported extensions: {ext}…").format(
                ext=", ".join(sorted(MEDIA_EXTENSIONS)[:10])
            ),
            file=sys.stderr,
        )
        return 2

    catalog = ModelCatalog()
    entry = _resolve_model(catalog, args.model)
    if entry is None:
        installed = catalog.installed()
        if not installed:
            print(
                _("No model is installed.\n"
                  "Start the GUI (python launcher/main.py) and download a model, or pass -m <id>:\n"
                  "  python launcher/run_app.py list-models"),
                file=sys.stderr,
            )
            return 2
        entry = installed[0]
        print(_("No model specified — using the installed one: {}").format(entry.id))

    if not entry.installed:
        print(_("Model \"{}\" is not installed — downloading ({})…").format(entry.id, entry.size_text))
        try:
            install_model(
                entry,
                progress=lambda fraction, status: print(
                    f"\r  {int(fraction * 100):3d}%  {status[:70]:<70}", end="", flush=True
                ),
            )
            print("\r  " + _("done") + " " * 70)
        except Exception as exc:  # noqa: BLE001
            print(_("Failed to download the model: {}").format(exc), file=sys.stderr)
            return 3
        catalog.refresh_installed()
        entry = catalog.get(entry.id) or entry

    output_dir = Path(args.output).expanduser() if args.output else settings.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = list(OUTPUT_FORMATS) if args.all_formats else [settings.output_format]
    options = JobOptions.from_settings(entry, settings, output_dir)
    options.formats = formats
    options.write_header = bool(args.header)

    print(_("Files: {n} · model: {model} · engine: {engine}").format(
        n=len(files), model=entry.display_name, engine=entry.engine))
    print(_("Results: {dir} ({fmt})\n").format(dir=output_dir, fmt=", ".join(formats)))

    ok = fail = 0
    started_all = time.time()
    for index, source in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {source.name}")
        started = time.time()

        def _progress(fraction: float, stage: str, _started=started) -> None:
            elapsed = time.time() - _started
            print(
                f"\r  {int(fraction * 100):3d}%  {stage[:60]:<60} {elapsed:6.1f} "
                + _("s"), end="", flush=True,
            )

        try:
            result = transcribe_file(source, options, progress=_progress)
        except KeyboardInterrupt:
            print()
            print(_("Interrupted by user."))
            return 130
        except Exception as exc:  # noqa: BLE001
            print()
            print(_("  ✗ error: {type}: {msg}").format(type=type(exc).__name__, msg=exc))
            fail += 1
            continue

        print()
        if result.ok:
            ok += 1
            for path in result.outputs:
                print(f"  ✓ {path}")
            print(
                _("  {words} words · {segments} segments · language {lang} · "
                  "{speed:.1f}× · {elapsed:.0f} s").format(
                    words=result.words, segments=result.segments, lang=result.language,
                    speed=result.speed_x, elapsed=result.elapsed_seconds,
                )
                + (_(" · dropped {} junk segments").format(result.dropped) if result.dropped else "")
            )
        else:
            fail += 1
            print(f"  ✗ {result.error}")
        for warning in result.warnings:
            print(f"  ! {warning}")

    total_time = time.time() - started_all
    print()
    print(_("Total: {ok} done, {fail} failed, {time:.0f} s").format(ok=ok, fail=fail, time=total_time))
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


def _inside_project_env() -> bool:
    """Running inside the project environment (.venv or portable runtime)."""
    if getattr(sys, "real_prefix", None):
        return True
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True
    exe = Path(sys.executable).resolve()
    try:
        exe.relative_to(paths.env_dir().resolve())
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    i18n.init()
    setup_logging()
    _configure_file_logging()

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
            print(_("GUI is disabled by --no-gui. Use a subcommand (transcribe/list-models/doctor)."))
            return 0
        return run_gui(argv)
    except KeyboardInterrupt:
        print(_("Interrupted."))
        return 130
    except Exception as exc:  # noqa: BLE001
        log.critical(_("Critical error:\n%s"), traceback.format_exc())
        _print_fatal(_("Critical error"), f"{type(exc).__name__}: {exc}")
        return 1


def _configure_file_logging() -> None:
    """File log per settings (off by default).

    Priority: LOCALASR_LOG=1/0 -> settings.file_logging -> off.
    """
    from . import logutils

    enabled = False
    try:
        from .settings import Settings

        enabled = bool(Settings.load().file_logging)
    except Exception as exc:  # noqa: BLE001
        log.debug("Failed to read the logging setting: %s", exc)
    logutils.configure_file_logging(enabled)


def cleanup_bootstrap_log() -> bool:
    """Delete logs/bootstrap.log after a SUCCESSFUL start.

    The install log is only needed until the app is up: on error the code
    never gets here and the file stays for diagnostics.
    """
    try:
        bootstrap_log = paths.logs_dir() / "bootstrap.log"
        if bootstrap_log.is_file():
            bootstrap_log.unlink()
            log.debug("bootstrap.log removed: the app started successfully")
            return True
    except OSError as exc:
        log.debug("Failed to remove bootstrap.log: %s", exc)
    return False


if __name__ == "__main__":
    sys.exit(main())
