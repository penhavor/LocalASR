"""Logging: console + live GUI feed + optional verbose per-launch file.

The file log is OFF by default (keeps logs/ clean). It is enabled via:
  * Settings -> Logging & diagnostics (applies immediately, no restart);
  * the LOCALASR_LOG=1 environment variable (for developers/tests);
  * file_logging=true in settings.json.

When enabled, each launch creates a NEW logs/localasr_<date_time>.log with
DEBUG-level detail (startup options, hardware, network, stages, timings,
full tracebacks). Old files are pruned (LOG_KEEP_FILES kept).

The module does not import PySide6, so CLI and workers can use it too;
the GUI subscribes via attach_qt_sink.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import paths
from .i18n import _

LOGGER_NAME = "localasr"
LOG_KEEP_FILES = 30
_configured = False

# --- file log state (toggled at runtime) -------------------------------------- #
_file_handler: Optional[logging.FileHandler] = None
_file_path: Optional[Path] = None
_file_lock = threading.RLock()

#: third-party loggers silenced down to WARNING
_NOISY_LIBS = (
    "faster_whisper", "ctranslate2", "pywhispercpp", "vosk", "urllib3", "av",
    "transformers", "torch", "torchaudio", "huggingface_hub", "filelock",
    "requests", "PIL", "matplotlib", "asyncio", "numba", "onnxruntime",
)


def _console_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _file_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | [%(threadName)s] | "
        "%(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logging(level: int = logging.INFO, root: Optional[Path] = None) -> logging.Logger:
    """Configure the application root logger (idempotent).

    The file log is NOT created here; see configure_file_logging().
    """
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)      # the file wants DEBUG; handlers filter
    logger.propagate = False

    if _configured:
        return logger

    # console, if any (pythonw -> sys.stdout is None)
    stream = sys.stderr if sys.stderr is not None else None
    if stream is not None and os.environ.get("LOCALASR_QUIET") != "1":
        try:
            console = logging.StreamHandler(stream)
            console.setFormatter(_console_formatter())
            console.setLevel(level)
            logger.addHandler(console)
        except (OSError, ValueError):
            pass

    # silence chatty third-party libraries (warnings still reach the file log)
    logging.getLogger().setLevel(logging.WARNING)
    for noisy in _NOISY_LIBS:
        lib_logger = logging.getLogger(noisy)
        lib_logger.handlers.clear()
        lib_logger.propagate = True
        lib_logger.setLevel(logging.WARNING)

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical(_("Unhandled exception"), exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    try:
        sys.excepthook = _excepthook
    except (AttributeError, TypeError):
        pass

    try:
        def _thread_excepthook(args):
            if issubclass(args.exc_type, SystemExit):
                return
            logger.critical(
                _("Unhandled exception in thread %s"), args.thread.name if args.thread else "?",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = _thread_excepthook
    except AttributeError:
        pass

    _configured = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not _configured:
        setup_logging()
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


# --------------------------------------------------------------------------- #
#  File log: toggle at runtime
# --------------------------------------------------------------------------- #


def _session_header() -> List[str]:
    """Startup header: everything needed to reproduce a bug."""
    from .constants import APP_NAME, APP_VERSION

    lines = [
        "=" * 88,
        f"{APP_NAME} {APP_VERSION} — " + _("logging session started"),
        f"  {_('time')}      : {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"  python     : {sys.version.splitlines()[0]} ({platform.python_implementation()})",
        f"  {_('platform')}  : {platform.platform()}",
        f"  {_('process')}    : PID {os.getpid()}",
        f"  {_('executable')}: {sys.executable}",
        f"  cwd        : {os.getcwd()}",
        f"  argv       : {' '.join(sys.argv) or '—'}",
        f"  {_('project')}     : {paths.ROOT_DIR}",
        "=" * 88,
    ]
    return lines


def current_log_file() -> Optional[Path]:
    """Path of the active file log (None when file logging is off)."""
    with _file_lock:
        return _file_path


def file_logging_enabled() -> bool:
    with _file_lock:
        return _file_handler is not None


def _prune_old_logs(log_dir: Path) -> int:
    """Keep the LOG_KEEP_FILES newest log files, delete older ones."""
    try:
        logs = sorted(
            log_dir.glob("localasr_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return 0
    removed = 0
    for old in logs[LOG_KEEP_FILES:]:
        try:
            old.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def enable_file_logging(root: Optional[Path] = None, context: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """Create a NEW log file (named by enable time) and attach it.

    Safe to call at any moment; idempotent while enabled.
    Returns the file path or None on failure.
    """
    global _file_handler, _file_path
    with _file_lock:
        if _file_handler is not None:
            return _file_path
        try:
            log_dir = paths.logs_dir(root)
            stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            path = log_dir / f"localasr_{stamp}.log"
            # two launches in one second (scripts/tests) get separate files
            counter = 2
            while path.exists():
                path = log_dir / f"localasr_{stamp}_{counter}.log"
                counter += 1
            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        except OSError as exc:
            get_logger().warning(_("Failed to create the file log: %s"), exc)
            return None

        handler.setFormatter(_file_formatter())
        handler.setLevel(logging.DEBUG)

        logger = logging.getLogger(LOGGER_NAME)
        logger.addHandler(handler)
        # third-party warnings go to the file too (root logger)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        _file_handler = handler
        _file_path = path

        try:
            for line in _session_header():
                logger.debug(line)
            for key, value in (context or {}).items():
                logger.debug(_("context: %s = %s"), key, value)
        except Exception:  # noqa: BLE001
            pass

        removed = _prune_old_logs(log_dir)
        logger.info(
            _("File log enabled: %s%s"), path,
            _(" (removed old: {})").format(removed) if removed else "",
        )
        return path


def disable_file_logging(final_note: bool = True) -> None:
    """Detach the file log (the file itself stays on disk)."""
    global _file_handler, _file_path
    with _file_lock:
        handler, _file_handler = _file_handler, None
        path, _file_path = _file_path, None
        if handler is None:
            return
        try:
            if final_note:
                logging.getLogger(LOGGER_NAME).info(_("File log disabled (%s)"), path)
            logging.getLogger(LOGGER_NAME).removeHandler(handler)
            logging.getLogger().removeHandler(handler)
            handler.flush()
            handler.close()
        except Exception:  # noqa: BLE001
            pass


def set_file_logging(enabled: bool, root: Optional[Path] = None) -> Optional[Path]:
    """Toggle the file log at runtime. Returns its path (or None)."""
    if enabled:
        return enable_file_logging(root)
    disable_file_logging()
    return None


def configure_file_logging(
    enabled: Optional[bool] = None,
    root: Optional[Path] = None,
    context: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """Apply the logging policy at startup.

    Priority: LOCALASR_LOG (0/1) -> settings.file_logging -> False.
    """
    env = os.environ.get("LOCALASR_LOG", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        enabled = True
    elif env in ("0", "false", "no", "off"):
        enabled = False
    if enabled is None:
        enabled = False
    if enabled:
        return enable_file_logging(root, context)
    return None


def list_log_files(root: Optional[Path] = None) -> List[Path]:
    """All application log files in logs/ (newest first)."""
    try:
        log_dir = paths.logs_dir(root)
        return sorted(log_dir.glob("localasr_*.log"), key=lambda p: p.name, reverse=True)
    except OSError:
        return []


atexit.register(disable_file_logging, final_note=False)


# --------------------------------------------------------------------------- #
#  GUI bridge
# --------------------------------------------------------------------------- #


class QtLogBridge(logging.Handler):
    """Forwards log records to a Qt signal (thread-safe)."""

    def __init__(self, emit_callable: Callable[[str, str], None], level: int = logging.INFO) -> None:
        super().__init__(level)
        self._emit = emit_callable
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001
            return
        try:
            self._emit(record.levelname, message)
        except RuntimeError:
            # the widget is gone — detach
            try:
                logging.getLogger(LOGGER_NAME).removeHandler(self)
            except Exception:  # noqa: BLE001
                pass


def attach_qt_sink(emit_callable: Callable[[str, str], None]) -> QtLogBridge:
    handler = QtLogBridge(emit_callable)
    get_logger().addHandler(handler)
    return handler


def detach_qt_sink(handler: Optional[logging.Handler]) -> None:
    if handler is not None:
        try:
            get_logger().removeHandler(handler)
        except Exception:  # noqa: BLE001
            pass
