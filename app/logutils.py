"""Логирование: файл + консоль + «живая» лента в GUI.

Важно: модуль не импортирует PySide6, поэтому его можно использовать и в CLI,
и в воркерах. GUI подписывается на записи через `attach_qt_sink`.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from . import paths

LOGGER_NAME = "localasr"
_configured = False


def setup_logging(level: int = logging.INFO, root: Optional[Path] = None) -> logging.Logger:
    """Настраивает корневой логгер приложения (идемпотентно)."""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if _configured:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- файл (с ротацией: 3 x 2 МБ, чтобы не разрастался) -----------------
    try:
        log_dir = paths.logs_dir(root)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "localasr.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except OSError as exc:  # pragma: no cover
        logger.warning("Не удалось создать файловый лог: %s", exc)

    # --- консоль, если она есть (pythonw -> sys.stdout is None) ------------
    stream = sys.stderr if sys.stderr is not None else None
    if stream is not None and os.environ.get("LOCALASR_QUIET") != "1":
        try:
            console = logging.StreamHandler(stream)
            console.setFormatter(fmt)
            console.setLevel(logging.INFO)
            logger.addHandler(console)
        except (OSError, ValueError):
            pass

    # --- перехват «чужих» библиотек (faster-whisper любит кричать) ---------
    for noisy in ("faster_whisper", "ctranslate2", "pywhispercpp", "vosk", "urllib3", "av"):
        lib_logger = logging.getLogger(noisy)
        lib_logger.handlers.clear()
        lib_logger.propagate = True
        lib_logger.setLevel(max(level, logging.WARNING))

    # критические исключения пишем в лог до того, как приложение упадёт
    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("Необработанное исключение", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    try:
        sys.excepthook = _excepthook
    except (AttributeError, TypeError):
        pass

    _configured = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not _configured:
        setup_logging()
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


class QtLogBridge(logging.Handler):
    """Пересылает записи лога в Qt-сигнал (безопасно для потоков)."""

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
            # виджет уже удалён — отцепляемся
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
