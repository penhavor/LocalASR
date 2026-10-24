"""Логирование: консоль + «живая» лента в GUI + подробный файл на каждый запуск.

Файловый лог ПО УМОЛЧАНИЮ ВЫКЛЮЧЕН (чтобы не засорять logs/). Включается:
  * галочкой «Сервис → Настройки → Журнал и диагностика» (сразу, без перезапуска);
  * переменной окружения LOCALASR_LOG=1 (для разработчика/тестов);
  * полем file_logging=true в settings.json.

Когда включён — каждый запуск (каждое включение) создаёт НОВЫЙ файл
`logs/localasr_ГГГГ-ММ-ДД_ЧЧ-ММ-СС.log` c исчерпывающей информацией уровня
DEBUG: параметры запуска, железо, все сетевые запросы, стадии обработки,
тайминги, полные стектрейсы. Старые логи автоматически подчищаются
(хранится не более LOG_KEEP_FILES штук).

Важно: модуль не импортирует PySide6, поэтому его можно использовать и в CLI,
и в воркерах. GUI подписывается на записи через `attach_qt_sink`.
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

LOGGER_NAME = "localasr"
LOG_KEEP_FILES = 30            # сколько последних файл-логов хранить
_configured = False

# --- состояние файлового лога (включается/выключается на лету) ------------- #
_file_handler: Optional[logging.FileHandler] = None
_file_path: Optional[Path] = None
_file_lock = threading.RLock()

#: библиотеки, чью болтовню мы приглушаем до WARNING
_NOISY_LIBS = (
    "faster_whisper", "ctranslate2", "pywhispercpp", "vosk", "urllib3", "av",
    "transformers", "torch", "torchaudio", "huggingface_hub", "filelock",
    "requests", "PIL", "matplotlib", "asyncio", "numba", "onnxruntime",
)


# --------------------------------------------------------------------------- #
#  Форматы
# --------------------------------------------------------------------------- #

def _console_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _file_formatter() -> logging.Formatter:
    # полный формат для файла: дата+время, поток и место вызова — чтобы любой
    # баг можно было воспроизвести по логу без догадок
    return logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | [%(threadName)s] | "
        "%(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --------------------------------------------------------------------------- #
#  Базовая настройка (консоль + перехват исключений)
# --------------------------------------------------------------------------- #


def setup_logging(level: int = logging.INFO, root: Optional[Path] = None) -> logging.Logger:
    """Настраивает корневой логгер приложения (идемпотентно).

    Файловый лог здесь НЕ создаётся: он включается отдельно
    (`configure_file_logging`) — по умолчанию выключен.
    """
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)      # файл хочет DEBUG; хендлеры сами фильтруют
    logger.propagate = False

    if _configured:
        return logger

    # --- консоль, если она есть (pythonw -> sys.stdout is None) ------------
    stream = sys.stderr if sys.stderr is not None else None
    if stream is not None and os.environ.get("LOCALASR_QUIET") != "1":
        try:
            console = logging.StreamHandler(stream)
            console.setFormatter(_console_formatter())
            console.setLevel(level)
            logger.addHandler(console)
        except (OSError, ValueError):
            pass

    # --- перехват «чужих» библиотек (faster-whisper любит кричать) ---------
    # Уровень WARNING: их сообщения не spam-ят консоль/журнал GUI, но когда
    # файловый лог включён, он ставится и на root-логгер — предупреждения
    # библиотек тоже попадут в файл (см. enable_file_logging).
    logging.getLogger().setLevel(logging.WARNING)
    for noisy in _NOISY_LIBS:
        lib_logger = logging.getLogger(noisy)
        lib_logger.handlers.clear()
        lib_logger.propagate = True
        lib_logger.setLevel(logging.WARNING)

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

    # потоки тоже могут падать молча — перехватываем и их
    try:
        def _thread_excepthook(args):
            if issubclass(args.exc_type, SystemExit):
                return
            logger.critical(
                "Необработанное исключение в потоке %s", args.thread.name if args.thread else "?",
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
#  Файловый лог: включение/выключение на лету
# --------------------------------------------------------------------------- #


def _session_header() -> List[str]:
    """Шапка запуска: всё, что нужно знать для воспроизведения бага."""
    from .constants import APP_NAME, APP_VERSION  # локальный импорт: constants лёгок

    lines = [
        "=" * 88,
        f"{APP_NAME} {APP_VERSION} — сессия логирования начата",
        f"  время      : {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"  python     : {sys.version.splitlines()[0]} ({platform.python_implementation()})",
        f"  платформа  : {platform.platform()}",
        f"  процесс    : PID {os.getpid()}",
        f"  исполняемый: {sys.executable}",
        f"  cwd        : {os.getcwd()}",
        f"  argv       : {' '.join(sys.argv) or '—'}",
        f"  проект     : {paths.ROOT_DIR}",
        "=" * 88,
    ]
    return lines


def current_log_file() -> Optional[Path]:
    """Путь к активному файл-логу (None — файловое логирование выключено)."""
    with _file_lock:
        return _file_path


def file_logging_enabled() -> bool:
    with _file_lock:
        return _file_handler is not None


def _prune_old_logs(log_dir: Path) -> int:
    """Оставляет LOG_KEEP_FILES свежайших лог-файлов, старые удаляет."""
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
    """Создаёт НОВЫЙ файл лога (имя = дата/время включения) и подключает его.

    Можно вызывать в любой момент работы программы — перезапуск не нужен.
    Повторный вызов без disable не создаёт второй файл (идемпотентно).
    Возвращает путь к файлу или None, если создать не удалось.
    """
    global _file_handler, _file_path
    with _file_lock:
        if _file_handler is not None:
            return _file_path
        try:
            log_dir = paths.logs_dir(root)
            stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            path = log_dir / f"localasr_{stamp}.log"
            # два запуска в одну секунду (скрипты/тесты) — не склеиваем в один
            # файл: добавляем счётчик, у каждого запуска свой лог
            counter = 2
            while path.exists():
                path = log_dir / f"localasr_{stamp}_{counter}.log"
                counter += 1
            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        except OSError as exc:
            get_logger().warning("Не удалось создать файловый лог: %s", exc)
            return None

        handler.setFormatter(_file_formatter())
        handler.setLevel(logging.DEBUG)

        logger = logging.getLogger(LOGGER_NAME)
        logger.addHandler(handler)
        # предупреждения сторонних библиотек — тоже в файл (root-логгер)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        _file_handler = handler
        _file_path = path

        try:
            for line in _session_header():
                logger.debug(line)
            for key, value in (context or {}).items():
                logger.debug("контекст: %s = %s", key, value)
        except Exception:  # noqa: BLE001
            pass

        removed = _prune_old_logs(log_dir)
        logger.info(
            "Файловый лог включён: %s%s", path, f" (удалено старых: {removed})" if removed else ""
        )
        return path


def disable_file_logging(final_note: bool = True) -> None:
    """Отключает файловый лог (файл остаётся на диске)."""
    global _file_handler, _file_path
    with _file_lock:
        handler, _file_handler = _file_handler, None
        path, _file_path = _file_path, None
        if handler is None:
            return
        try:
            if final_note:
                logging.getLogger(LOGGER_NAME).info("Файловый лог выключен (%s)", path)
            logging.getLogger(LOGGER_NAME).removeHandler(handler)
            logging.getLogger().removeHandler(handler)
            handler.flush()
            handler.close()
        except Exception:  # noqa: BLE001
            pass


def set_file_logging(enabled: bool, root: Optional[Path] = None) -> Optional[Path]:
    """Включает/выключает файловый лог на лету. Возвращает путь (или None)."""
    if enabled:
        return enable_file_logging(root)
    disable_file_logging()
    return None


def configure_file_logging(
    enabled: Optional[bool] = None,
    root: Optional[Path] = None,
    context: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """Применяет политику логирования при старте приложения.

    Приоритет: LOCALASR_LOG (0/1) → settings.file_logging → False (по умолчанию).
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
    """Все лог-файлы приложения в logs/ (новые первыми)."""
    try:
        log_dir = paths.logs_dir(root)
        return sorted(log_dir.glob("localasr_*.log"), key=lambda p: p.name, reverse=True)
    except OSError:
        return []


atexit.register(disable_file_logging, final_note=False)


# --------------------------------------------------------------------------- #
#  Мост в GUI
# --------------------------------------------------------------------------- #


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
