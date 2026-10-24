# -*- coding: utf-8 -*-
"""Тесты системы логирования (задача: подробный лог на каждый запуск).

Проверяется:
  * по умолчанию файловый лог ВЫКЛЮЧЕН (logs/ не засоряется);
  * включение создаёт файл logs/localasr_ГГГГ-ММ-ДД_ЧЧ-ММ-СС.log СРАЗУ
    (без перезапуска), и в него попадают записи уровня DEBUG;
  * выключение прекращает запись (файл остаётся на диске);
  * повторное включение создаёт НОВЫЙ файл (не дописывает старый);
  * шапка сессии содержит версию/python/платформу/проект;
  * старые логи подчищаются (хранится не более LOG_KEEP_FILES);
  * LOCALASR_LOG=1 включает лог даже при file_logging=False в настройках;
  * settings.file_logging по умолчанию False и переживает save/load.

Запуск:  cd <корень проекта> && python tests/test_logging_system.py
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("LOCALASR_QUIET", "1")
os.environ.pop("LOCALASR_LOG", None)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import logutils  # noqa: E402
from app.settings import Settings  # noqa: E402

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


def _reset_state() -> None:
    logutils.disable_file_logging(final_note=False)
    logger = logging.getLogger(logutils.LOGGER_NAME)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            continue
        logger.removeHandler(handler)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="localasr-logs-"))
    try:
        _reset_state()

        print("== по умолчанию ==")
        check(not logutils.file_logging_enabled(), "файловый лог по умолчанию выключен")
        check(logutils.current_log_file() is None, "current_log_file() = None")

        print("== включение на лету (без перезапуска) ==")
        path = logutils.enable_file_logging(root=tmp, context={"режим": "тест"})
        check(path is not None and path.is_file(), f"файл создан сразу: {path.name if path else '—'}")
        check(
            path is not None and path.name.startswith("localasr_") and path.name.endswith(".log"),
            "имя файла = localasr_<дата>_<время>.log",
        )
        log = logutils.get_logger("test")
        log.debug("отладочная запись DEBUG %d", 42)
        log.info("информационная запись INFO")
        try:
            raise ValueError("тестовая ошибка")
        except ValueError:
            log.error("ошибка с трейсбеком", exc_info=True)
        for handler in logging.getLogger(logutils.LOGGER_NAME).handlers:
            handler.flush()
        text = path.read_text(encoding="utf-8") if path else ""
        check("отладочная запись DEBUG 42" in text, "DEBUG-записи попадают в файл")
        check("информационная запись INFO" in text, "INFO-записи попадают в файл")
        check("Traceback" in text and "ValueError" in text, "ошибки пишутся с полным трейсбеком")
        check("LocalASR" in text and "python" in text.lower(), "шапка сессии содержит версию и python")
        check("режим = тест" in text, "контекст запуска записан в шапку")
        check("| [MainThread] |" in text, "в файле видно имя потока")

        print("== идемпотентность и новый файл на каждое включение ==")
        again = logutils.enable_file_logging(root=tmp)
        check(again == path, "повторное включение не создаёт второй файл")
        logutils.disable_file_logging()
        check(not logutils.file_logging_enabled(), "выключение снимает хендлер")
        size_before = path.stat().st_size if path else 0
        log.debug("эта запись НЕ должна попасть в файл")
        check(
            path is not None and path.stat().st_size == size_before,
            "после выключения файл не растёт",
        )
        time.sleep(1.05)   # имя файла включает секунды — ждём новую секунду
        path2 = logutils.enable_file_logging(root=tmp)
        check(path2 is not None and path2 != path and path2.is_file(),
              "повторное включение создаёт НОВЫЙ файл")
        logutils.disable_file_logging()

        print("== чистка старых логов ==")
        logs_dir = tmp / "logs"
        for index in range(logutils.LOG_KEEP_FILES + 8):
            fake = logs_dir / f"localasr_2020-01-01_00-{index:02d}-00.log"
            fake.write_text("старый лог", encoding="utf-8")
        before = len(list(logs_dir.glob("localasr_*.log")))
        path3 = logutils.enable_file_logging(root=tmp)
        logutils.disable_file_logging()
        after = len(list(logs_dir.glob("localasr_*.log")))
        check(before > logutils.LOG_KEEP_FILES, f"до чистки файлов больше лимита ({before})")
        check(after <= logutils.LOG_KEEP_FILES, f"после чистки не более {logutils.LOG_KEEP_FILES} ({after})")
        newest = sorted(logs_dir.glob("localasr_*.log"), key=lambda p: p.name)[-1]
        check(newest.exists(), "самый свежий лог не удалён")

        print("== LOCALASR_LOG переопределяет настройку ==")
        os.environ["LOCALASR_LOG"] = "1"
        forced = logutils.configure_file_logging(False, root=tmp)
        check(forced is not None and forced.is_file(), "LOCALASR_LOG=1 включает лог принудительно")
        logutils.disable_file_logging()
        os.environ["LOCALASR_LOG"] = "0"
        blocked = logutils.configure_file_logging(True, root=tmp)
        check(blocked is None, "LOCALASR_LOG=0 запрещает лог даже при settings=true")
        os.environ.pop("LOCALASR_LOG", None)

        print("== настройка settings.file_logging ==")
        sroot = Path(tempfile.mkdtemp(prefix="localasr-set-"))
        settings = Settings.load(sroot)
        check(settings.file_logging is False, "file_logging по умолчанию False")
        settings.file_logging = True
        settings.save()
        reloaded = Settings.load(sroot)
        check(reloaded.file_logging is True, "file_logging переживает save/load")
        check(logutils.configure_file_logging(reloaded.file_logging, root=tmp) is not None,
              "configure_file_logging(True) создаёт файл")
        logutils.disable_file_logging()
        check(logutils.configure_file_logging(False, root=tmp) is None,
              "configure_file_logging(False) ничего не создаёт")

        print("== bootstrap.log удаляется при успешном запуске ==")
        import app.paths as app_paths
        from app.cli import cleanup_bootstrap_log

        boot = tmp / "logs" / "bootstrap.log"
        boot.parent.mkdir(parents=True, exist_ok=True)
        boot.write_text("лог установки окружения", encoding="utf-8")
        original_root = app_paths.ROOT_DIR
        try:
            app_paths.ROOT_DIR = tmp
            check(cleanup_bootstrap_log() is True, "существующий bootstrap.log удалён")
            check(not boot.exists(), "файла больше нет")
            check(cleanup_bootstrap_log() is False, "повторный вызов без файла -> False (не ошибка)")
        finally:
            app_paths.ROOT_DIR = original_root

        print("== list_log_files ==")
        files = logutils.list_log_files(tmp)
        check(len(files) >= 1 and all(p.suffix == ".log" for p in files),
              f"list_log_files находит логи ({len(files)})")
    finally:
        _reset_state()

    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("ТЕСТ ЛОГИРОВАНИЯ ПРОЙДЕН")
    return 0


if __name__ == "__main__":
    sys.exit(main())
