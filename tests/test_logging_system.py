# -*- coding: utf-8 -*-
"""Logging system tests.

  * by default the file log is OFF (logs/ stays clean);
  * enabling creates logs/localasr_YYYY-MM-DD_HH-MM-SS.log IMMEDIATELY (no
    restart) and DEBUG records reach it;
  * disabling stops the writes (the file stays on disk);
  * enabling again creates a NEW file (does not append to the old one);
  * the session header contains version/python/platform/project;
  * old logs are pruned (at most LOG_KEEP_FILES kept);
  * LOCALASR_LOG=1 enables logging even with file_logging=False in settings;
  * settings.file_logging defaults to False and survives save/load.

Run:  cd <project root> && python tests/test_logging_system.py
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

        print("== defaults ==")
        check(not logutils.file_logging_enabled(), "file log is off by default")
        check(logutils.current_log_file() is None, "current_log_file() = None")

        print("== enabling on the fly (no restart) ==")
        path = logutils.enable_file_logging(root=tmp, context={"режим": "тест"})
        check(path is not None and path.is_file(), f"file created immediately: {path.name if path else '—'}")
        check(
            path is not None and path.name.startswith("localasr_") and path.name.endswith(".log"),
            "file name = localasr_<date>_<time>.log",
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
        check("отладочная запись DEBUG 42" in text, "DEBUG records reach the file")
        check("информационная запись INFO" in text, "INFO records reach the file")
        check("Traceback" in text and "ValueError" in text, "errors are written with a full traceback")
        check("LocalASR" in text and "python" in text.lower(), "session header contains version and python")
        check("режим = тест" in text, "launch context is written into the header")
        check("| [MainThread] |" in text, "the file shows the thread name")

        print("== idempotence and a new file per enable ==")
        again = logutils.enable_file_logging(root=tmp)
        check(again == path, "re-enabling does not create a second file")
        logutils.disable_file_logging()
        check(not logutils.file_logging_enabled(), "disabling removes the handler")
        size_before = path.stat().st_size if path else 0
        log.debug("эта запись НЕ должна попасть в файл")
        check(
            path is not None and path.stat().st_size == size_before,
            "the file does not grow after disabling",
        )
        time.sleep(1.05)   # the file name includes seconds — wait for a new one
        path2 = logutils.enable_file_logging(root=tmp)
        check(path2 is not None and path2 != path and path2.is_file(),
              "re-enabling creates a NEW file")
        logutils.disable_file_logging()

        print("== pruning old logs ==")
        logs_dir = tmp / "logs"
        for index in range(logutils.LOG_KEEP_FILES + 8):
            fake = logs_dir / f"localasr_2020-01-01_00-{index:02d}-00.log"
            fake.write_text("old log", encoding="utf-8")
        before = len(list(logs_dir.glob("localasr_*.log")))
        path3 = logutils.enable_file_logging(root=tmp)
        logutils.disable_file_logging()
        after = len(list(logs_dir.glob("localasr_*.log")))
        check(before > logutils.LOG_KEEP_FILES, f"before pruning: more files than the limit ({before})")
        check(after <= logutils.LOG_KEEP_FILES, f"after pruning: at most {logutils.LOG_KEEP_FILES} ({after})")
        newest = sorted(logs_dir.glob("localasr_*.log"), key=lambda p: p.name)[-1]
        check(newest.exists(), "the newest log is not deleted")

        print("== LOCALASR_LOG overrides the setting ==")
        os.environ["LOCALASR_LOG"] = "1"
        forced = logutils.configure_file_logging(False, root=tmp)
        check(forced is not None and forced.is_file(), "LOCALASR_LOG=1 forces the log on")
        logutils.disable_file_logging()
        os.environ["LOCALASR_LOG"] = "0"
        blocked = logutils.configure_file_logging(True, root=tmp)
        check(blocked is None, "LOCALASR_LOG=0 forbids the log even with settings=true")
        os.environ.pop("LOCALASR_LOG", None)

        print("== the settings.file_logging option ==")
        sroot = Path(tempfile.mkdtemp(prefix="localasr-set-"))
        settings = Settings.load(sroot)
        check(settings.file_logging is False, "file_logging defaults to False")
        settings.file_logging = True
        settings.save()
        reloaded = Settings.load(sroot)
        check(reloaded.file_logging is True, "file_logging survives save/load")
        check(logutils.configure_file_logging(reloaded.file_logging, root=tmp) is not None,
              "configure_file_logging(True) creates a file")
        logutils.disable_file_logging()
        check(logutils.configure_file_logging(False, root=tmp) is None,
              "configure_file_logging(False) creates nothing")

        print("== bootstrap.log is deleted on a successful start ==")
        import app.paths as app_paths
        from app.cli import cleanup_bootstrap_log

        boot = tmp / "logs" / "bootstrap.log"
        boot.parent.mkdir(parents=True, exist_ok=True)
        boot.write_text("environment install log", encoding="utf-8")
        original_root = app_paths.ROOT_DIR
        try:
            app_paths.ROOT_DIR = tmp
            check(cleanup_bootstrap_log() is True, "existing bootstrap.log deleted")
            check(not boot.exists(), "the file is gone")
            check(cleanup_bootstrap_log() is False, "repeat call without a file -> False (not an error)")
        finally:
            app_paths.ROOT_DIR = original_root

        print("== list_log_files ==")
        files = logutils.list_log_files(tmp)
        check(len(files) >= 1 and all(p.suffix == ".log" for p in files),
              f"list_log_files finds the logs ({len(files)})")
    finally:
        _reset_state()

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("LOGGING TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
