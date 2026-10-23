#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_app.py — запуск GUI LocalASR.

Этот файл запускается **интерпретатором из .venv** (на Windows — pythonw.exe,
поэтому консольное окно не появляется). Если вызвать его «голым» Python, он
сам делегирует работу bootstrap.py, который подготовит окружение.

Поддерживает и headless-режим:
    python run_app.py transcribe "лекция 1.mp4" --model large-v3-turbo
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _inside_project_venv() -> bool:
    exe = os.path.abspath(sys.executable)
    venv = os.path.join(ROOT, ".venv")
    try:
        return os.path.commonpath([exe, os.path.abspath(venv)]) == os.path.abspath(venv)
    except ValueError:
        return False


def main() -> int:
    headless = len(sys.argv) > 1 and sys.argv[1] in {"transcribe", "doctor", "models", "list-models"}

    if not _inside_project_venv() and not headless:
        # не в том окружении — пусть bootstrap всё подготовит и перезапустит нас
        import bootstrap

        return bootstrap.run(root=ROOT, argv=sys.argv[1:])

    try:
        from app import cli
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            "\n[LocalASR] Не удалось загрузить приложение:\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            "Удали папку .venv и запусти main.py ещё раз — окружение пересоздастся.\n"
        )
        try:
            input("Нажми Enter для выхода...")
        except (EOFError, KeyboardInterrupt, OSError):
            pass
        return 1

    return cli.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
