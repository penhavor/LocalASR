#!/usr/bin/env python3
"""Сверка app/ui/compat.py с .pyi-стабами установленного PySide6.

    python scripts/dev_check_qt_compat.py            # текущий интерпретатор
    python scripts/dev_check_qt_compat.py .venv      # конкретный venv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    search = None
    if len(sys.argv) > 1:
        venv = Path(sys.argv[1])
        search = []
        for pattern in ("lib/python3*/site-packages", "Lib/site-packages", "site-packages"):
            search.extend(str(p) for p in venv.glob(pattern))
        search.append(str(venv))

    from app.ui import qt_pyi_check

    result = qt_pyi_check.verify(search)
    where = result.get("pyside6") or "(PySide6 не найден)"
    print(f"PySide6: {where}")
    if result.get("note"):
        print(result["note"])
        return 0

    checked = result.get("checked", [])
    problems = result.get("problems", [])
    print(f"Проверено членов перечислений: {len(checked)}")
    if problems:
        print("\nПРОБЛЕМЫ:")
        for item in problems:
            print("  ✗", item)
    else:
        print("  все перечисления найдены в ожидаемых классах ✓")

    value_problems = qt_pyi_check.verify_compat_values()
    if value_problems:
        print("\ncompat вернул int вместо enum:")
        for item in value_problems:
            print("  ✗", item)

    return 1 if (problems or value_problems) else 0


if __name__ == "__main__":
    sys.exit(main())
