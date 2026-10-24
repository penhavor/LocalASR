# -*- coding: utf-8 -*-
"""Единый раннер тестов LocalASR.

Запускает КАЖДЫЙ tests/test_*.py в отдельном процессе (изоляция Qt/torch/
состояния модулей), собирает результаты и печатает сводку. Модули, которым
не хватает тяжёлых зависимостей (torch/PySide6/transformers/…), помечаются
SKIP — это не ошибка: базовые тесты не требуют ничего кроме numpy.

Запуск:
    cd <корень проекта> && python tests/run_all.py            # все тесты
    cd <корень проекта> && python tests/run_all.py --list     # список
    cd <корень проекта> && python tests/run_all.py --only net,logging
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent

# маркеры «не хватает зависимости» — такие модули считаем пропущенными
SKIP_MARKERS = ("ModuleNotFoundError", "ImportError", "DLL load failed")

TIMEOUT_SECONDS = int(os.environ.get("LOCALASR_TEST_TIMEOUT", "900"))


def discover() -> list[Path]:
    return sorted(
        p for p in TESTS_DIR.glob("test_*.py")
        if p.name != Path(__file__).name
    )


def run_one(path: Path) -> tuple[str, float, str]:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("LOCALASR_QUIET", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.pop("LOCALASR_LOG", None)

    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            env=env,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        code = proc.returncode
    except subprocess.TimeoutExpired:
        return "FAIL", time.time() - started, f"таймаут {TIMEOUT_SECONDS} с"

    elapsed = time.time() - started
    if code == 0:
        status = "SKIP" if ("пропущен" in output.lower() and "✓" not in output) else "OK"
        return status, elapsed, output
    if any(marker in output for marker in SKIP_MARKERS) and "ПРОВАЛЕНО" not in output:
        first = next(
            (line for line in output.splitlines() if any(m in line for m in SKIP_MARKERS)),
            "",
        )
        return "SKIP", elapsed, f"не хватает зависимости: {first.strip()[:120]}"
    tail = "\n".join(output.strip().splitlines()[-25:])
    return "FAIL", elapsed, tail


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    tests = discover()
    if "--list" in flags:
        for path in tests:
            print(path.name)
        return 0
    if args:
        wanted = {part.strip().lower() for arg in args for part in arg.split(",")}
        tests = [p for p in tests if any(w in p.stem.lower() for w in wanted)]

    if not tests:
        print("Тесты не найдены.")
        return 2

    print(f"LocalASR: запуск {len(tests)} тестовых модулей (python {sys.version.split()[0]})")
    print("=" * 72)

    results: list[tuple[str, str, float]] = []
    for path in tests:
        print(f"\n>>> {path.name}")
        status, elapsed, output = run_one(path)
        if status == "FAIL":
            print(output)
        results.append((path.name, status, elapsed))
        print(f"    [{status}] {path.name} за {elapsed:.1f} с")

    print("\n" + "=" * 72)
    print("СВОДКА")
    ok = sum(1 for _, s, _ in results if s == "OK")
    failed = [name for name, s, _ in results if s == "FAIL"]
    skipped = [name for name, s, _ in results if s == "SKIP"]
    for name, status, elapsed in results:
        print(f"  {status:4}  {name:38} {elapsed:6.1f} с")
    print(f"\nИтого: {ok} OK, {len(failed)} FAIL, {len(skipped)} SKIP")
    if failed:
        print("Провалены: " + ", ".join(failed))
        return 1
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
