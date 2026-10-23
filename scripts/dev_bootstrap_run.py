"""Полный прогон bootstrap «как у пользователя»: venv + pip install base.txt.

Отличие от реального запуска только одно: вместо открытия GUI (в песочнице нет
системных библиотек Qt) подменяется функция перезапуска — всё остальное
(поиск Python, создание .venv, установка зависимостей, проверка импортов,
отпечаток окружения, быстрый путь при повторном запуске) работает по-настоящему.

    python scripts/dev_bootstrap_run.py [папка-копия-проекта]
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bootstrap  # noqa: E402


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/localasr-boot")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(
        ROOT,
        target,
        ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "logs", "models", "outputs"),
    )
    (target / "models").mkdir()
    (target / "outputs").mkdir()
    print(f"Копия проекта: {target}")

    relaunch_calls: list[list[str]] = []

    def fake_relaunch(root, python_exe, log, argv):
        relaunch_calls.append([root, python_exe, *argv])
        log(f"(имитация) перезапуск GUI: {python_exe}")
        # проверяем, что окружение действительно готово
        missing = bootstrap.check_imports(python_exe, bootstrap.BASE_REQUIREMENTS, log)
        log(f"(имитация) недостающие пакеты после установки: {missing}")
        if missing:
            return 3
        return 0

    bootstrap.relaunch_gui = fake_relaunch  # type: ignore[assignment]

    print("\n=== 1. ПЕРВЫЙ ЗАПУСК (создание .venv + установка) ===")
    started = time.time()
    code = bootstrap.run(root=str(target))
    first_run = time.time() - started
    print(f"\nкод возврата: {code} · время: {first_run:.0f} с")
    assert code == 0, "первый запуск должен завершиться успешно"
    assert relaunch_calls, "GUI должен быть перезапушен"
    assert (target / ".venv").is_dir(), ".venv не создан"
    assert (target / ".venv" / bootstrap.STAMP_FILENAME).is_file(), "нет отпечатка окружения"

    print("\n=== 2. ПОВТОРНЫЙ ЗАПУСК (должен быть мгновенным) ===")
    relaunch_calls.clear()
    started = time.time()
    code = bootstrap.run(root=str(target))
    second_run = time.time() - started
    print(f"код возврата: {code} · время: {second_run:.1f} с")
    assert code == 0
    assert second_run < first_run / 3, "повторный запуск обязан быть заметно быстрее"

    print("\n=== 3. ЗАПУСК ИЗНУТРИ VENV (как после перезапуска) ===")
    import os
    import subprocess

    env = os.environ.copy()
    env[bootstrap.CHILD_ENV_FLAG] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["LOCALASR_QUIET"] = "1"
    venv_python = bootstrap.venv_python(str(target))
    probe = subprocess.run(
        [venv_python, "-c", "import PySide6, faster_whisper, numpy, requests, av; print('imports ok')"],
        capture_output=True, text=True, env=env, cwd=str(target), timeout=300,
    )
    print("импорты в .venv:", probe.stdout.strip() or probe.stderr.strip()[-400:])
    assert "imports ok" in probe.stdout

    doctor = subprocess.run(
        [venv_python, str(target / "run_app.py"), "doctor"],
        capture_output=True, text=True, env=env, cwd=str(target), timeout=300,
    )
    print("\n--- doctor ---")
    print(doctor.stdout[-1600:] or doctor.stderr[-800:])
    assert doctor.returncode == 0, doctor.stderr[-500:]

    listing = subprocess.run(
        [venv_python, str(target / "run_app.py"), "list-models"],
        capture_output=True, text=True, env=env, cwd=str(target), timeout=300,
    )
    assert listing.returncode == 0
    assert "faster-whisper" in listing.stdout

    print(f"\n✅ BOOTSTRAP E2E OK (первый запуск {first_run:.0f} с, повторный {second_run:.1f} с)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
