"""Проверка создания venv «вживую» (одноразовый скрипт для песочницы)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bootstrap  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/venv-check")
root.mkdir(parents=True, exist_ok=True)
log = bootstrap.Logger(str(root / "log.txt"), echo=True)

base, report = bootstrap.find_base_python(log)
print("base python:", base)
print(report)

started = time.time()
try:
    py = bootstrap.create_venv(base, str(root), log)
except Exception as exc:  # noqa: BLE001
    print("ОШИБКА create_venv:", type(exc).__name__, exc)
    raise SystemExit(1) from exc

print(f"venv python: {py} (создан за {time.time() - started:.1f} с)")
print("pip работает:", bootstrap._pip_works(py, log))
print("не хватает:", bootstrap.check_imports(py, bootstrap.BASE_REQUIREMENTS, log))
print("отпечаток:", bootstrap.requirements_fingerprint(str(root), py))
