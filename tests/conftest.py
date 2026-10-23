"""Общая настройка тестов.

Если настоящий PySide6 не установлен (типично для CI-контейнера), подключаем
`tests/stubs` — там минимальная замена Qt, которой достаточно, чтобы импортировать
и частично исполнить UI-код и поймать реальные баги (опечатки, сигнатуры, сигналы).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUBS = ROOT / "tests" / "stubs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pyside_available() -> bool:
    """Именно ИМПОРТ, а не find_spec: PySide6 бывает установлен, но без
    системных библиотек (libglib/libGL) — тогда работать он не будет."""
    if "PySide6.QtWidgets" in sys.modules:
        return True
    try:
        importlib.import_module("PySide6.QtWidgets")
        return True
    except Exception:  # noqa: BLE001 - ImportError, OSError, RuntimeError…
        return False


USING_STUB = False
if not pyside_available():
    for name in [name for name in sys.modules if name == "PySide6" or name.startswith("PySide6.")]:
        del sys.modules[name]
    sys.path.insert(0, str(STUBS))
    USING_STUB = True

import pytest  # noqa: E402


@pytest.fixture()
def temp_root(tmp_path: Path) -> Path:
    """Копия структуры проекта во временной папке."""
    (tmp_path / "models").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "settings").mkdir()
    resources = tmp_path / "resources"
    resources.mkdir()
    source_catalog = ROOT / "resources" / "models_catalog.json"
    if source_catalog.is_file():
        resources.joinpath("models_catalog.json").write_bytes(source_catalog.read_bytes())
    return tmp_path


@pytest.fixture()
def sample_media(tmp_path: Path) -> list[Path]:
    """Фейковые медиафайлы (содержимое не важно — тестируем не декодирование)."""
    files = []
    for name in ("лекция 1.mp4", "lecture-02.mkv", "подкаст.mp3", "notes.txt"):
        path = tmp_path / "media" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 1024)
        files.append(path)
    return files
