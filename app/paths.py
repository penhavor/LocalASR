"""Пути, имена файлов и прочие «бытовые» утилиты.

Здесь не должно быть тяжёлых импортов: модуль используется и в тестах, и в CLI.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

from .constants import (
    LOGS_DIRNAME,
    MEDIA_EXTENSIONS,
    MODELS_DIRNAME,
    OUTPUTS_DIRNAME,
    RESOURCES_DIRNAME,
    TEMP_DIRNAME,
)

# --------------------------------------------------------------------------- #
#  Корень проекта
# --------------------------------------------------------------------------- #


def find_root() -> Path:
    """Корень проекта.

    Приоритет:
      1. переменная окружения LOCALASR_ROOT (её ставит bootstrap при перезапуске);
      2. папка, из которой запущен процесс, если в ней есть main.py;
      3. папка на два уровня выше этого файла (обычный запуск из дерева проекта);
      4. папка исполняемого файла (сборка в exe).
    """
    env = os.environ.get("LOCALASR_ROOT")
    if env and (Path(env) / "main.py").is_file():
        return Path(env).resolve()

    cwd = Path.cwd()
    if (cwd / "main.py").is_file() and (cwd / "app").is_dir():
        return cwd.resolve()

    here = Path(__file__).resolve().parent.parent
    if (here / "main.py").is_file():
        return here

    exe_dir = Path(sys.executable).resolve().parent
    if (exe_dir / "main.py").is_file():
        return exe_dir

    return here


ROOT_DIR: Path = find_root()


def models_dir(root: Optional[Path] = None) -> Path:
    return _ensure((root or ROOT_DIR) / MODELS_DIRNAME)


def outputs_dir(root: Optional[Path] = None) -> Path:
    return _ensure((root or ROOT_DIR) / OUTPUTS_DIRNAME)


def logs_dir(root: Optional[Path] = None) -> Path:
    return _ensure((root or ROOT_DIR) / LOGS_DIRNAME)


def resources_dir(root: Optional[Path] = None) -> Path:
    return (root or ROOT_DIR) / RESOURCES_DIRNAME


def cache_dir(root: Optional[Path] = None) -> Path:
    """Временные файлы (WAV-декод, частичные загрузки)."""
    path = (root or ROOT_DIR) / TEMP_DIRNAME
    _ensure(path)
    return path


def venv_dir(root: Optional[Path] = None) -> Path:
    return (root or ROOT_DIR) / ".venv"


def resource_path(name: str, root: Optional[Path] = None) -> Optional[Path]:
    """Файл из resources/ (работает и в собранном виде — через sys._MEIPASS)."""
    base = resources_dir(root)
    candidate = base / name
    if candidate.is_file():
        return candidate
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        alt = Path(meipass) / RESOURCES_DIRNAME / name
        if alt.is_file():
            return alt
    return None


def _ensure(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path


# --------------------------------------------------------------------------- #
#  Имена файлов
# --------------------------------------------------------------------------- #

# Символы, которые нельзя использовать в именах файлов (Windows строже всех)
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
MAX_STEM_LENGTH = 120


def sanitize_stem(name: str) -> str:
    """Делает из произвольной строки безопасное имя файла (без расширения).

    «лекция 1.mp4»                       -> «лекция 1»
    «C:\\Video\\лекция 1.mp4»           -> «лекция 1»   (и на Linux тоже)
    «a/b:c*.mp4»                         -> «b_c»
    """
    text = str(name).replace("\x00", "")
    # Windows-пути должны разбираться и на Linux/macOS (пользователь мог
    # скопировать путь из письма или из логов)
    normalized = text.replace("\\", "/")
    stem = os.path.splitext(os.path.basename(normalized))[0]
    stem = _ILLEGAL_CHARS.sub("_", stem)
    stem = stem.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    stem = re.sub(r"\s{2,}", " ", stem).strip(" .")
    stem = stem.strip("_") if stem.strip("_") else stem  # «b:c*.mp4» -> «b_c», а не «b_c_»
    if not stem:
        stem = "transcript"
    if stem.upper() in _RESERVED_NAMES:
        stem = f"_{stem}"
    if len(stem) > MAX_STEM_LENGTH:
        stem = stem[:MAX_STEM_LENGTH].rstrip(" .")
    return stem or "transcript"


def unique_path(directory: Path, stem: str, suffix: str, overwrite: bool = False) -> Path:
    """Путь результата: `лекция 1.txt`; при коллизии — `лекция 1 (2).txt`."""
    directory = Path(directory)
    _ensure(directory)
    target = directory / f"{stem}{suffix}"
    if overwrite:
        return target
    counter = 2
    while target.exists():
        target = directory / f"{stem} ({counter}){suffix}"
        counter += 1
        if counter > 999:  # защита от бесконечного цикла
            target = directory / f"{stem}-{hashlib.md5(str(target).encode()).hexdigest()[:6]}{suffix}"
            break
    return target


def output_suffix(fmt: str) -> str:
    return {
        "txt": ".txt",
        "txt-time": ".txt",
        "srt": ".srt",
        "vtt": ".vtt",
        "json": ".json",
        "md": ".md",
    }.get(fmt, ".txt")


def is_media_file(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def expand_media_inputs(inputs: Iterable[str]) -> List[Path]:
    """Раскрывает файлы и папки в плоский список медиафайлов.

    Из папок берутся только медиафайлы; из явно указанных файлов — любые
    (расширение может отсутствовать, а декодер сам разберётся).
    """
    result: List[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and is_media_file(child) and not child.name.startswith("."):
                    result.append(child)
        elif path.is_file():
            if path.suffix.lower() and not is_media_file(path):
                continue  # явно не медиа (например, .txt) — пропускаем
            result.append(path)
    # убираем дубликаты, сохраняя порядок
    seen = set()
    unique: List[Path] = []
    for item in result:
        key = str(item.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# --------------------------------------------------------------------------- #
#  Человекочитаемые размеры и место на диске
# --------------------------------------------------------------------------- #


def human_size(num_bytes: Optional[float]) -> str:
    if num_bytes is None:
        return "—"
    value = float(num_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if abs(value) < 1024 or unit == "ТБ":
            if unit == "Б":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}".replace(".0", "")
        value /= 1024
    return f"{value:.1f} ТБ"


def free_space(path: Path) -> Optional[int]:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return None


def ensure_free_space(path: Path, needed_bytes: int, margin_bytes: int = 512 * 1024 * 1024) -> None:
    """Бросает понятную ошибку, если на диске не хватает места."""
    free = free_space(path)
    if free is None:
        return
    if free < needed_bytes + margin_bytes:
        raise OSError(
            f"Недостаточно места на диске: нужно {human_size(needed_bytes + margin_bytes)}, "
            f"свободно {human_size(free)} ({path})."
        )


def temp_dir() -> Path:
    """Системная временная папка (для WAV при декодировании)."""
    return Path(tempfile.gettempdir())


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Пишем во временный файл и переименовываем — никаких «половинок» при сбое."""
    path = Path(path)
    _ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding=encoding, newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    _ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def dir_size(path: Path) -> int:
    total = 0
    try:
        for child in Path(path).rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def short_path(path: Path | str, limit: int = 62) -> str:
    """Сокращает длинный путь для показа в интерфейсе: …/очень/длинный/файл.mp4"""
    text = str(path)
    if len(text) <= limit:
        return text
    head, tail = os.path.splitdrive(text)
    parts = [p for p in tail.replace("\\", "/").split("/") if p]
    if not parts:
        return "…" + text[-limit:]
    out = parts[-1]
    for part in reversed(parts[:-1]):
        candidate = f"{part}/{out}"
        if len(candidate) + 1 > limit - len(head):
            break
        out = candidate
    return f"{head}…/{out}" if head else f"…/{out}"
