"""Paths, file names and other everyday utilities.

No heavy imports here: the module is used by tests and the CLI too.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

from .constants import (
    LOGS_DIRNAME,
    MEDIA_EXTENSIONS,
    MODELS_DIRNAME,
    OUTPUTS_DIRNAME,
    RESOURCES_DIRNAME,
    RUNTIME_DIRNAME,
)
from .i18n import _


# --------------------------------------------------------------------------- #
#  Project root
# --------------------------------------------------------------------------- #


def find_root() -> Path:
    """Project root.

    Priority: LOCALASR_ROOT env var (set by bootstrap on relaunch) -> current
    working dir -> parent of this package -> directory of the executable.
    """
    marker = Path("launcher") / "main.py"

    env = os.environ.get("LOCALASR_ROOT")
    if env and (Path(env) / marker).is_file():
        return Path(env).resolve()

    cwd = Path.cwd()
    if (cwd / marker).is_file() and (cwd / "app").is_dir():
        return cwd.resolve()

    here = Path(__file__).resolve().parent.parent
    if (here / marker).is_file():
        return here

    exe_dir = Path(sys.executable).resolve().parent
    if (exe_dir / marker).is_file():
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


def venv_dir(root: Optional[Path] = None) -> Path:
    return (root or ROOT_DIR) / ".venv"


def runtime_dir(root: Optional[Path] = None) -> Path:
    return (root or ROOT_DIR) / RUNTIME_DIRNAME


def embedded_python(root: Optional[Path] = None) -> Optional[Path]:
    """Portable interpreter (runtime/python, downloaded on first run)."""
    base = runtime_dir(root) / "python"
    exe = base / ("python.exe" if os.name == "nt" else "bin/python3")
    return exe if exe.is_file() else None


def env_dir(root: Optional[Path] = None) -> Path:
    """Python environment the app installs packages into.

    runtime/python for the portable mode, .venv otherwise.
    """
    embedded = embedded_python(root)
    return embedded.parent if embedded else venv_dir(root)


def resource_path(name: str, root: Optional[Path] = None) -> Optional[Path]:
    """File from resources/ (also works in frozen builds via sys._MEIPASS)."""
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
#  File names
# --------------------------------------------------------------------------- #

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
MAX_STEM_LENGTH = 120


def sanitize_stem(name: str) -> str:
    """Turn an arbitrary string into a safe file name (without extension)."""
    text = str(name).replace("\x00", "")
    # Windows paths must also parse on Linux/macOS (a path may come from an
    # email or a log)
    normalized = text.replace("\\", "/")
    stem = os.path.splitext(os.path.basename(normalized))[0]
    stem = _ILLEGAL_CHARS.sub("_", stem)
    stem = stem.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    stem = re.sub(r"\s{2,}", " ", stem).strip(" .")
    stem = stem.strip("_") if stem.strip("_") else stem
    if not stem:
        stem = "transcript"
    if stem.upper() in _RESERVED_NAMES:
        stem = f"_{stem}"
    if len(stem) > MAX_STEM_LENGTH:
        stem = stem[:MAX_STEM_LENGTH].rstrip(" .")
    return stem or "transcript"


def datetime_stamp(when: Optional[float] = None) -> str:
    """Date-time stamp for file names: 2026-10-24 15-30-05.

    Colons are replaced with dashes — Windows forbids them in file names.
    """
    return time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(when if when is not None else time.time()))


def unique_path(directory: Path, stem: str, suffix: str, overwrite: bool = False) -> Path:
    """Result path: `lecture 1.txt`.

    On collision (overwrite off) the name gets a date-time tail; if that is
    taken too (two runs in one second), a counter is appended.
    """
    directory = Path(directory)
    _ensure(directory)
    target = directory / f"{stem}{suffix}"
    if overwrite or not target.exists():
        return target
    target = directory / f"{stem} {datetime_stamp()}{suffix}"
    if not target.exists():
        return target
    counter = 2
    base = f"{stem} {datetime_stamp()}"
    while target.exists():
        target = directory / f"{base} ({counter}){suffix}"
        counter += 1
        if counter > 999:
            target = directory / f"{base}-{hashlib.md5(str(target).encode()).hexdigest()[:6]}{suffix}"
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
    """Expand files and folders into a flat list of media files.

    Folders contribute media files only; explicitly listed files are taken
    as-is (extension may be missing, the decoder will figure it out).
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
                continue
            result.append(path)
    seen = set()
    unique: List[Path] = []
    for item in result:
        key = str(item.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# --------------------------------------------------------------------------- #
#  Human-readable sizes and disk space
# --------------------------------------------------------------------------- #


def human_size(num_bytes: Optional[float]) -> str:
    if num_bytes is None:
        return "—"
    value = float(num_bytes)
    units = (_("B"), _("KB"), _("MB"), _("GB"), _("TB"))
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            if unit == units[0]:
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}".replace(".0", "")
        value /= 1024
    return f"{value:.1f} {units[-1]}"


def free_space(path: Path) -> Optional[int]:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return None


def ensure_free_space(path: Path, needed_bytes: int, margin_bytes: int = 512 * 1024 * 1024) -> None:
    """Raise a clear error when the disk has not enough free space."""
    free = free_space(path)
    if free is None:
        return
    if free < needed_bytes + margin_bytes:
        raise OSError(
            _("Not enough disk space: need {}, free {} ({})").format(
                human_size(needed_bytes + margin_bytes), human_size(free), path
            )
        )


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write to a temp file and rename — no half-written files on failure."""
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
    """Shorten a long path for display: …/very/long/file.mp4"""
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
