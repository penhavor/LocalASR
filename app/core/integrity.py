"""Целостность программы: файлы проекта, папки, пакеты .venv.

Политика простая: программа должна слабо поддаваться деградации.
  * лёгкие папки (models/, outputs/, logs/, settings/) создаются автоматически —
    их отсутствие не ошибка;
  * если удалены файлы самой программы — не падаем, а сообщаем и предлагаем
    скачать их заново с GitHub (вес — считаные килобайты);
  * если в .venv не хватает базовых библиотек — предлагаем починить;
  * если в .venv есть пакеты, которые программе не нужны, — показываем их
    с размером и предлагаем удалить (исключительно по решению пользователя).

Модуль не импортирует Qt: его одинаково используют GUI-воркеры и CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

from .. import paths
from ..constants import REPO_RAW_BASE
from ..logutils import get_logger

log = get_logger("integrity")

ProgressCallback = Callable[[float, str], None]
CancelToken = Callable[[], bool]

#: Верхнеуровневые дистрибутивы, которые программа ставит сама (база + движки).
#: Всё, что не здесь и не требуется никем из установленных (pip --not-required),
#: считается «возможно лишним» — но удаляется ТОЛЬКО по явному выбору пользователя.
KNOWN_TOP_LEVEL = {
    # база (requirements/base.txt)
    "pyside6", "pyside6-addons", "pyside6-essentials", "shiboken6",
    "faster-whisper", "ctranslate2", "av", "numpy", "tokenizers",
    "onnxruntime", "huggingface-hub", "tqdm", "requests", "imageio-ffmpeg",
    # движки по требованию (requirements/extras.txt, engines.json)
    "pywhispercpp", "vosk",
    "torch", "torchaudio", "torchvision", "transformers", "accelerate",
    "soundfile", "hydra-core", "omegaconf", "sentencepiece",
    # CUDA
    "triton",
    # сервис
    "pip", "setuptools", "wheel", "uv",
}

#: префиксы дистрибутивов, которые всегда ожидаемы (CUDA-библиотеки NVIDIA и пр.)
KNOWN_PREFIXES = ("nvidia-", "cuda-", "cudnn", "localasr")


def _normalize(name: str) -> str:
    return (name or "").strip().lower().replace("_", "-")


# --------------------------------------------------------------------------- #
#  Файлы проекта
# --------------------------------------------------------------------------- #


def _manifest() -> List[str]:
    """Манифест файлов проекта из bootstrap.py (единый источник истины)."""
    try:
        root = str(paths.ROOT_DIR)
        if root not in sys.path:
            sys.path.insert(0, root)
        import bootstrap  # noqa: PLC0415 - локальный импорт: bootstrap лежит в корне

        return list(getattr(bootstrap, "PROJECT_FILES", []))
    except Exception as exc:  # noqa: BLE001
        log.warning("Манифест проекта недоступен: %s", exc)
        return []


def missing_project_files(root: Optional[Path] = None) -> List[str]:
    """Файлы из манифеста, которых нет на диске."""
    base = Path(root) if root else paths.ROOT_DIR
    missing = []
    for rel in _manifest():
        if not (base / rel).is_file():
            missing.append(rel)
    return missing


def download_url_for(relpath: str) -> str:
    return f"{REPO_RAW_BASE}/{quote(relpath)}"


def restore_files(
    relpaths: Sequence[str],
    root: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[CancelToken] = None,
) -> Tuple[List[str], List[str]]:
    """Скачивает файлы с GitHub в папку проекта. Возвращает (ok, ошибки)."""
    from . import net  # локальный импорт: requests нужен не всегда

    base = Path(root) if root else paths.ROOT_DIR
    restored: List[str] = []
    failed: List[str] = []
    total = max(1, len(relpaths))
    for index, rel in enumerate(relpaths):
        if cancel is not None and cancel():
            failed.append(f"{rel}: отменено")
            continue
        dest = base / rel
        url = download_url_for(rel)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            def report(done: int, size: int, speed: float, filename: str, _rel=rel) -> None:
                if progress is not None:
                    progress(index / total, f"Скачиваю {_rel}…")

            net.download_file(url, dest, progress=report, cancel=cancel)
            restored.append(rel)
            log.info("Восстановлен файл: %s", rel)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{rel}: {exc}")
            log.warning("Не удалось восстановить %s: %s", rel, exc)
        if progress is not None:
            progress((index + 1) / total, f"{rel} — готово" if rel in restored else f"{rel} — ошибка")
    return restored, failed


# --------------------------------------------------------------------------- #
#  Папки
# --------------------------------------------------------------------------- #


def folder_status(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Состояние рабочих папок. Все они создаются автоматически при нужде."""
    base = Path(root) if root else paths.ROOT_DIR
    items = []
    for name, rel, purpose in (
        ("models/", "models", "скачанные модели"),
        ("outputs/", "outputs", "результаты распознавания"),
        ("logs/", "logs", "журналы работы"),
        ("settings/", "settings", "настройки программы"),
    ):
        path = base / rel
        exists = path.is_dir()
        items.append(
            {
                "name": name,
                "path": str(path),
                "exists": exists,
                "purpose": purpose,
                "status": "на месте" if exists else "нет — создастся автоматически",
            }
        )
    return items


# --------------------------------------------------------------------------- #
#  Аудит .venv
# --------------------------------------------------------------------------- #


def _venv_python() -> str:
    return sys.executable or str(paths.venv_dir() / "Scripts" / "python.exe")


def _run_py(code: str, args: Sequence[str] = (), timeout: int = 120) -> Optional[str]:
    """Запускает код интерпретатором .venv и возвращает stdout (или None)."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        out = subprocess.run(
            [_venv_python(), "-c", code, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Не удалось запустить проверку в .venv: %s", exc)
        return None
    if out.returncode != 0:
        log.debug("Проверка в .venv вернула код %s: %s", out.returncode, (out.stderr or "")[-400:])
        return None
    return out.stdout or ""


def check_base_imports() -> List[str]:
    """Каких базовых библиотек не хватает (дистрибутивы, как в bootstrap)."""
    root = str(paths.ROOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import bootstrap

        pairs = list(bootstrap.BASE_REQUIREMENTS)
    except Exception:  # noqa: BLE001
        pairs = [
            ("PySide6", "PySide6"), ("numpy", "numpy"), ("requests", "requests"),
            ("imageio-ffmpeg", "imageio_ffmpeg"), ("faster-whisper", "faster_whisper"),
            ("av", "av"),
        ]
    code = (
        "import importlib,json,sys\n"
        "mods=json.loads(sys.argv[1])\n"
        "missing=[]\n"
        "for dist,mod in mods:\n"
        "    try:\n"
        "        importlib.import_module(mod)\n"
        "    except BaseException:\n"
        "        missing.append(dist)\n"
        "print('MISSING:'+json.dumps(missing))\n"
    )
    payload = json.dumps([(dist, mod) for dist, mod in pairs if mod])
    out = _run_py(code, [payload], timeout=300)
    if out is None:
        return [dist for dist, mod in pairs if mod]
    for line in out.splitlines():
        if line.startswith("MISSING:"):
            try:
                return list(json.loads(line[len("MISSING:"):]))
            except ValueError:
                pass
    return []


_SIZE_CODE = r"""
import importlib.metadata as md, json, os, sys
wanted = set(json.loads(sys.argv[1]))
out = {}
for dist in md.distributions():
    try:
        name = dist.metadata["Name"]
    except Exception:
        continue
    if not name:
        continue
    key = name.strip().lower().replace("_", "-")
    if key not in wanted:
        continue
    total = 0
    try:
        files = dist.files or []
    except Exception:
        files = []
    for f in files:
        try:
            total += os.path.getsize(dist.locate_file(f))
        except OSError:
            pass
    out[key] = {"version": dist.version, "size": total}
print(json.dumps(out))
"""


def find_extra_packages() -> List[Dict[str, Any]]:
    """Пакеты в .venv, которые не нужны программе (по данным pip --not-required).

    «Не нужен» = никем из установленных пакетов не требуется И отсутствует в
    списке известных программе дистрибутивов. Транзитивные зависимости сюда не
    попадают: pip помечает их как требуемые.
    """
    out = _run_py(
        "import json,subprocess,sys\n"
        "r=subprocess.run([sys.executable,'-m','pip','list','--not-required','--format=json'],\n"
        "                 capture_output=True,text=True)\n"
        "print(r.stdout)\n",
        timeout=180,
    )
    if not out:
        return []
    try:
        # pip может печатать предупреждения до JSON — ищем первую '['
        start = out.find("[")
        leaves = json.loads(out[start:]) if start >= 0 else []
    except ValueError:
        return []

    candidates = []
    for item in leaves:
        name = _normalize(str(item.get("name") or ""))
        if not name:
            continue
        if name in KNOWN_TOP_LEVEL:
            continue
        if any(name.startswith(prefix) for prefix in KNOWN_PREFIXES):
            continue
        candidates.append(name)
    if not candidates:
        return []

    sizes: Dict[str, Dict[str, Any]] = {}
    out = _run_py(_SIZE_CODE, [json.dumps(candidates)], timeout=180)
    if out:
        try:
            sizes = json.loads(out.strip().splitlines()[-1])
        except ValueError:
            sizes = {}

    result = []
    for name in sorted(candidates):
        info = sizes.get(name, {})
        result.append(
            {
                "name": name,
                "version": str(info.get("version") or ""),
                "size_bytes": int(info.get("size") or 0),
            }
        )
    return result


def uninstall_packages(names: Sequence[str]) -> Tuple[bool, str]:
    """pip uninstall -y (строго в .venv). Возвращает (ok, вывод)."""
    if not names:
        return True, "Нечего удалять"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        out = subprocess.run(
            [_venv_python(), "-m", "pip", "uninstall", "-y", *names],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Не удалось запустить pip: {exc}"
    text = (out.stdout or "") + (out.stderr or "")
    return out.returncode == 0, text[-2000:]


def full_audit() -> Dict[str, Any]:
    """Полный отчёт для GUI-диагностики (вызывается из фонового воркера)."""
    return {
        "python": _venv_python(),
        "missing_files": missing_project_files(),
        "folders": folder_status(),
        "base_missing": check_base_imports(),
        "extra_packages": find_extra_packages(),
    }
