"""Program integrity: project files, folders, environment packages.

Policy: degrade gracefully. Working folders are created on demand; deleted
program files can be re-downloaded from GitHub; missing base libraries can
be repaired; unneeded packages are listed with sizes and removed only on
explicit user choice. The module does not import Qt (used by GUI and CLI).
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
from ..i18n import _
from ..logutils import get_logger

log = get_logger("integrity")

ProgressCallback = Callable[[float, str], None]
CancelToken = Callable[[], bool]

#: Top-level distributions the app installs itself (base + engines).
#: Anything else that nobody requires (pip --not-required) is "possibly
#: extra" — removed ONLY on explicit user choice.
KNOWN_TOP_LEVEL = {
    # base (requirements/base.txt)
    "pyside6", "pyside6-addons", "pyside6-essentials", "shiboken6",
    "faster-whisper", "ctranslate2", "av", "numpy", "tokenizers",
    "onnxruntime", "huggingface-hub", "tqdm", "requests", "imageio-ffmpeg",
    # on-demand engines (requirements/extras.txt, engines.json)
    "pywhispercpp", "vosk",
    "torch", "torchaudio", "torchvision", "transformers", "accelerate",
    "soundfile", "hydra-core", "omegaconf", "sentencepiece",
    # CUDA
    "triton",
    # service
    "pip", "setuptools", "wheel", "uv",
}

#: distribution prefixes that are always expected (NVIDIA CUDA libs etc.)
KNOWN_PREFIXES = ("nvidia-", "cuda-", "cudnn", "localasr")


def _normalize(name: str) -> str:
    return (name or "").strip().lower().replace("_", "-")


# --------------------------------------------------------------------------- #
#  Project files
# --------------------------------------------------------------------------- #


def _manifest() -> List[str]:
    """Project file manifest from bootstrap.py (single source of truth)."""
    try:
        launcher = str(paths.ROOT_DIR / "launcher")
        if launcher not in sys.path:
            sys.path.insert(0, launcher)
        import bootstrap  # noqa: PLC0415 - local import: bootstrap lives in launcher/

        return list(getattr(bootstrap, "PROJECT_FILES", []))
    except Exception as exc:  # noqa: BLE001
        log.warning("Project manifest unavailable: %s", exc)
        return []


def missing_project_files(root: Optional[Path] = None) -> List[str]:
    """Manifest files missing on disk."""
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
    """Download files from GitHub into the project folder. Returns (ok, errors)."""
    from . import net  # local import: requests is not always needed

    base = Path(root) if root else paths.ROOT_DIR
    restored: List[str] = []
    failed: List[str] = []
    total = max(1, len(relpaths))
    for index, rel in enumerate(relpaths):
        if cancel is not None and cancel():
            failed.append(_("{rel}: cancelled").format(rel=rel))
            continue
        dest = base / rel
        url = download_url_for(rel)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            def report(done: int, size: int, speed: float, filename: str, _rel=rel) -> None:
                if progress is not None:
                    progress(index / total, _("Downloading {rel}…").format(rel=_rel))

            net.download_file(url, dest, progress=report, cancel=cancel)
            restored.append(rel)
            log.info(_("File restored: %s"), rel)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{rel}: {exc}")
            log.warning(_("Failed to restore %s: %s"), rel, exc)
        if progress is not None:
            progress((index + 1) / total,
                     _("{rel} — done").format(rel=rel) if rel in restored
                     else _("{rel} — error").format(rel=rel))
    return restored, failed


# --------------------------------------------------------------------------- #
#  Folders
# --------------------------------------------------------------------------- #


def folder_status(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Working folder status. All of them are created automatically on demand."""
    base = Path(root) if root else paths.ROOT_DIR
    items = []
    for name, rel, purpose in (
        ("models/", "models", _("downloaded models")),
        ("outputs/", "outputs", _("recognition results")),
        ("logs/", "logs", _("activity logs")),
        ("settings/", "settings", _("application settings")),
    ):
        path = base / rel
        exists = path.is_dir()
        items.append(
            {
                "name": name,
                "path": str(path),
                "exists": exists,
                "purpose": purpose,
                "status": _("present") if exists else _("missing — will be created automatically"),
            }
        )
    return items


# --------------------------------------------------------------------------- #
#  Environment audit
# --------------------------------------------------------------------------- #


def _venv_python() -> str:
    return sys.executable or str(paths.env_dir() / "Scripts" / "python.exe")


def _run_py(code: str, args: Sequence[str] = (), timeout: int = 120) -> Optional[str]:
    """Run code with the environment interpreter and return stdout (or None)."""
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
        log.warning("Failed to run the environment check: %s", exc)
        return None
    if out.returncode != 0:
        log.debug("Environment check returned code %s: %s", out.returncode, (out.stderr or "")[-400:])
        return None
    return out.stdout or ""


def check_base_imports() -> List[str]:
    """Which base libraries are missing (distribution names, like bootstrap)."""
    launcher = str(paths.ROOT_DIR / "launcher")
    if launcher not in sys.path:
        sys.path.insert(0, launcher)
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
    """Packages the app does not need (per pip --not-required).

    "Not needed" = required by nobody AND absent from the known distribution
    list. Transitive deps never land here: pip marks them as required.
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
        # pip may print warnings before the JSON — find the first '['
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
    """pip uninstall -y (strictly inside the project env). Returns (ok, output)."""
    if not names:
        return True, _("Nothing to delete")
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
        return False, _("Failed to run pip: {exc}").format(exc=exc)
    text = (out.stdout or "") + (out.stderr or "")
    return out.returncode == 0, text[-2000:]


def full_audit() -> Dict[str, Any]:
    """Full report for GUI diagnostics (called from a background worker)."""
    return {
        "python": _venv_python(),
        "missing_files": missing_project_files(),
        "folders": folder_status(),
        "base_missing": check_base_imports(),
        "extra_packages": find_extra_packages(),
    }
