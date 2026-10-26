#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bootstrap.py — LocalASR environment preparation.

This module MUST run on a bare Python: standard library only, no module-level
`app.*` imports (otherwise "chicken and egg": reading the dependency list
would require the dependencies).

What run() does:
  1. Picks the interpreter environment:
     * portable mode (Windows): runtime/python — a full portable Python
       (official CPython build) that RUN.bat downloads on the first run;
       packages install into its Lib/site-packages;
     * regular mode: finds a system Python (3.10+, 64-bit) and creates .venv.
  2. Compares the requirements "fingerprint" with the saved one: if it
     matches, dependency checks are skipped and startup is instant.
  3. Otherwise checks imports and installs what is missing (uv if present,
     else pip) — strictly into the project environment.
  4. Relaunches the app with the environment interpreter; on Windows via
     pythonw.exe, so the console disappears and only the GUI remains.

Any failure is shown in a native dialog (Tkinter, when available) and written
to the log instead of dying silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

APP_NAME = "LocalASR"
VENV_DIRNAME = ".venv"
RUNTIME_DIRNAME = "runtime"
PYTHON_DIRNAME = "python"
STAMP_FILENAME = ".localasr-env.json"
LOCK_FILENAME = ".localasr-install.lock"
CHILD_ENV_FLAG = "LOCALASR_BOOTSTRAPPED"

MIN_PYTHON = (3, 10)
# Last version with manually verified wheels for ALL dependencies
# (cp314/win64+linux: ctranslate2, PySide6-abi3, numpy, av, onnxruntime,
# tokenizers, pywhispercpp, vosk). Not a ban: newer versions are accepted
# too — see _version_acceptable().
TESTED_PYTHON_MAX = (3, 14)
# With several Pythons in the system, prefer the most battle-tested ones.
PREFERRED_LAUNCHER_FLAGS = ("-3.12", "-3.13", "-3.11", "-3.10", "-3.14", "-3")

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"

# PyPI distribution -> import name. Order = installation order.
BASE_REQUIREMENTS: List[Tuple[str, str]] = [
    ("PySide6", "PySide6"),
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("imageio-ffmpeg", "imageio_ffmpeg"),
    ("faster-whisper", "faster_whisper"),
    ("av", "av"),
]

# Extra sets installed on demand from the GUI (not at first launch).
EXTRA_REQUIREMENTS: Dict[str, List[Tuple[str, str]]] = {
    "whispercpp": [("pywhispercpp", "pywhispercpp")],
    "vosk": [("vosk", "vosk")],
    "cuda": [
        ("nvidia-cublas-cu12", None),
        ("nvidia-cudnn-cu12", None),
    ],
}

# --------------------------------------------------------------------------- #
#  Minimal i18n (stdlib only; shares resources/i18n/*.json with the app)
# --------------------------------------------------------------------------- #

_TRANSLATIONS: Dict[str, str] = {}
_LANG = "en"


def _detect_system_language() -> str:
    candidates: List[str] = []
    if IS_WINDOWS:
        try:
            import ctypes

            langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            candidates.append({0x19: "ru", 0x09: "en"}.get(langid & 0x3FF, ""))
        except Exception:  # noqa: BLE001
            pass
    for var in ("LANGUAGE", "LC_ALL", "LC_CTYPE", "LANG"):
        value = os.environ.get(var)
        if value:
            candidates.extend(value.split(":"))
    for cand in candidates:
        code = cand.lower().split(".")[0].replace("-", "_").split("_")[0]
        if code in ("en", "ru"):
            return code
    return "en"


def init_i18n(root: str) -> str:
    """Pick the UI language: settings.json -> system -> English."""
    global _LANG, _TRANSLATIONS
    lang = ""
    try:
        settings_path = os.path.join(root, "settings", "settings.json")
        with open(settings_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        payload = data.get("settings", data) if isinstance(data, dict) else {}
        value = str(payload.get("ui_language", "") or "")
        if value in ("en", "ru"):
            lang = value
    except (OSError, ValueError):
        pass
    _LANG = lang or _detect_system_language()
    _TRANSLATIONS = {}
    if _LANG != "en":
        try:
            catalog = os.path.join(root, "resources", "i18n", f"{_LANG}.json")
            with open(catalog, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                _TRANSLATIONS = {k: v for k, v in loaded.items() if isinstance(v, str)}
        except (OSError, ValueError):
            pass
    return _LANG


def tr(msgid: str) -> str:
    return _TRANSLATIONS.get(msgid, msgid)


# --------------------------------------------------------------------------- #
#  Project integrity: file manifest and GitHub recovery
# --------------------------------------------------------------------------- #

REPO_RAW_BASE = "https://raw.githubusercontent.com/penhavor/LocalASR/main"

#: All program files. User data (models/, outputs/, logs/, settings/) and the
#: portable runtime (runtime/) are NOT here — they are created on demand.
PROJECT_FILES: List[str] = [
    "launcher/main.py",
    "launcher/run_app.py",
    "launcher/bootstrap.py",
    "RUN.bat",
    "Create Shortcut.bat",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "requirements/base.txt",
    "requirements/extras.txt",
    "requirements/gpu.txt",
    "requirements/engines.json",
    "resources/models_catalog.json",
    "resources/app.ico",
    "resources/i18n/ru.json",
    "tools/launch.vbs",
    "tools/make_shortcut.vbs",
    "tools/download_runtime.ps1",
    "app/__init__.py",
    "app/cli.py",
    "app/constants.py",
    "app/i18n.py",
    "app/logutils.py",
    "app/paths.py",
    "app/settings.py",
    "app/asr/__init__.py",
    "app/asr/base.py",
    "app/asr/faster_whisper_engine.py",
    "app/asr/hf_pipeline_engine.py",
    "app/asr/registry.py",
    "app/asr/torch_engines.py",
    "app/asr/transcriber.py",
    "app/asr/vosk_engine.py",
    "app/asr/whispercpp_engine.py",
    "app/core/__init__.py",
    "app/core/downloader.py",
    "app/core/exporters.py",
    "app/core/ffmpeg.py",
    "app/core/integrity.py",
    "app/core/models_catalog.py",
    "app/core/net.py",
    "app/core/postprocess.py",
    "app/core/relaunch.py",
    "app/core/sysinfo.py",
    "app/core/updater.py",
    "app/core/winutil.py",
    "app/ui/__init__.py",
    "app/ui/compat.py",
    "app/ui/dialogs.py",
    "app/ui/main_window.py",
    "app/ui/models_panel.py",
    "app/ui/theme.py",
    "app/ui/wheelguard.py",
    "app/ui/widgets.py",
    "app/ui/workers.py",
]


def check_project_files(root: str) -> List[str]:
    """Missing project files (empty = everything in place)."""
    missing = []
    for rel in PROJECT_FILES:
        if not os.path.isfile(os.path.join(root, rel.replace("/", os.sep))):
            missing.append(rel)
    return missing


def restore_project_files(root: str, files: Sequence[str], log: "Logger") -> Tuple[List[str], List[str]]:
    """Download missing files from GitHub (stdlib urllib).

    Returns (restored, errors). Each file is written atomically: .part next
    to the target, then os.replace.
    """
    import urllib.request
    from urllib.parse import quote

    restored: List[str] = []
    failed: List[str] = []
    for rel in files:
        url = f"{REPO_RAW_BASE}/{quote(rel)}"
        dest = os.path.join(root, rel.replace("/", os.sep))
        tmp = dest + ".part"
        try:
            os.makedirs(os.path.dirname(dest) or root, exist_ok=True)
            request = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}-repair"})
            with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as fh:
                while True:
                    block = resp.read(256 * 1024)
                    if not block:
                        break
                    fh.write(block)
            os.replace(tmp, dest)
            restored.append(rel)
            log(tr("File restored: {rel}").format(rel=rel))
        except Exception as exc:  # noqa: BLE001 - network/permissions may fail
            try:
                os.remove(tmp)
            except OSError:
                pass
            failed.append(f"{rel}: {exc}")
            log(tr("!! could not restore {rel}: {exc}").format(rel=rel, exc=exc))
    return restored, failed


def _offer_restore_files(root: str, missing: List[str], log: "Logger") -> bool:
    """Ask the user and restore deleted program files.

    True when fewer files are missing after the restore (GUI run can retry).
    """
    listing = "\n".join("  • " + name for name in missing[:20])
    if len(missing) > 20:
        listing += tr("\n  … and {n} more").format(n=len(missing) - 20)
    message = tr(
        "The program folder lacks {n} file(s):\n{listing}\n\n"
        "LocalASR cannot work fully without them.\n"
        "Download them again from GitHub ({url})?\n"
        "The files weigh a few kilobytes; user data is not affected."
    ).format(n=len(missing), listing=listing, url=REPO_RAW_BASE)
    answer = _ask_yes_no(tr("{app} — program files are missing").format(app=APP_NAME), message)
    if not answer:
        return False
    restored, failed = restore_project_files(root, missing, log)
    if failed:
        _show_error(
            tr("Some files could not be restored:\n\n") + "\n".join(failed[:10])
            + tr("\n\nCheck the internet connection and run the program again."),
            log,
        )
    return bool(restored)


def _ask_yes_no(title: str, message: str) -> bool:
    """Tkinter yes/no question; falls back to the console when Tk is absent."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        result = messagebox.askyesno(title, message, parent=root)
        root.destroy()
        return bool(result)
    except Exception:  # noqa: BLE001
        pass
    print(f"\n[{APP_NAME}] {title}\n{message}")
    try:
        reply = input(tr("\nProceed? [y/n]: ")).strip().lower()
        return reply in ("y", "yes", "д", "да")
    except (EOFError, KeyboardInterrupt, OSError, ValueError):
        return False


def repair_missing_files(root: str, log: Optional["Logger"] = None) -> List[str]:
    """Check the manifest and (with consent) restore files.

    Returns the list of files still missing. Used at startup
    (bootstrap/run_app) and from GUI diagnostics.
    """
    log = log or Logger(log_path(root), echo=False)
    missing = check_project_files(root)
    if not missing:
        return []
    if _offer_restore_files(root, missing, log):
        missing = check_project_files(root)
    return missing


PIP_TIMEOUT_SECONDS = 60


# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #


def project_root(start: Optional[str] = None) -> str:
    """Project root = the folder containing launcher/bootstrap.py."""
    return os.path.dirname(os.path.dirname(os.path.abspath(start or __file__)))


def venv_dir(root: str) -> str:
    return os.path.join(root, VENV_DIRNAME)


def portable_dir(root: str) -> str:
    """Folder of the portable Python (downloaded on first run; may not exist)."""
    return os.path.join(root, RUNTIME_DIRNAME, PYTHON_DIRNAME)


def portable_python(root: str) -> Optional[str]:
    exe = os.path.join(portable_dir(root), "python.exe" if IS_WINDOWS else "bin/python3")
    return exe if os.path.isfile(exe) else None


def env_dir(root: str) -> str:
    """Environment packages are installed into: portable runtime or .venv."""
    emb = portable_python(root)
    return os.path.dirname(emb) if emb else venv_dir(root)


def env_python(root: str) -> Optional[str]:
    """Interpreter inside the project environment (never pythonw — GUI only)."""
    emb = portable_python(root)
    if emb:
        return emb
    vd = venv_dir(root)
    if IS_WINDOWS:
        candidate = os.path.join(vd, "Scripts", "python.exe")
        return candidate if os.path.exists(candidate) else None
    py3 = os.path.join(vd, "bin", "python3")
    return py3 if os.path.exists(py3) else os.path.join(vd, "bin", "python")


def venv_python(root: str) -> str:
    """Interpreter inside .venv (kept for the non-portable flow)."""
    vd = venv_dir(root)
    if IS_WINDOWS:
        return os.path.join(vd, "Scripts", "python.exe")
    return os.path.join(vd, "bin", "python3") if os.path.exists(
        os.path.join(vd, "bin", "python3")
    ) else os.path.join(vd, "bin", "python")


def env_pythonw(root: str) -> str:
    """Windowless interpreter (Windows) — the GUI is relaunched with it."""
    base = env_python(root) or venv_python(root)
    if IS_WINDOWS:
        candidate = os.path.join(os.path.dirname(base), "pythonw.exe")
        if os.path.isfile(candidate):
            return candidate
        candidate = os.path.join(venv_dir(root), "Scripts", "pythonw.exe")
        return candidate if os.path.exists(candidate) else base
    return base


def requirements_files(root: str) -> List[str]:
    req_dir = os.path.join(root, "requirements")
    primary = os.path.join(req_dir, "base.txt")
    if os.path.isfile(primary):
        return [primary]
    fallback = os.path.join(root, "requirements.txt")
    return [fallback] if os.path.isfile(fallback) else []


def stamp_path(root: str) -> str:
    return os.path.join(env_dir(root), STAMP_FILENAME)


def log_path(root: str) -> str:
    return os.path.join(root, "logs", "bootstrap.log")


# --------------------------------------------------------------------------- #
#  Logging (file + callback for the GUI install window)
# --------------------------------------------------------------------------- #


class Logger:
    def __init__(self, path: str, echo: bool = True) -> None:
        self.path = path
        self.echo = echo
        self._lock = threading.Lock()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except OSError:
            pass

    def __call__(self, message: str = "", *, force_echo: bool = False) -> None:
        with self._lock:
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}".rstrip()
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
            if self.echo or force_echo:
                try:
                    print(line, flush=True)
                except (OSError, ValueError):
                    pass  # stdout closed (pythonw) — that is normal


# --------------------------------------------------------------------------- #
#  Finding a suitable system Python
# --------------------------------------------------------------------------- #


def _probe_python(cmd: Sequence[str]) -> Optional[Tuple[str, int]]:
    """Run an interpreter and return ("3.14.8", 64) or None.

    Bitness is read immediately: 32-bit Python does not fit (most deps have
    no wheels for it), and skipping it silently would mean an unexplained
    install failure.
    """
    code = (
        "import sys,struct;"
        "print('%d.%d.%d %d' % (*sys.version_info[:3], struct.calcsize('P') * 8))"
    )
    try:
        probe = subprocess.run(
            [str(c) for c in cmd] + ["-c", code],
            capture_output=True,
            text=True,
            timeout=25,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if probe.returncode != 0:
        return None
    lines = (probe.stdout or "").strip().splitlines()
    if not lines:
        return None
    parts = lines[-1].split()
    if not parts:
        return None
    version = parts[0]
    bits = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return version, bits


def _version_tuple(version: str) -> Tuple[int, int]:
    try:
        parts = [int(x) for x in version.split(".")[:2]]
    except ValueError:
        return (0, 0)
    while len(parts) < 2:
        parts.append(0)
    return parts[0], parts[1]


def _version_acceptable(parts: Tuple[int, int]) -> bool:
    """The lower bound is hard; there is NO upper bound: a new Python is fine."""
    return parts >= MIN_PYTHON


def _python_is_store_stub(exe: str) -> bool:
    """Microsoft Store stub (`WindowsApps\\python.exe`) — not a real Python."""
    norm = exe.replace("\\", "/").lower()
    return "windowsapps" in norm or "microsoft.store" in norm


def find_base_python(log: Logger) -> Tuple[Optional[str], str]:
    """Find an interpreter able to create a venv. Returns (path, report)."""
    report: List[str] = []
    candidates: List[str] = []

    if IS_WINDOWS:
        launcher = shutil.which("py")
        if launcher:
            for flag in PREFERRED_LAUNCHER_FLAGS:
                candidates.append(f"{launcher} {flag}")
    candidates.append(sys.executable)

    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    if IS_WINDOWS:
        local = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python")
        if os.path.isdir(local):
            for entry in sorted(os.listdir(local), reverse=True):
                exe = os.path.join(local, entry, "python.exe")
                if os.path.isfile(exe):
                    candidates.append(exe)

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        parts = cand.split(" ")
        exe = parts[0]
        if _python_is_store_stub(exe):
            report.append(tr("skipped (Microsoft Store stub): {exe}").format(exe=exe))
            continue
        if not os.path.exists(exe):
            report.append(tr("skipped (not found): {cand}").format(cand=cand))
            continue
        probed = _probe_python(parts)
        if probed is None:
            report.append(tr("skipped (does not run or no such version): {cand}").format(cand=cand))
            continue
        version, bits = probed
        if bits and bits != 64:
            report.append(tr("skipped ({bits}-bit, 64-bit required): {cand}").format(bits=bits, cand=cand))
            continue
        version_parts = _version_tuple(version)
        if not _version_acceptable(version_parts):
            report.append(tr(
                "skipped (Python {min0}.{min1}+ required, found {ver}): {cand}"
            ).format(min0=MIN_PYTHON[0], min1=MIN_PYTHON[1], ver=version, cand=cand))
            continue
        if version_parts > TESTED_PYTHON_MAX:
            note = tr(
                "Python {ver} is newer than the tested {max0}.{max1} — enabling support "
                "(dependency wheels use the stable ABI and support it)"
            ).format(ver=version, max0=TESTED_PYTHON_MAX[0], max1=TESTED_PYTHON_MAX[1])
            report.append(note)
            log(note)
        report.append(tr("suitable: {cand} (Python {ver}, {bits}-bit)").format(
            cand=cand, ver=version, bits=bits or "?"))
        log(tr("Base Python: {cand} -> {ver}").format(cand=cand, ver=version))
        return cand, "\n".join(report)

    report.append(tr("no candidate fit"))
    return None, "\n".join(report)


# --------------------------------------------------------------------------- #
#  Portable Python preparation
# --------------------------------------------------------------------------- #


def _ensure_pip(py: str, log: Logger, on_line=None) -> None:
    """Make sure the portable Python has a working pip.

    The full CPython distribution ships with pip; this is a safety net:
    ensurepip first, get-pip.py download as the last resort.
    """
    if _pip_works(py, log):
        return
    on_line and on_line(tr("Installing pip into the portable Python…"))
    log("pip is missing — trying ensurepip")
    _run([py, "-m", "ensurepip", "--upgrade"], log, 300, on_line)
    if not _pip_works(py, log):
        # last resort: download get-pip.py (requires internet)
        get_pip = os.path.join(os.path.dirname(py), "get-pip.py")
        _download_file("https://bootstrap.pypa.io/get-pip.py", get_pip, log)
        _run([py, get_pip, "--no-warn-script-location"], log, 900, on_line)
    if not _pip_works(py, log):
        raise BootstrapError(tr(
            "The portable Python has no working pip and it could not be installed.\n"
            "Check the internet connection or reinstall the program folder."
        ))


# --------------------------------------------------------------------------- #
#  venv creation (non-portable mode)
# --------------------------------------------------------------------------- #


def _run(cmd: Sequence[str], log: Logger, timeout: int = 900, on_line=None) -> int:
    """Run a command with streamed stdout/stderr."""
    printable = " ".join(f'"{c}"' if " " in c else str(c) for c in cmd)
    log(f"$ {printable}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env.pop("PYTHONHOME", None)
    # the parent's virtual environment must not leak into the child pip
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)

    creationflags = 0
    if IS_WINDOWS:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
    except OSError as exc:
        log(tr("!! failed to run the command: {exc}").format(exc=exc))
        return 127

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        log(line, force_echo=False)
        if on_line is not None:
            try:
                on_line(line)
            except Exception:  # noqa: BLE001
                pass
    proc.wait(timeout=timeout)
    log(tr("-> return code {code}").format(code=proc.returncode))
    return proc.returncode


def create_venv(base_python: str, root: str, log: Logger, on_line=None) -> str:
    """Create (or repair) .venv. Returns the python path inside it."""
    vd = venv_dir(root)
    py = venv_python(root)

    if os.path.isfile(py):
        probe = subprocess.run(
            [py, "-c", "print('ok')"], capture_output=True, text=True, timeout=60
        )
        if probe.returncode == 0:
            log(tr("The virtual environment already exists — using it."))
            return py
        log(tr("The existing .venv is broken — recreating."))
        try:
            shutil.rmtree(vd, ignore_errors=True)
        except OSError as exc:
            log(tr("Failed to delete .venv: {exc}").format(exc=exc))

    base_cmd = base_python.split(" ")
    log(tr("Creating the virtual environment: {vd}").format(vd=vd))
    on_line and on_line(tr("Creating the virtual environment (.venv)…"))
    code = _run(base_cmd + ["-m", "venv", vd], log, timeout=600, on_line=on_line)

    if code != 0 or not os.path.isfile(venv_python(root)):
        # typical Debian/Ubuntu case: python3-venv (ensurepip) not installed
        log(tr("Regular venv failed — trying --without-pip."))
        on_line and on_line(tr("Trying to create the environment without pip (--without-pip)…"))
        shutil.rmtree(vd, ignore_errors=True)
        code = _run(
            base_cmd + ["-m", "venv", "--without-pip", vd],
            log,
            timeout=600,
            on_line=on_line,
        )
        if code != 0 or not os.path.isfile(venv_python(root)):
            raise BootstrapError(tr(
                "Failed to create the virtual environment.\n"
                "Windows: reinstall Python with \"Install launcher for all users\" checked\n"
                "Linux (Debian/Ubuntu): sudo apt install python3-venv python3-full\n"
                "\nDetails in the log: {log}"
            ).format(log=log.path))
        # add pip manually
        on_line and on_line(tr("Installing pip into the environment…"))
        _run(
            [venv_python(root), "-c", "import ensurepip, sys; ensurepip._bootstrap()"],
            log,
            timeout=600,
            on_line=on_line,
        )
        if not _pip_works(venv_python(root), log):
            get_pip = os.path.join(vd, "get-pip.py")
            _download_file(
                "https://bootstrap.pypa.io/get-pip.py", get_pip, log
            )
            _run([venv_python(root), get_pip, "--no-warn-script-location"], log, 900, on_line)

    if not os.path.isfile(venv_python(root)):
        raise BootstrapError(tr("venv created but the interpreter was not found: {py}").format(py=venv_python(root)))

    # a current pip = fewer wheel/metadata surprises
    on_line and on_line(tr("Updating pip…"))
    _run(
        [venv_python(root), "-m", "pip", "install", "--upgrade",
         "pip>=24.0", "setuptools>=68", "wheel"],
        log,
        timeout=900,
        on_line=on_line,
    )
    return venv_python(root)


def _pip_works(py: str, log: Logger) -> bool:
    try:
        out = subprocess.run(
            [py, "-m", "pip", "--version"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return False
    log(f"pip: {out.stdout.strip() or out.stderr.strip()}")
    return out.returncode == 0


def _download_file(url: str, dest: str, log: Logger) -> bool:
    """stdlib download (urllib) — requests may not exist at this stage."""
    import urllib.request

    log(tr("Downloading {url}").format(url=url))
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        return True
    except Exception as exc:  # noqa: BLE001 - the network may fail for any reason
        log(tr("!! failed to download {url}: {exc}").format(url=url, exc=exc))
        return False


# --------------------------------------------------------------------------- #
#  Dependency checks
# --------------------------------------------------------------------------- #


def requirements_fingerprint(root: str, python_exe: str) -> str:
    payload = {
        "python": python_exe,
        "version": "%d.%d" % sys.version_info[:2],
        "platform": platform.platform(),
        "base": [dist for dist, _ in BASE_REQUIREMENTS],
        "files": {},
    }
    for path in requirements_files(root):
        try:
            with open(path, "rb") as fh:
                payload["files"][os.path.basename(path)] = hashlib.sha256(
                    fh.read()
                ).hexdigest()
        except OSError:
            continue
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _read_stamp(root: str) -> dict:
    try:
        with open(stamp_path(root), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_stamp(root: str, fingerprint: str, extras: Sequence[str]) -> None:
    data = {
        "fingerprint": fingerprint,
        "extras": sorted(extras),
        "python": sys.executable,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        os.makedirs(os.path.dirname(stamp_path(root)), exist_ok=True)
        tmp = stamp_path(root) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, stamp_path(root))
    except OSError:
        pass


def check_imports(python_exe: str, pairs: Sequence[Tuple[str, str]], log: Logger) -> List[str]:
    """One interpreter call for all packages — the fastest way.

    Returns the list of missing distributions.
    """
    active = [(dist, mod) for dist, mod in pairs if mod]
    if not active:
        return []

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
    payload = json.dumps(active)
    try:
        out = subprocess.run(
            [python_exe, "-c", code, payload],
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log(tr("!! the import check crashed: {exc} — treating everything as missing").format(exc=exc))
        return [dist for dist, _ in active]

    marker = "MISSING:"
    missing: List[str] = []
    for line in (out.stdout or "").splitlines():
        if line.startswith(marker):
            try:
                missing = json.loads(line[len(marker):])
            except ValueError:
                missing = [dist for dist, _ in active]
            break
    else:
        log(tr("!! no response from the import check: {out!r} {err!r}").format(
            out=out.stdout, err=out.stderr))
        missing = [dist for dist, _ in active]
    return missing


# --------------------------------------------------------------------------- #
#  Package installation (uv when available, else pip)
# --------------------------------------------------------------------------- #


def _uv_executable(python_exe: str) -> Optional[str]:
    uv = shutil.which("uv")
    if uv:
        return uv
    scripts = os.path.dirname(python_exe)
    for cand in (os.path.join(scripts, "uv.exe" if IS_WINDOWS else "uv"),):
        if os.path.isfile(cand):
            return cand
    return None


def install_packages(
    python_exe: str,
    distributions: Sequence[str],
    root: str,
    log: Logger,
    on_line=None,
    extra_args: Optional[Sequence[str]] = None,
) -> int:
    if not distributions:
        return 0

    extra_args = list(extra_args or [])
    uv = _uv_executable(python_exe)
    if uv:
        cmd = [uv, "pip", "install", "--python", python_exe, *extra_args, *distributions]
        log(tr("Using uv (fast installation)."))
        on_line and on_line(tr("Installing (uv): {list}").format(list=", ".join(distributions)))
        code = _run(cmd, log, timeout=3600, on_line=on_line)
        if code == 0:
            return 0
        log(tr("uv did not manage — trying pip."))

    on_line and on_line(tr("Installing (pip): {list}").format(list=", ".join(distributions)))
    cmd = [
        python_exe,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--prefer-binary",
        "--timeout",
        str(PIP_TIMEOUT_SECONDS),
        "--retries",
        "3",
        *extra_args,
        *distributions,
    ]
    return _run(cmd, log, timeout=5400, on_line=on_line)


def install_requirements_file(
    python_exe: str, req_file: str, log: Logger, on_line=None
) -> int:
    uv = _uv_executable(python_exe)
    if uv:
        code = _run(
            [uv, "pip", "install", "--python", python_exe, "-r", req_file],
            log,
            timeout=5400,
            on_line=on_line,
        )
        if code == 0:
            return 0
    return _run(
        [
            python_exe, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-input", "--prefer-binary", "--timeout", str(PIP_TIMEOUT_SECONDS),
            "--retries", "3", "-r", req_file,
        ],
        log,
        timeout=5400,
        on_line=on_line,
    )


# --------------------------------------------------------------------------- #
#  Re-entrancy lock (two windows must not install packages at once)
# --------------------------------------------------------------------------- #


class InstallLock:
    def __init__(self, root: str) -> None:
        self.path = os.path.join(env_dir(root), LOCK_FILENAME)
        self.acquired = False

    def __enter__(self) -> "InstallLock":
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            self.acquired = True
        except FileExistsError:
            # a stale lock (>30 min) counts as dead
            try:
                if time.time() - os.path.getmtime(self.path) > 1800:
                    os.remove(self.path)
                    return self.__enter__()
            except OSError:
                pass
            self.acquired = False
        except OSError:
            self.acquired = True  # not critical: better to continue
        return self

    def __exit__(self, *exc_info) -> None:
        if self.acquired:
            try:
                os.remove(self.path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
#  Relaunch without a console
# --------------------------------------------------------------------------- #


def relaunch_gui(root: str, python_exe: str, log: Logger, argv: Sequence[str]) -> int:
    """Start the app in the project environment and (on Windows) without a console."""
    target = env_pythonw(root)
    if not os.path.exists(target):
        target = python_exe
    app_entry = os.path.join(root, "launcher", "run_app.py")
    if not os.path.isfile(app_entry):
        app_entry = os.path.join(root, "launcher", "main.py")

    env = os.environ.copy()
    env[CHILD_ENV_FLAG] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["LOCALASR_ROOT"] = root
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)

    cmd = [target, app_entry, *argv]
    log(tr("Relaunching the GUI: ") + " ".join(cmd))

    kwargs: dict = {"cwd": root, "env": env, "stdin": subprocess.DEVNULL}
    if IS_WINDOWS:
        # DETACHED_PROCESS and CREATE_NO_WINDOW are mutually exclusive flags;
        # passing both makes CreateProcess fail.
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        kwargs.update(
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=detached | new_group,
        )
    else:
        kwargs.update(
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    try:
        proc = subprocess.Popen(cmd, **kwargs)
    except OSError as exc:
        log(tr("!! failed to relaunch the GUI: {exc}").format(exc=exc))
        # plan B: run in the current process (console stays, but the app opens)
        return 2

    # wait a couple of seconds: if the process died instantly — report the error
    try:
        code = proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        log(tr("GUI started, pid={pid}").format(pid=proc.pid))
        return 0
    log(tr("!! the GUI exited immediately with code {code}").format(code=code))
    return code or 3


# --------------------------------------------------------------------------- #
#  Native install window (Tkinter) — visible progress, no console needed
# --------------------------------------------------------------------------- #


class ProgressUI:
    """Mini Tkinter window. If Tk is unavailable — prints to the console."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.ok = True
        self._root = None
        self._text = None
        self._status = None
        self._progress = None
        self._lock = threading.Lock()
        self._cancelled = False
        try:
            import tkinter as tk
            from tkinter import ttk

            self._tk = tk
            self._ttk = ttk
        except Exception:  # noqa: BLE001
            self.ok = False
            return

        try:
            root = tk.Tk()
            root.title(title)
            root.geometry("820x520")
            root.minsize(560, 360)
            try:
                root.attributes("-topmost", True)
            except Exception:  # noqa: BLE001
                pass

            header = tk.Frame(root, bg="#15171c")
            header.pack(fill="x")
            tk.Label(
                header,
                text=tr("{app} — environment preparation").format(app=APP_NAME),
                bg="#15171c",
                fg="#f2f4f8",
                font=("Segoe UI" if IS_WINDOWS else "TkDefaultFont", 14, "bold"),
                anchor="w",
                padx=16,
                pady=12,
            ).pack(fill="x")
            self._status = tk.Label(
                header,
                text=tr("Initializing…"),
                bg="#15171c",
                fg="#9aa4b2",
                anchor="w",
                padx=16,
                pady=(0, 10),
            )
            self._status.pack(fill="x")

            self._progress = ttk.Progressbar(root, mode="indeterminate")
            self._progress.pack(fill="x", padx=16, pady=(8, 4))
            self._progress.start(12)

            self._text = tk.Text(
                root, bg="#0e1014", fg="#c9d1d9", insertbackground="#c9d1d9",
                wrap="none", font=("Consolas" if IS_WINDOWS else "TkFixedFont", 9),
            )
            self._text.pack(fill="both", expand=True, padx=16, pady=8)

            bottom = tk.Frame(root)
            bottom.pack(fill="x", padx=16, pady=(0, 12))
            tk.Label(
                bottom,
                text=tr("This window closes automatically when the interface starts."),
                fg="#6b7280",
                anchor="w",
            ).pack(side="left")

            self._root = root
            root.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:  # noqa: BLE001
            self.ok = False

    # -- public methods ---------------------------------------------------- #
    def set_status(self, message: str) -> None:
        if not self.ok:
            return
        with self._lock:
            try:
                self._root.after(0, lambda: self._status.configure(text=message))
            except Exception:  # noqa: BLE001
                pass

    def log_line(self, line: str) -> None:
        if not self.ok:
            return
        with self._lock:
            try:
                self._root.after(0, self._append, line[:400])
            except Exception:  # noqa: BLE001
                pass

    def _append(self, line: str) -> None:
        try:
            self._text.configure(state="normal")
            self._text.insert("end", line + "\n")
            # do not grow the buffer forever
            if int(self._text.index("end-1c").split(".")[0]) > 3000:
                self._text.delete("1.0", "500.0")
            self._text.see("end")
            self._text.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def finish(self) -> None:
        if not self.ok:
            return
        try:
            self._progress.stop()
            self._root.destroy()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.ok = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _on_close(self) -> None:
        self._cancelled = True
        self.finish()


# --------------------------------------------------------------------------- #
#  Main logic
# --------------------------------------------------------------------------- #


class BootstrapError(RuntimeError):
    pass


def _wheel_hint(root: str) -> str:
    if portable_python(root):
        return tr(
            "The portable Python is fixed and known-good; the usual causes are "
            "no internet, a blocking antivirus or a full disk."
        )
    return tr(
        "If the error looks like \"no wheels for your Python version\":\n"
        "install Python 3.12 or 3.13 alongside (https://www.python.org/downloads/) —\n"
        "on the next launch LocalASR picks it automatically; nothing must be deleted."
    )


def run(root: Optional[str] = None, argv: Optional[Sequence[str]] = None) -> int:
    root = root or project_root()
    argv = list(argv if argv is not None else sys.argv[1:])

    # service modes (callable even without a ready environment)
    if "--doctor" in argv:
        return doctor(root)
    if "--force-install" in argv:
        os.environ["LOCALASR_FORCE_INSTALL"] = "1"
        argv = [a for a in argv if a != "--force-install"]
    if os.environ.get("LOCALASR_FORCE_INSTALL") == "1":
        argv = [a for a in argv if a != "--force-install"]

    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    init_i18n(root)

    already_child = os.environ.get(CHILD_ENV_FLAG) == "1"
    log = Logger(log_path(root), echo=not already_child and sys.stdout is not None)
    log("=" * 70)
    log(f"{APP_NAME} bootstrap | python={sys.executable} | {platform.platform()}")

    portable = portable_python(root)
    py = portable or venv_python(root)

    # 1. Already inside the right environment? -> fast path, straight to GUI.
    if already_child or _same_interpreter(sys.executable, py):
        missing = _quick_check(py, root, log)
        if not missing:
            if already_child:
                log(tr("Environment ready — starting the GUI in this process."))
                return _run_gui_inprocess(root, argv, log)
            _write_stamp(root, requirements_fingerprint(root, py), [])
            return relaunch_gui(root, py, log, argv)
        log(tr("Missing in the environment: {missing} — installing.").format(missing=missing))

    # 2. Full environment preparation is needed.
    ui = ProgressUI(f"{APP_NAME} — " + tr("installation"))
    on_line = ui.log_line if ui.ok else None
    ui.set_status(tr("Preparing the environment…")) if ui.ok else None

    try:
        if portable:
            ui.set_status(tr("Preparing the portable Python…")) if ui.ok else None
            _ensure_pip(portable, log, on_line=on_line)
            base_python = None
        else:
            ui.set_status(tr("Looking for a suitable Python…")) if ui.ok else None
            base_python, report = find_base_python(log)
            if base_python is None:
                raise BootstrapError(tr(
                    "No suitable Python found.\n\n"
                    "Python {min0}.{min1} or newer, 64-bit, is required "
                    "(tested up to {max0}.{max1}; newer versions are accepted too).\n"
                    "Download it from https://www.python.org/downloads/ and check\n"
                    "\"Add python.exe to PATH\" during installation. 32-bit Python will not work.\n\n"
                    "What was checked:\n{report}"
                ).format(min0=MIN_PYTHON[0], min1=MIN_PYTHON[1],
                         max0=TESTED_PYTHON_MAX[0], max1=TESTED_PYTHON_MAX[1],
                         report=textwrap.indent(report, "  ")))

        with InstallLock(root) as lock:
            if not lock.acquired:
                raise BootstrapError(tr(
                    "Another LocalASR window is already installing the environment.\n"
                    "Wait for it to finish and run the program again."
                ))

            if not portable:
                ui.set_status(tr("Creating the virtual environment…")) if ui.ok else None
                py = create_venv(base_python, root, log, on_line=on_line)

            fingerprint = requirements_fingerprint(root, py)
            stamp = _read_stamp(root)
            force = os.environ.get("LOCALASR_FORCE_INSTALL") == "1"

            if stamp.get("fingerprint") == fingerprint and not force:
                missing = _quick_check(py, root, log)
            else:
                ui.set_status(tr("Installing dependencies (this may take a few minutes)…")) if ui.ok else None
                for req in requirements_files(root):
                    code = install_requirements_file(py, req, log, on_line=on_line)
                    if code != 0:
                        raise BootstrapError(tr(
                            "pip/uv exited with an error (code {code}) while installing {req}.\n"
                            "Full log: {log}\n\n{hint}"
                        ).format(code=code, req=os.path.basename(req), log=log.path,
                                 hint=_wheel_hint(root)))
                missing = _quick_check(py, root, log)

            if missing:
                ui.set_status(tr("Installing: {list}").format(list=", ".join(missing))) if ui.ok else None
                code = install_packages(py, missing, root, log, on_line=on_line)
                if code != 0:
                    raise BootstrapError(tr(
                        "Failed to install packages: {list}\n\nFull log: {log}\n\n{hint}"
                    ).format(list=", ".join(missing), log=log.path, hint=_wheel_hint(root)))
                missing = _quick_check(py, root, log)

            if missing:
                raise BootstrapError(tr(
                    "Packages still cannot be imported after installation: {list}\n\n"
                    "Possible causes: antivirus blocked files, the disk is full, "
                    "or an incompatible Python version.\nFull log: {log}"
                ).format(list=", ".join(missing), log=log.path))

            _write_stamp(root, fingerprint, [])
            log(tr("Environment ready."))

        ui.set_status(tr("Starting LocalASR…")) if ui.ok else None
        code = relaunch_gui(root, py, log, argv)
        ui.finish()
        if code != 0:
            raise BootstrapError(tr(
                "The application did not start (code {code}).\n"
                "Startup logs: {logs}\n"
                "Install log: {log}"
            ).format(code=code, logs=os.path.join(root, "logs"), log=log.path))
        return 0

    except BootstrapError as exc:
        ui.finish()
        _show_error(str(exc), log)
        return 1
    except Exception:  # noqa: BLE001
        ui.finish()
        import traceback

        _show_error(
            tr("Unexpected error while preparing the environment:\n\n")
            + traceback.format_exc(limit=12),
            log,
        )
        return 1


def _quick_check(py: str, root: str, log: Logger) -> List[str]:
    return check_imports(py, BASE_REQUIREMENTS, log)


def _same_interpreter(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except (OSError, AttributeError, ValueError):
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _run_gui_inprocess(root: str, argv: Sequence[str], log: Logger) -> int:
    """Already inside the ready environment — start the GUI without relaunching."""
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from app import cli  # local import: dependencies exist by now
    except Exception as exc:  # noqa: BLE001
        log(tr("Failed to import the application: {exc!r}").format(exc=exc))
        # a common cause: the user accidentally deleted program files.
        # Check the manifest and offer to restore them from GitHub.
        missing = check_project_files(root)
        if missing and _offer_restore_files(root, missing, log):
            for name in list(sys.modules):
                if name == "app" or name.startswith("app."):
                    del sys.modules[name]
            try:
                from app import cli  # noqa: F811 - retry after the restore
            except Exception as exc2:  # noqa: BLE001
                log(tr("Import after the restore failed: {exc!r}").format(exc=exc2))
                _show_error(tr(
                    "Files were restored but the application still does not load:\n\n"
                    "{type}: {exc}\n\nLog: {log}"
                ).format(type=type(exc2).__name__, exc=exc2, log=log.path), log)
                return 1
            return cli.main(argv)
        _show_error(tr(
            "Failed to load the application modules.\n\n{type}: {exc}\n\n"
        ).format(type=type(exc).__name__, exc=exc)
            + (tr(
                "The program folder lacks files (listed above) — restore them "
                "or download the project again from GitHub.\n\n"
            ) if missing else "")
            + tr(
                "If the files are in place — delete the environment folder and run the "
                "program again: the environment will be recreated automatically.\n\n"
            )
            + tr("Log: {log}").format(log=log.path),
            log,
        )
        return 1
    return cli.main(argv)


def _show_error(message: str, log: Logger) -> None:
    log(tr("ERROR: ") + message.replace("\n", " | "))
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror(tr("{app} — startup error").format(app=APP_NAME), message, parent=root)
        root.destroy()
    except Exception:  # noqa: BLE001
        pass

    print("\n" + "=" * 70, file=sys.stderr)
    print(f"[{APP_NAME}] " + tr("ERROR:"), file=sys.stderr)
    print(message, file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    # keep the console open: on a double click it would vanish instantly
    if not os.environ.get("LOCALASR_NO_PAUSE"):
        try:
            input(tr("\nPress Enter to close the window..."))
        except (EOFError, KeyboardInterrupt, OSError, ValueError):
            pass
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
#  Service commands
# --------------------------------------------------------------------------- #


def doctor(root: Optional[str] = None) -> int:
    """Environment diagnostics: python, env, ffmpeg, dependencies, disk space."""
    root = root or project_root()
    init_i18n(root)
    log = Logger(log_path(root), echo=True)
    print(f"{APP_NAME} doctor")
    print("-" * 60)
    print(tr("Project root : {root}").format(root=root))
    print(f"{tr('System')}       : {platform.platform()}")
    print(tr("Current Python: {exe} ({ver})").format(exe=sys.executable, ver=sys.version.split()[0]))
    portable = portable_python(root)
    if portable:
        print(tr("Portable Python: {py}").format(py=portable))
    else:
        base, report = find_base_python(log)
        print(tr("Base Python  : {base}").format(base=base or tr("NOT FOUND")))
        print(textwrap.indent(report, "  "))
    py = env_python(root) or venv_python(root)
    print(tr("env python   : {py} ({state})").format(
        py=py, state=tr("exists") if os.path.isfile(py) else tr("absent")))
    if os.path.isfile(py):
        missing = check_imports(py, BASE_REQUIREMENTS, log)
        print(tr("Missing      : {missing}").format(
            missing=missing or tr("nothing, everything is installed")))
    stamp = _read_stamp(root)
    print(tr("env fingerprint: {fp}").format(fp=stamp.get("fingerprint", "—")))
    try:
        free = shutil.disk_usage(root).free / (1024 ** 3)
        print(tr("Free on FS   : {free:.1f} GiB").format(free=free))
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    if "--doctor" in sys.argv:
        sys.exit(doctor())
    sys.exit(run())
