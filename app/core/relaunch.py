"""Restart the application process (used after a UI language change)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .. import paths
from ..i18n import _
from ..logutils import get_logger

log = get_logger("relaunch")


def _gui_python() -> str:
    """Prefer the windowless interpreter next to the current one (Windows)."""
    exe = Path(sys.executable)
    if os.name == "nt" and exe.name.lower() == "python.exe":
        candidate = exe.with_name("pythonw.exe")
        if candidate.is_file():
            return str(candidate)
    return str(exe)


def restart_app() -> bool:
    """Spawn a detached copy of the app; the caller should exit right after."""
    root = paths.ROOT_DIR
    entry = root / "launcher" / "run_app.py"
    if not entry.is_file():
        entry = root / "launcher" / "main.py"
    env = os.environ.copy()
    env["LOCALASR_ROOT"] = str(root)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    cmd = [_gui_python(), str(entry)]
    kwargs: dict = {
        "cwd": str(root),
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        kwargs["creationflags"] = detached | new_group
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(cmd, **kwargs)
    except OSError as exc:
        log.warning(_("Restart failed: %s"), exc)
        return False
    log.info(_("Restarting the application"))
    return True
