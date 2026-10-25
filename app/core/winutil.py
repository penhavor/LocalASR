"""Windows specifics for the GUI process: invisible child consoles.

The app runs under pythonw.exe (no console). When ANY library spawns a
console program (ffmpeg, nvidia-smi, pip…) via subprocess WITHOUT
CREATE_NO_WINDOW, Windows creates a new console window that flashes in the
taskbar and steals focus. Our own code sets creationflags properly, but
third-party code (e.g. modeling_gigaam.py calling ffmpeg) does not — so at
GUI startup (only when the process really has no console) we add
CREATE_NO_WINDOW to every subprocess.Popen in the process.

DETACHED_PROCESS / CREATE_NEW_CONSOLE / CREATE_NO_WINDOW are mutually
exclusive: if the caller already set one of them, we do not interfere.
"""

from __future__ import annotations

import os
import subprocess

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
CREATE_NEW_CONSOLE = 0x00000010
_CREATE_EXCLUSIVE = CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_CONSOLE


def _has_console() -> bool:
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:  # noqa: BLE001
        return True  # cannot tell — assume a console exists (do not interfere)


def suppress_child_consoles() -> bool:
    """Make all child console processes invisible (Windows + pythonw only).

    Returns True when the patch was applied. Idempotent.
    """
    if os.name != "nt":
        return False
    if getattr(subprocess.Popen, "_localasr_no_window", False):
        return True
    if _has_console():
        # a process with a console (terminal/bat launch): children inherit it
        return False

    original_init = subprocess.Popen.__init__

    def patched_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        flags = kwargs.get("creationflags") or 0
        if not flags & _CREATE_EXCLUSIVE:
            kwargs["creationflags"] = flags | CREATE_NO_WINDOW
        original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init  # type: ignore[method-assign]
    subprocess.Popen._localasr_no_window = True  # type: ignore[attr-defined]
    return True
