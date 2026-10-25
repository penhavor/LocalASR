#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_app.py — start the LocalASR GUI.

This file is launched by the project environment interpreter (pythonw.exe on
Windows, hence no console window). If called with a "foreign" Python, it
delegates to bootstrap.py, which prepares the environment.

Headless mode is supported too:
    python launcher/run_app.py transcribe "lecture 1.mp4" --model large-v3-turbo
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # <root>/launcher
ROOT = os.path.dirname(HERE)                       # project root
for _entry in (ROOT, HERE):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    import bootstrap

    bootstrap.init_i18n(ROOT)
    _tr = bootstrap.tr
except Exception:  # noqa: BLE001
    def _tr(msgid: str) -> str:
        return msgid


def _inside_project_env() -> bool:
    """Running from the project environment (.venv or portable runtime)."""
    exe = os.path.abspath(sys.executable)
    for env in (os.path.join(ROOT, ".venv"), os.path.join(ROOT, "runtime", "python")):
        try:
            if os.path.commonpath([exe, os.path.abspath(env)]) == os.path.abspath(env):
                return True
        except ValueError:
            continue
    return False


def _has_console() -> bool:
    if os.name != "nt":
        return sys.stdout is not None
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:  # noqa: BLE001
        return sys.stdout is not None


def _relaunch_visible_console() -> int:
    """Broken environment + no console (pythonw): reopen main.py in a visible
    console so the user sees the repair flow instead of a silent failure."""
    exe = sys.executable
    candidates = []
    if os.name == "nt" and os.path.basename(exe).lower() == "pythonw.exe":
        candidates.append(os.path.join(os.path.dirname(exe), "python.exe"))
    candidates.append(exe)
    main_py = os.path.join(HERE, "main.py")
    for python in candidates:
        if not os.path.isfile(python):
            continue
        try:
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
                subprocess.Popen([python, main_py], cwd=ROOT, creationflags=flags, close_fds=True)
            else:
                subprocess.Popen([python, main_py], cwd=ROOT, start_new_session=True, close_fds=True)
            return 0
        except OSError:
            continue
    return 1


def main() -> int:
    headless = len(sys.argv) > 1 and sys.argv[1] in {
        "transcribe", "doctor", "models", "list-models", "version",
    }

    if not _inside_project_env() and not headless:
        # wrong environment — let bootstrap prepare everything and relaunch us
        import bootstrap

        return bootstrap.run(root=ROOT, argv=sys.argv[1:])

    try:
        from app import cli
    except Exception as exc:  # noqa: BLE001
        # The user may have deleted program files — check the manifest and
        # offer to restore them from GitHub.
        import bootstrap

        still_missing: list = []
        try:
            still_missing = bootstrap.check_project_files(ROOT)
            if still_missing:
                bootstrap.repair_missing_files(ROOT)
                still_missing = bootstrap.check_project_files(ROOT)
        except Exception:  # noqa: BLE001
            pass
        if not still_missing:
            try:
                import importlib

                for name in list(sys.modules):
                    if name == "app" or name.startswith("app."):
                        del sys.modules[name]
                importlib.invalidate_caches()
                from app import cli  # noqa: F811

                return cli.main(sys.argv[1:])
            except Exception as exc2:  # noqa: BLE001
                exc = exc2

        if not _has_console() and not headless:
            # invisible failure is worse than a popped-up console
            return _relaunch_visible_console()

        sys.stderr.write(
            "\n[LocalASR] " + _tr("Failed to load the application:\n")
            + f"  {type(exc).__name__}: {exc}\n\n"
        )
        if still_missing:
            sys.stderr.write(
                _tr("The program folder lacks files:\n  ")
                + "\n  ".join(still_missing[:20])
                + _tr("\n\nRestore them (Tools → Diagnostics) or download the project again.\n")
            )
        else:
            sys.stderr.write(
                _tr("Delete the environment folder and run launcher/main.py again — it will be recreated.\n")
            )
        try:
            input(_tr("Press Enter to exit..."))
        except (EOFError, KeyboardInterrupt, OSError):
            pass
        return 1

    return cli.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
