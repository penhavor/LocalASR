#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LocalASR — entry point.

Run this file (double click or `python launcher/main.py`) — everything else
is automatic:

1. The portable runtime/python is used when present (RUN.bat downloads it on
   the first run); otherwise a system Python is located and a project `.venv`
   is created.
2. All libraries needed by the GUI are checked.
3. Anything missing is installed (strictly into the project environment).
4. The app is relaunched WITHOUT a console (pythonw.exe on Windows).

This file intentionally uses the standard library only: at first launch the
system may have nothing but Python.
"""

from __future__ import annotations

import os
import sys
import traceback

# --- early initialization: encodings, paths, environment ---------------------
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
# Qt: correct HiDPI rendering and sane behavior in RDP/VM
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
# HF Hub may complain about a missing cache — let it write into the project
HERE = os.path.dirname(os.path.abspath(__file__))  # <root>/launcher
ROOT = os.path.dirname(HERE)                       # project root
for _entry in (ROOT, HERE):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

try:
    import bootstrap

    bootstrap.init_i18n(ROOT)
    _tr = bootstrap.tr
except Exception:  # noqa: BLE001 - bootstrap missing/broken: English fallback
    def _tr(msgid: str) -> str:
        return msgid


def _fatal(title: str, message: str, exc: BaseException | None = None) -> int:
    """Show the error prominently and keep the console from closing instantly."""
    details = message
    if exc is not None:
        details += "\n\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    log_path = os.path.join(ROOT, "logs", "bootstrap_error.log")
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("=" * 78 + "\n")
            fh.write(details + "\n")
    except OSError:
        pass

    try:  # Tkinter ships with CPython — usually available even on a bare system
        import tkinter as tk
        from tkinter import messagebox, scrolledtext

        root = tk.Tk()
        root.withdraw()
        top = tk.Toplevel(root)
        top.title(title)
        top.geometry("760x460")
        top.attributes("-topmost", True)
        tk.Label(top, text=message, justify="left", anchor="w", padx=12, pady=8).pack(fill="x")
        txt = scrolledtext.ScrolledText(top, wrap="word", height=20)
        txt.insert("1.0", details)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        btns = tk.Frame(top)
        btns.pack(fill="x", padx=12, pady=(0, 10))
        tk.Button(btns, text=_tr("Close"), command=top.destroy).pack(side="right")
        messagebox.showinfo(title, _tr("Details saved to:\n{path}").format(path=log_path), parent=top)
        top.deiconify()
        top.protocol("WM_DELETE_WINDOW", root.destroy)
        root.mainloop()
    except Exception:  # noqa: BLE001
        pass

    print("\n" + "=" * 78, file=sys.stderr)
    print(f"[LocalASR] {title}", file=sys.stderr)
    print(details, file=sys.stderr)
    print("\n" + _tr("Log: {path}").format(path=log_path), file=sys.stderr)
    print("=" * 78, file=sys.stderr)

    # On a double-click launch the console closes with the process before the
    # user can read anything — hold it until Enter (when stdin exists at all).
    if not os.environ.get("LOCALASR_NO_PAUSE"):
        try:
            input(_tr("\nPress Enter to close this window..."))
        except (EOFError, KeyboardInterrupt, OSError, ValueError):
            pass
        except Exception:  # noqa: BLE001
            pass
    return 1


def main() -> int:
    try:
        import bootstrap  # local module, stdlib only
    except Exception as exc:  # pragma: no cover - broken installation guard
        return _fatal(
            "bootstrap.py not found",
            "The file bootstrap.py must sit next to main.py. / "
            "Файл bootstrap.py должен лежать рядом с main.py.",
            exc,
        )

    try:
        return bootstrap.run(root=ROOT)
    except SystemExit as exc:  # clean exit (e.g. the user cancelled the install)
        return int(exc.code or 0)
    except KeyboardInterrupt:
        print("\n[LocalASR] " + _tr("Interrupted by user."))
        return 130
    except Exception as exc:
        return _fatal(
            _tr("Error while starting LocalASR"),
            _tr("Failed to prepare the environment."),
            exc,
        )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
