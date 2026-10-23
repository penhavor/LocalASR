#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalASR — точка входа.

Запусти этот файл (двойным кликом или `python main.py`) — дальше всё само:

1. Проверяется, что мы внутри виртуального окружения проекта (`.venv`).
2. Проверяется наличие всех библиотек, нужных для запуска GUI.
3. Чего не хватает — доустанавливается (только в `.venv`, системный Python не трогается).
4. Приложение перезапускается уже БЕЗ консоли (на Windows — через `pythonw.exe`).

Сам файл намеренно написан только на стандартной библиотеке: на момент первого
запуска в системе может не быть вообще ничего, кроме Python.
"""

from __future__ import annotations

import os
import sys
import traceback

# --- ранняя инициализация: кодировки, пути, переменные окружения ----------------
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
# Qt: корректный рендер на HiDPI и безглючная работа в RDP/VM
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
# HF Hub может ругаться на отсутствие кэша — пусть пишет в папку проекта
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _fatal(title: str, message: str, exc: BaseException | None = None) -> int:
    """Показать ошибку максимально заметно и не дать окну консоли мгновенно закрыться."""
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

    try:  # Tkinter входит в состав CPython — обычно доступен даже в голой системе
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
        tk.Button(btns, text="Закрыть", command=top.destroy).pack(side="right")
        messagebox.showinfo(title, f"Подробности сохранены в:\n{log_path}", parent=top)
        top.deiconify()
        top.protocol("WM_DELETE_WINDOW", root.destroy)
        root.mainloop()
    except Exception:
        pass

    print("\n" + "=" * 78, file=sys.stderr)
    print(f"[LocalASR] {title}", file=sys.stderr)
    print(details, file=sys.stderr)
    print(f"\nЛог: {log_path}", file=sys.stderr)
    print("=" * 78, file=sys.stderr)

    # Если запуск был двойным кликом, консоль закроется вместе с процессом и
    # пользователь не успеет ничего прочитать. Поэтому держим её до нажатия Enter
    # (если stdin вообще доступен; при запуске из pythonw его нет).
    if not os.environ.get("LOCALASR_NO_PAUSE"):
        try:
            input("\nНажми Enter, чтобы закрыть это окно...")
        except (EOFError, KeyboardInterrupt, OSError, ValueError):
            pass
        except Exception:  # noqa: BLE001
            pass
    return 1


def main() -> int:
    try:
        import bootstrap  # локальный модуль, только stdlib
    except Exception as exc:  # pragma: no cover - защита от битой установки
        return _fatal("Не найден bootstrap.py", "Файл bootstrap.py должен лежать рядом с main.py.", exc)

    try:
        return bootstrap.run(root=ROOT)
    except SystemExit as exc:  # корректный выход (например, пользователь отменил установку)
        return int(exc.code or 0)
    except KeyboardInterrupt:
        print("\n[LocalASR] Прервано пользователем.")
        return 130
    except Exception as exc:
        return _fatal("Ошибка при запуске LocalASR", "Не удалось подготовить окружение.", exc)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
