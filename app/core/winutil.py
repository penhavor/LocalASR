"""Windows-специфика для GUI-процесса: невидимые дочерние консоли.

Проблема: приложение работает под pythonw.exe (без консоли). Когда ЛЮБАЯ
библиотека внутри процесса запускает консольную программу (ffmpeg, nvidia-smi,
pip…) через subprocess БЕЗ флага CREATE_NO_WINDOW, Windows создаёт для неё
НОВОЕ консольное окно — оно мигает в панели задач и крадёт фокус.

Свой код мы запускаем аккуратно (creationflags везде проставлены), но чужой —
нет: например, modeling_gigaam.py дёргает ffmpeg напрямую. Поэтому при старте
GUI (и только если у процесса действительно нет консоли) мы добавляем
CREATE_NO_WINDOW всем subprocess.Popen в процессе.

Флаги DETACHED_PROCESS / CREATE_NEW_CONSOLE / CREATE_NO_WINDOW взаимоисключающие —
если вызывающий уже выставил один из них, мы не вмешиваемся.
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
        return True  # не смогли определить — считаем, что консоль есть (не мешаем)


def suppress_child_consoles() -> bool:
    """Делает все дочерние консольные процессы невидимыми (только Windows+pythonw).

    Возвращает True, если патч применён. Идемпотентна.
    """
    if os.name != "nt":
        return False
    if getattr(subprocess.Popen, "_localasr_no_window", False):
        return True
    if _has_console():
        # процесс с консолью (запуск из терминала/батника) — окна детей
        # наследуют существующую консоль и не мигают; ничего не трогаем
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
