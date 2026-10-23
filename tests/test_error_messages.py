"""Проверка «человеческих» сообщений об ошибках — их пользователь видит глазами."""

from __future__ import annotations

from app import cli


def test_hint_for_missing_system_qt_libraries() -> None:
    exc = ImportError("libglib-2.0.so.0: cannot open shared object file: No such file or directory")
    hint = cli._pyside_import_hint(exc)
    assert "СИСТЕМНЫХ библиотек" in hint
    assert "apt install" in hint
    assert "libglib2.0-0" in hint
    assert "QT_OPENGL=software" in hint


def test_hint_for_windows_dll_problem() -> None:
    exc = ImportError("DLL load failed while importing QtCore: не найден указанный модуль")
    hint = cli._pyside_import_hint(exc)
    assert "СИСТЕМНЫХ библиотек" in hint


def test_hint_for_not_installed_pyside() -> None:
    exc = ModuleNotFoundError("No module named 'PySide6'")
    hint = cli._pyside_import_hint(exc)
    assert "не установлен" in hint.lower()
    assert ".venv" in hint


def test_hint_generic_fallback() -> None:
    hint = cli._pyside_import_hint(RuntimeError("что-то странное"))
    assert "Не удалось загрузить PySide6" in hint
    assert "main.py" in hint


def test_hint_always_contains_original_error() -> None:
    exc = ImportError("libEGL.so.1: cannot open shared object file")
    hint = cli._pyside_import_hint(exc)
    assert "libEGL.so.1" in hint
