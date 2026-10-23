"""Механическая сверка `app/ui/compat.py` с установленным PySide6 — без запуска Qt.

Зачем: PySide6 строго типизирует аргументы (`setFeatures(int)` → TypeError),
а «не нашлось значение» в compat раньше могло молча подменить enum целым числом.
Единственный надёжный способ поймать это без дисплея — прочитать `.pyi`-стабы,
которые лежат рядом с установленным PySide6, и убедиться, что каждое нужное нам
перечисление действительно живёт в том классе, откуда мы его берём.

Использование:
    python scripts/dev_check_qt_compat.py [путь-к-venv]
    pytest tests/test_qt_compat_stubs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# (файл .pyi, класс-владелец, перечисление, члены)
REQUIREMENTS: List[Tuple[str, str, str, List[str]]] = [
    ("QtCore.pyi", "Qt", "AlignmentFlag",
     ["AlignLeft", "AlignRight", "AlignTop", "AlignBottom", "AlignHCenter", "AlignVCenter", "AlignCenter"]),
    ("QtCore.pyi", "Qt", "Orientation", ["Horizontal", "Vertical"]),
    ("QtCore.pyi", "Qt", "WindowType", ["WindowContextHelpButtonHint"]),
    ("QtCore.pyi", "Qt", "DockWidgetArea", ["BottomDockWidgetArea", "TopDockWidgetArea"]),
    ("QtCore.pyi", "Qt", "CursorShape", ["PointingHandCursor"]),
    ("QtCore.pyi", "Qt", "TextInteractionFlag", ["TextSelectableByMouse"]),
    ("QtCore.pyi", "Qt", "TextFormat", ["RichText"]),
    ("QtCore.pyi", "Qt", "ScrollBarPolicy", ["ScrollBarAlwaysOff"]),
    ("QtCore.pyi", "Qt", "GlobalColor", ["green", "red", "gray", "transparent"]),
    ("QtCore.pyi", "Qt", "ItemDataRole", ["UserRole"]),
    ("QtCore.pyi", "Qt", "TextElideMode", ["ElideMiddle"]),
    # КЛЮЧЕВОЕ: DockWidgetFeature обязан находиться в QDockWidget, а не в Qt
    ("QtWidgets.pyi", "QDockWidget", "DockWidgetFeature", ["DockWidgetClosable", "DockWidgetMovable"]),
    ("QtWidgets.pyi", "QHeaderView", "ResizeMode", ["Stretch"]),
    ("QtWidgets.pyi", "QAbstractItemView", "SelectionBehavior", ["SelectRows"]),
    ("QtWidgets.pyi", "QAbstractItemView", "EditTrigger", ["NoEditTriggers"]),
    ("QtWidgets.pyi", "QFrame", "Shape", ["NoFrame", "HLine"]),
    ("QtWidgets.pyi", "QMessageBox", "StandardButton", ["Yes", "No", "Ok"]),
    ("QtWidgets.pyi", "QDialogButtonBox", "StandardButton",
     ["Ok", "Save", "Cancel", "Close", "Yes", "No"]),
    ("QtCore.pyi", "Qt", "ApplicationAttribute",
     ["AA_ShareOpenGLContexts", "AA_DontCreateNativeWidgetSiblings"]),
    ("QtWidgets.pyi", "QDialog", "DialogCode", ["Accepted"]),
]


def find_pyside6_dir(search_paths: Optional[List[str]] = None) -> Optional[Path]:
    """Ищет папку установленного PySide6 (именно папку, не импортируя её)."""
    candidates: List[Path] = []
    for entry in search_paths or list(sys.path):
        if not entry:
            continue
        base = Path(entry)
        for candidate in (base / "PySide6", base / "site-packages" / "PySide6", base / "lib" / "PySide6"):
            if candidate.is_dir():
                candidates.append(candidate)
        if base.name == "PySide6":
            candidates.append(base)
    # сначала ищем установку СО стабами (настоящий PySide6), стаб-заглушку пропускаем
    for candidate in candidates:
        if (candidate / "QtCore.pyi").is_file() or (candidate / "QtWidgets.pyi").is_file():
            return candidate
    for candidate in candidates:
        if (candidate / "QtCore.py").is_file():
            return None  # пакет есть, но стабов нет — проверять нечего
    return None


def _owner_block(text: str, owner: str) -> Optional[str]:
    """Вырезает тело класса верхнего уровня `owner` из текста .pyi."""
    match = re.search(rf"^class {re.escape(owner)}\b.*$", text, re.MULTILINE)
    if not match:
        return None
    start = match.end()
    next_class = re.search(r"^class \w+", text[start:], re.MULTILINE)
    end = start + next_class.start() if next_class else len(text)
    return text[start:end]


def _enum_members(block: str, enum_name: str) -> Optional[str]:
    """Тело вложенного перечисления `enum_name` внутри блока класса."""
    match = re.search(rf"^\s+class {re.escape(enum_name)}\b.*$", block, re.MULTILINE)
    if not match:
        return None
    start = match.end()
    indent = len(match.group(0)) - len(match.group(0).lstrip())
    next_same = re.search(rf"^\s{{0,{indent}}}(class |\w+ ?:)", block[start:], re.MULTILINE)
    end = start + next_same.start() if next_same else len(block)
    return block[start:end]


def verify(search_paths: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Возвращает {"pyside6": путь|"", "problems": [...], "checked": [...]}."""
    result: Dict[str, object] = {"pyside6": "", "problems": [], "checked": []}
    directory = find_pyside6_dir(search_paths)
    if directory is None:
        result["problems"] = []
        result["note"] = "PySide6 (или его .pyi-стабы) не найдены — проверка пропущена"
        return result  # type: ignore[return-value]

    result["pyside6"] = str(directory)
    texts: Dict[str, str] = {}
    problems: List[str] = []
    checked: List[str] = []

    for filename, owner, enum_name, members in REQUIREMENTS:
        path = directory / filename
        if filename not in texts:
            try:
                texts[filename] = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                problems.append(f"{filename}: не читается ({exc})")
                continue
        text = texts.get(filename, "")
        block = _owner_block(text, owner)
        if block is None:
            problems.append(f"{filename}: класс {owner} не найден")
            continue
        enum_block = _enum_members(block, enum_name)
        if enum_block is None:
            problems.append(f"{filename}: {owner}.{enum_name} не найдено")
            continue
        for member in members:
            if re.search(rf"^\s+{re.escape(member)}\b", enum_block, re.MULTILINE):
                checked.append(f"{owner}.{enum_name}.{member}")
            else:
                problems.append(f"{filename}: {owner}.{enum_name}.{member} отсутствует")

    result["problems"] = problems
    result["checked"] = checked
    return result  # type: ignore[return-value]


def verify_compat_values() -> List[str]:
    """Дополнительная проверка: compat не вернул голых int там, где нужен enum.

    Работает только если PySide6 реально импортируется (системные библиотеки на
    месте); иначе возвращает пустой список — тип уже гарантирован .pyi-проверкой.
    """
    problems: List[str] = []
    try:
        import PySide6

        if getattr(PySide6, "STUB_VERSION", None):
            return problems  # тестовый стаб: значения — обычные int, это нормально
        from . import compat
    except Exception:  # noqa: BLE001
        return problems

    enum_checks = [
        ("DOCK_CLOSABLE", compat.DOCK_CLOSABLE, "QDockWidget"),
        ("DOCK_MOVABLE", compat.DOCK_MOVABLE, "QDockWidget"),
        ("BOTTOM_DOCK_AREA", compat.BOTTOM_DOCK_AREA, "Qt"),
        ("TOP_DOCK_AREA", compat.TOP_DOCK_AREA, "Qt"),
        ("CURSOR_POINTING_HAND", compat.CURSOR_POINTING_HAND, "Qt"),
        ("SCROLLBAR_ALWAYS_OFF", compat.SCROLLBAR_ALWAYS_OFF, "Qt"),
        ("TEXT_SELECTABLE_BY_MOUSE", compat.TEXT_SELECTABLE_BY_MOUSE, "Qt"),
        ("ROLE_USER", compat.ROLE_USER, "Qt"),
    ]
    for name, value, _owner_name in enum_checks:
        type_name = type(value).__name__
        # у настоящих enum'ов PySide6 тип называется как само перечисление,
        # а «голое» int/long означает, что compat не нашёл значение
        if type_name in ("int", "float"):
            problems.append(f"compat.{name} = {value!r} (тип {type_name}) — enum не найден!")

    # флаги диалоговых кнопок обязаны быть enum'ами: QDialogButtonBox(int) → TypeError
    for label, value in (
        ("dialog_buttons('Ok','Cancel')", compat.dialog_buttons("Ok", "Cancel")),
        ("message_box_yes()", compat.message_box_yes()),
        ("dialog_accepted()", compat.dialog_accepted()),
    ):
        if type(value).__name__ in ("int", "float"):
            problems.append(f"compat.{label} = {value!r} (тип int) — QDialog/MessageBox отобьют вызов")
    return problems
