"""Совместимость перечислений Qt между разными версиями PySide6.

В PySide6 6.4+ «плоские» алиасы (`Qt.AlignCenter`) помечены устаревшими,
в 6.8+ часть из них убрана, а в старых сборках наоборот нет вложенных scope'ов.
Хуже того: некоторые перечисления живут НЕ в Qt, а в своих классах
(например, `QDockWidget.DockWidgetFeature`), и попытка взять их из Qt молча
даёт «значение по умолчанию», которое строгий PySide6 потом отбивает
TypeError'ом прямо посреди создания окна.

Поэтому здесь:
  1. каждое значение ищется по цепочке владельцев (класс.Enum.член → класс.член);
  2. если владелец не нашёлся — значение КОНСТРУИРУЕТСЯ из enum-класса по числу
     (enum-классы в PySide6 принимают int), а не подставляется голой int-константой;
  3. комбинации флагов собираются оператором | над самими enum'ами, чтобы тип
     результата совпадал с ожидаемым сигнатурой Qt.

"""

from __future__ import annotations

import warnings
from typing import Any, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning, module="PySide6")

from PySide6.QtCore import Qt  # noqa: E402


# --------------------------------------------------------------------------- #
#  Базовые помощники
# --------------------------------------------------------------------------- #


def _owner(*names: str) -> Any:
    """Импортирует класс Qt по имени ('QtWidgets.QDockWidget' и т.п.)."""
    module_name, _, class_name = names[0].partition(".")
    module = __import__(f"PySide6.{module_name}", fromlist=[class_name])
    return getattr(module, class_name)


def _enum(scope: Any, *names: str, default: Any = None) -> Any:
    """Достаёт значение перечисления по цепочке имён ('EnumName.member', 'member')."""
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"_enum: имена перечислений должны быть строками, получено {name!r} "
                "(числовое значение передавай как default=…)"
            )
        target: Any = scope
        ok = True
        for part in name.split("."):
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                ok = False
                break
        if ok:
            return target
    return default


def _flag(owner_spec: str, enum_name: str, member: str, value: int) -> Any:
    """Гарантированно возвращает ЭНУМ-флаг, а не int.

    Порядок: класс.Enum.член → класс.член → класс.Enum(int) → int (крайний случай).
    """
    owner = _owner(owner_spec)
    found = _enum(owner, f"{enum_name}.{member}", member, default=None)
    if found is not None:
        return found
    enum_cls = getattr(owner, enum_name, None)
    if enum_cls is not None:
        try:
            return enum_cls(value)
        except (ValueError, TypeError):
            pass
    return value


def _combine(*values: Any) -> Any:
    result = values[0]
    for item in values[1:]:
        result = result | item
    return result


# --------------------------------------------------------------------------- #
#  Выравнивание / ориентация (Qt namespace — стабильно во всех версиях)
# --------------------------------------------------------------------------- #

ALIGN_LEFT = _enum(Qt, "AlignmentFlag.AlignLeft", "AlignLeft", default=0x0001)
ALIGN_RIGHT = _enum(Qt, "AlignmentFlag.AlignRight", "AlignRight", default=0x0002)
ALIGN_TOP = _enum(Qt, "AlignmentFlag.AlignTop", "AlignTop", default=0x0020)
ALIGN_BOTTOM = _enum(Qt, "AlignmentFlag.AlignBottom", "AlignBottom", default=0x0040)
ALIGN_HCENTER = _enum(Qt, "AlignmentFlag.AlignHCenter", "AlignHCenter", default=0x0004)
ALIGN_VCENTER = _enum(Qt, "AlignmentFlag.AlignVCenter", "AlignVCenter", default=0x0080)
ALIGN_CENTER = _enum(Qt, "AlignmentFlag.AlignCenter", "AlignCenter",
                     default=_combine(ALIGN_VCENTER, ALIGN_HCENTER))
ALIGN_RIGHT_VCENTER = _combine(ALIGN_RIGHT, ALIGN_VCENTER)
ALIGN_LEFT_VCENTER = _combine(ALIGN_LEFT, ALIGN_VCENTER)

HORIZONTAL = _enum(Qt, "Orientation.Horizontal", "Horizontal", default=0x1)
VERTICAL = _enum(Qt, "Orientation.Vertical", "Vertical", default=0x2)

# --------------------------------------------------------------------------- #
#  Окна, доки, курсоры
# --------------------------------------------------------------------------- #

WINDOW_CONTEXT_HELP_HINT = _enum(
    Qt, "WindowType.WindowContextHelpButtonHint", "WindowContextHelpButtonHint", 0x00000400
)

# ВАЖНО: DockWidgetArea живёт в Qt, а DockWidgetFeature — в QDockWidget.
BOTTOM_DOCK_AREA = _flag("QtCore.Qt", "DockWidgetArea", "BottomDockWidgetArea", 0x8)
TOP_DOCK_AREA = _flag("QtCore.Qt", "DockWidgetArea", "TopDockWidgetArea", 0x4)
DOCK_CLOSABLE = _flag("QtWidgets.QDockWidget", "DockWidgetFeature", "DockWidgetClosable", 0x1)
DOCK_MOVABLE = _flag("QtWidgets.QDockWidget", "DockWidgetFeature", "DockWidgetMovable", 0x2)
DOCK_FEATURES_DEFAULT = _combine(DOCK_CLOSABLE, DOCK_MOVABLE)

CURSOR_POINTING_HAND = _flag("QtCore.Qt", "CursorShape", "PointingHandCursor", 13)

TEXT_SELECTABLE_BY_MOUSE = _flag("QtCore.Qt", "TextInteractionFlag", "TextSelectableByMouse", 0x4)
TEXT_FORMAT_RICH = _flag("QtCore.Qt", "TextFormat", "RichText", 1)

SCROLLBAR_ALWAYS_OFF = _flag("QtCore.Qt", "ScrollBarPolicy", "ScrollBarAlwaysOff", 1)
SCROLLBAR_AS_NEEDED = _flag("QtCore.Qt", "ScrollBarPolicy", "ScrollBarAsNeeded", 0)
SCROLLBAR_ALWAYS_ON = _flag("QtCore.Qt", "ScrollBarPolicy", "ScrollBarAlwaysOn", 2)

KEEP_ASPECT_SMOOTH = _enum(Qt, "AspectRatioMode.KeepAspectRatio", "KeepAspectRatio", default=0x1)
SMOOTH_TRANSFORM = _enum(
    Qt, "TransformationMode.SmoothTransformation", "SmoothTransformation", default=0x1
)

# --------------------------------------------------------------------------- #
#  Цвета
# --------------------------------------------------------------------------- #

COLOR_GREEN = _enum(Qt, "GlobalColor.green", "green", default=8)
COLOR_RED = _enum(Qt, "GlobalColor.red", "red", default=7)
COLOR_DIM = _enum(Qt, "GlobalColor.gray", "gray", default=5)
COLOR_TRANSPARENT = _enum(Qt, "GlobalColor.transparent", "transparent", default=19)

# --------------------------------------------------------------------------- #
#  Роли данных / прочее из Qt
# --------------------------------------------------------------------------- #

ROLE_USER = _enum(Qt, "ItemDataRole.UserRole", "UserRole", default=0x0100)

# Флаги/состояния чекбоксов в списках (QListWidgetItem.setCheckState и др.)
ITEM_USER_CHECKABLE = _flag("QtCore.Qt", "ItemFlag", "ItemIsUserCheckable", 0x0010)
CHECK_UNCHECKED = _enum(Qt, "CheckState.Unchecked", "Unchecked", default=0)
CHECK_CHECKED = _enum(Qt, "CheckState.Checked", "Checked", default=2)


# --------------------------------------------------------------------------- #
#  Аттрибуты приложения
# --------------------------------------------------------------------------- #


def _app_attribute(name: str, value: int) -> Any:
    """AA_* живут в Qt.ApplicationAttribute; QApplication их лишь дублирует."""
    for owner_spec, enum_name in (
        ("QtCore.Qt", "ApplicationAttribute"),
        ("QtWidgets.QApplication", "Attribute"),
    ):
        owner = _owner(owner_spec)
        found = _enum(owner, f"{enum_name}.{name}", name, default=None)
        if found is not None:
            return found
        enum_cls = getattr(owner, enum_name, None)
        if enum_cls is not None:
            try:
                return enum_cls(value)
            except (ValueError, TypeError):
                continue
    return None


AA_SHARE_OPENGL_CONTEXTS = _app_attribute("AA_ShareOpenGLContexts", 0x20)
AA_DONT_CREATE_NATIVE_WIDGET_SIBLINGS = _app_attribute("AA_DontCreateNativeWidgetSiblings", 0x24)
AA_ENABLE_HIGHDPI_SCALING = _app_attribute("AA_EnableHighDpiScaling", 0x14)


# --------------------------------------------------------------------------- #
#  Виджеты: заголовки, списки, фреймы
# --------------------------------------------------------------------------- #


def header_stretch_mode() -> Any:
    return _flag("QtWidgets.QHeaderView", "ResizeMode", "Stretch", 1)


def selection_behaviour_rows() -> Any:
    return _flag("QtWidgets.QAbstractItemView", "SelectionBehavior", "SelectRows", 1)


def no_edit_triggers() -> Any:
    return _flag("QtWidgets.QAbstractItemView", "EditTrigger", "NoEditTriggers", 0)


def frame_no_frame() -> Any:
    return _flag("QtWidgets.QFrame", "Shape", "NoFrame", 0)


# --------------------------------------------------------------------------- #
#  Диалоги: кнопки и коды результата
# --------------------------------------------------------------------------- #

_MESSAGE_BOX_VALUES = {"Ok": 0x00000400, "Cancel": 0x00400000, "Yes": 0x00004000, "No": 0x00010000}
_DIALOG_BUTTON_VALUES = {
    "Ok": 0x00000400,
    "Save": 0x00000800,
    "Cancel": 0x00400000,
    "Close": 0x00200000,
    "Yes": 0x00004000,
    "No": 0x00010000,
}


def _standard_button(owner_spec: str, enum_name: str, name: str, values: dict) -> Any:
    owner = _owner(owner_spec)
    found = _enum(owner, f"{enum_name}.{name}", name, default=None)
    if found is not None:
        return found
    enum_cls = getattr(owner, enum_name, None)
    if enum_cls is not None and name in values:
        try:
            return enum_cls(values[name])
        except (ValueError, TypeError):
            pass
    raise RuntimeError(f"Не удалось получить {owner_spec}.{enum_name}.{name} — проверь версию PySide6")


def message_box_yes() -> Any:
    return _standard_button("QtWidgets.QMessageBox", "StandardButton", "Yes", _MESSAGE_BOX_VALUES)


def message_box_no() -> Any:
    return _standard_button("QtWidgets.QMessageBox", "StandardButton", "No", _MESSAGE_BOX_VALUES)


def message_box_ok() -> Any:
    return _standard_button("QtWidgets.QMessageBox", "StandardButton", "Ok", _MESSAGE_BOX_VALUES)


def dialog_buttons(*names: str) -> Any:
    """QDialogButtonBox.Ok | Cancel и т.п. — возвращает ТИПИЗИРОВАННЫЕ флаги."""
    result: Optional[Any] = None
    for name in names:
        value = _standard_button(
            "QtWidgets.QDialogButtonBox", "StandardButton", name, _DIALOG_BUTTON_VALUES
        )
        result = value if result is None else (result | value)
    if result is None:
        raise ValueError("dialog_buttons() вызван без имён")
    return result


def dialog_accepted() -> Any:
    """Код результата QDialog.Accepted (enum-член, если доступен)."""
    owner = _owner("QtWidgets.QDialog")
    found = _enum(owner, "DialogCode.Accepted", "Accepted", default=None)
    if found is not None:
        return found
    enum_cls = getattr(owner, "DialogCode", None)
    if enum_cls is not None:
        try:
            return enum_cls(1)
        except (ValueError, TypeError):
            pass
    return 1
