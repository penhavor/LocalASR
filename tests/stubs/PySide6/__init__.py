"""Тестовый стаб PySide6.

Нужен ТОЛЬКО для тестов в окружении, где Qt не установлен (CI, песочница):
позволяет импортировать и частично исполнять UI-код, чтобы отлавливать
опечатки, неверные сигнатуры и сломанные связи сигнал/слот.

В рабочей программе всегда используется настоящий PySide6 из .venv —
этот каталог подключается к sys.path исключительно из tests/conftest.py
и только если настоящий PySide6 не импортируется.
"""

STUB_VERSION = "0.0.0-stub"


class _QtNamespace:
    """Мини-замена PySide6.QtCore.Qt: только те флаги, что использует код."""

    class AlignFlag:
        AlignLeft = 1
        AlignRight = 2
        AlignHCenter = 4
        AlignTop = 32
        AlignBottom = 64
        AlignVCenter = 128
        AlignCenter = 132

    class Orientation:
        Horizontal = 1
        Vertical = 2

    class WindowType:
        Window = 1
        Dialog = 3
        Popup = 8
        Tool = 10
        WindowContextHelpButtonHint = 16
        FramelessWindowHint = 2048

    class DockWidgetArea:
        BottomDockWidgetArea = 8
        TopDockWidgetArea = 4
        LeftDockWidgetArea = 1
        RightDockWidgetArea = 2

    class DockWidgetFeature:
        DockWidgetClosable = 1
        DockWidgetMovable = 2
        DockWidgetFloatable = 4

    class ItemDataRole:
        UserRole = 256
        DisplayRole = 0

    class TextInteractionFlag:
        TextSelectableByMouse = 4
        NoTextInteraction = 0

    class TextFormat:
        PlainText = 0
        RichText = 1
        AutoText = 2

    class ScrollBarPolicy:
        ScrollBarAsNeeded = 0
        ScrollBarAlwaysOff = 1
        ScrollBarAlwaysOn = 2

    class SelectionMode:
        NoSelection = 0
        SingleSelection = 1
        MultiSelection = 2
        ExtendedSelection = 3

    class GlobalColor:
        white = 3
        black = 2
        red = 7
        green = 8
        blue = 9
        transparent = 19

    class FocusPolicy:
        NoFocus = 0
        StrongFocus = 11

    class SortOrder:
        AscendingOrder = 0
        DescendingOrder = 1

    class CursorShape:
        ArrowCursor = 0
        PointingHandCursor = 13

    class CheckState:
        Unchecked = 0
        PartiallyChecked = 1
        Checked = 2

    # плоские алиасы (в PySide6 доступны и так, и так)
    AlignLeft = AlignFlag.AlignLeft
    AlignRight = AlignFlag.AlignRight
    AlignHCenter = AlignFlag.AlignHCenter
    AlignTop = AlignFlag.AlignTop
    AlignBottom = AlignFlag.AlignBottom
    AlignVCenter = AlignFlag.AlignVCenter
    AlignCenter = AlignFlag.AlignCenter
    Horizontal = Orientation.Horizontal
    Vertical = Orientation.Vertical
    WindowContextHelpButtonHint = WindowType.WindowContextHelpButtonHint
    BottomDockWidgetArea = DockWidgetArea.BottomDockWidgetArea
    TopDockWidgetArea = DockWidgetArea.TopDockWidgetArea
    DockWidgetClosable = DockWidgetFeature.DockWidgetClosable
    DockWidgetMovable = DockWidgetFeature.DockWidgetMovable
    UserRole = ItemDataRole.UserRole
    TextSelectableByMouse = TextInteractionFlag.TextSelectableByMouse
    RichText = TextFormat.RichText
    ScrollBarAlwaysOff = ScrollBarPolicy.ScrollBarAlwaysOff
    ExtendedSelection = SelectionMode.ExtendedSelection
    PointingHandCursor = CursorShape.PointingHandCursor
    green = GlobalColor.green
    red = GlobalColor.red
    transparent = GlobalColor.transparent


Qt = _QtNamespace()
