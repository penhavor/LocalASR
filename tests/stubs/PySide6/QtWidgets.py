"""Стаб PySide6.QtWidgets для тестов (без настоящего Qt).

Реализовано ровно то поведение, на которое опирается код LocalASR:
сигналы, тексты, данные comboBox, элементы списков, раскладки. Это позволяет
в CI поймать опечатки, неверные сигнатуры и сломанную логику виджетов.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .QtCore import QObject, QSize, Signal
from .QtGui import QAction, QFont, QIcon


# --------------------------------------------------------------------------- #
#  Базовые виджеты
# --------------------------------------------------------------------------- #


class QWidget(QObject):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._init_signals()
        self._layout: Optional[Any] = None
        self._text = ""
        self._size_hint = QSize(100, 30)
        self._fixed_size: Optional[tuple] = None
        self._style_sheet = ""
        self._cursor = None
        self._font: Optional[QFont] = None

    # геометрия
    def setMinimumSize(self, width: int, height: int) -> None:
        self._min_size = (width, height)

    def setMinimumWidth(self, width: int) -> None:
        self._min_width = width

    def setMinimumHeight(self, height: int) -> None:
        self._min_height = height

    def setMaximumHeight(self, height: int) -> None:
        self._max_height = height

    def setFixedSize(self, width: int, height: int) -> None:
        self._fixed_size = (width, height)

    def setFixedWidth(self, width: int) -> None:
        self._fixed_size = (width, None)

    def setFixedHeight(self, height: int) -> None:
        self._fixed_size = (None, height)

    def resize(self, width: int, height: int) -> None:
        self._size = (width, height)

    def sizeHint(self) -> QSize:
        return self._size_hint

    def setSizePolicy(self, *args) -> None:
        self._size_policy = args

    # внешний вид
    def setStyleSheet(self, sheet: str) -> None:
        self._style_sheet = sheet

    def styleSheet(self) -> str:
        return self._style_sheet

    def setCursor(self, cursor: Any) -> None:
        self._cursor = cursor

    def setFont(self, font: QFont) -> None:
        self._font = font

    def font(self) -> QFont:
        return self._font or QFont()

    def setGraphicsEffect(self, effect: Any) -> None:
        self._effect = effect

    def setAcceptDrops(self, accept: bool) -> None:
        self._accept_drops = accept

    def setWindowTitle(self, title: str) -> None:
        self._window_title = title

    def windowTitle(self) -> str:
        return getattr(self, "_window_title", "")

    def setWindowIcon(self, icon: QIcon) -> None:
        self._icon = icon

    def setWindowFlag(self, flag: Any, on: bool = True) -> None:
        self._window_flags = getattr(self, "_window_flags", []) + [(flag, on)]

    # показ
    def show(self) -> None:
        self._visible = True

    def hide(self) -> None:
        self._visible = False

    def close(self) -> bool:
        self._visible = False
        return True

    def setFocus(self) -> None:
        return None

    def update(self) -> None:
        return None

    def repaint(self) -> None:
        return None

    def grab(self) -> Any:
        return None

    # раскладки
    def setLayout(self, layout: Any) -> None:
        self._layout = layout
        if isinstance(layout, _Layout):
            layout._owner = self
            for item in layout._items:
                layout._register_child(item.widget() or item.layout())

    def layout(self) -> Any:
        return self._layout


class QLabel(QWidget):
    def __init__(self, text: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self._text = text
        self._word_wrap = False
        self._interaction = None

    def setText(self, text: str) -> None:
        self._text = str(text)

    def text(self) -> str:
        return self._text

    def setWordWrap(self, enabled: bool) -> None:
        self._word_wrap = bool(enabled)

    def setAlignment(self, alignment: Any) -> None:
        self._alignment = alignment

    def setTextInteractionFlags(self, flags: Any) -> None:
        self._interaction = flags

    def setTextFormat(self, fmt: Any) -> None:
        self._format = fmt


class QPushButton(QWidget):
    clicked = Signal()
    toggled = Signal()      # есть у Qt6 для checkable-кнопок
    pressed = Signal()

    def __init__(self, text: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self._text = text
        self._checkable = False
        self._checked = False
        self._shortcut = None

    def setText(self, text: str) -> None:
        self._text = str(text)

    def text(self) -> str:
        return self._text

    def setCheckable(self, checkable: bool) -> None:
        self._checkable = bool(checkable)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        changed = self._checked != bool(checked)
        self._checked = bool(checked)
        if changed:
            self.toggled.emit(self._checked)

    def setShortcut(self, shortcut: Any) -> None:
        self._shortcut = shortcut

    def click(self) -> None:
        self.clicked.emit()


class QToolButton(QPushButton):
    def setAutoRaise(self, enabled: bool) -> None:
        self._auto_raise = enabled


class QCheckBox(QWidget):
    toggled = Signal()
    stateChanged = Signal()

    def __init__(self, text: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self._text = text
        self._checked = False

    def setText(self, text: str) -> None:
        self._text = str(text)

    def text(self) -> str:
        return self._text

    def setChecked(self, checked: bool) -> None:
        changed = self._checked != bool(checked)
        self._checked = bool(checked)
        if changed:
            self.toggled.emit(self._checked)

    def isChecked(self) -> bool:
        return self._checked

    def check(self) -> None:
        self.setChecked(True)


class QRadioButton(QCheckBox):
    pass


class QLineEdit(QWidget):
    textChanged = Signal()
    editingFinished = Signal()
    returnPressed = Signal()

    def __init__(self, text: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self._text = text
        self._placeholder = ""

    def setText(self, text: str) -> None:
        self._text = str(text)
        self.textChanged.emit(self._text)

    def text(self) -> str:
        return self._text

    def setPlaceholderText(self, text: str) -> None:
        self._placeholder = text

    def placeholderText(self) -> str:
        return self._placeholder

    def setClearButtonEnabled(self, enabled: bool) -> None:
        self._clear_button = enabled

    def setReadOnly(self, enabled: bool) -> None:
        self._readonly = enabled


class _TextEditBase(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._lines: List[str] = []
        self._placeholder = ""
        self._readonly = False
        self._max_blocks = 0

    def setPlainText(self, text: str) -> None:
        self._lines = str(text).split("\n")

    def toPlainText(self) -> str:
        return "\n".join(self._lines)

    def appendPlainText(self, text: str) -> None:
        self._lines.append(str(text))
        if self._max_blocks and len(self._lines) > self._max_blocks:
            self._lines = self._lines[-self._max_blocks:]

    def append(self, text: str) -> None:
        self.appendPlainText(text)

    def clear(self) -> None:
        self._lines = []

    def setReadOnly(self, enabled: bool) -> None:
        self._readonly = bool(enabled)

    def setPlaceholderText(self, text: str) -> None:
        self._placeholder = text

    def setMaximumBlockCount(self, count: int) -> None:
        self._max_blocks = count

    def verticalScrollBar(self) -> "_ScrollBar":
        return _ScrollBar()


class QPlainTextEdit(_TextEditBase):
    textChanged = Signal()


class QTextEdit(_TextEditBase):
    textChanged = Signal()

    def setText(self, text: str) -> None:
        self.setPlainText(text)

    def setHtml(self, html: str) -> None:
        self._lines = [html]


class _ScrollBar(QObject):
    def __init__(self) -> None:
        super().__init__(None)
        self._value = 0

    def setValue(self, value: int) -> None:
        self._value = value

    def value(self) -> int:
        return self._value

    def maximum(self) -> int:
        return 100


class QComboBox(QWidget):
    currentIndexChanged = Signal()
    currentTextChanged = Signal()
    activated = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._items: List[tuple[str, Any]] = []
        self._index = -1

    def addItem(self, text: str, data: Any = None) -> None:
        self._items.append((str(text), data))
        if self._index < 0:
            self._index = 0

    def addItems(self, items: List[str]) -> None:
        for item in items:
            self.addItem(item)

    def findData(self, data: Any) -> int:
        for index, (_, item_data) in enumerate(self._items):
            if item_data == data:
                return index
        return -1

    def setCurrentIndex(self, index: int) -> None:
        if -1 <= index < len(self._items) and index != self._index:
            self._index = index
            self.currentIndexChanged.emit(index)

    def currentIndex(self) -> int:
        return self._index

    def currentData(self) -> Any:
        if 0 <= self._index < len(self._items):
            return self._items[self._index][1]
        return None

    def currentText(self) -> str:
        if 0 <= self._index < len(self._items):
            return self._items[self._index][0]
        return ""

    def count(self) -> int:
        return len(self._items)

    def itemData(self, index: int) -> Any:
        return self._items[index][1]

    def clear(self) -> None:
        self._items = []
        self._index = -1


class _NumericBase(QWidget):
    valueChanged = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._value = 0
        self._range = (0, 99)
        self._suffix = ""
        self._special_text = ""

    def setRange(self, minimum: int, maximum: int) -> None:
        self._range = (minimum, maximum)

    def setMinimum(self, value: int) -> None:
        self._range = (value, self._range[1])

    def setMaximum(self, value: int) -> None:
        self._range = (self._range[0], value)

    def setValue(self, value: Any) -> None:
        try:
            cast = int if isinstance(self._range[0], int) else float
            value = cast(value)
        except (TypeError, ValueError):
            return
        minimum, maximum = self._range
        clamped = max(minimum, min(maximum, value))
        if clamped != self._value:
            self._value = clamped
            self.valueChanged.emit(clamped)
        else:
            self._value = clamped

    def value(self) -> Any:
        return self._value

    def setSuffix(self, suffix: str) -> None:
        self._suffix = suffix

    def setPrefix(self, prefix: str) -> None:
        self._prefix = prefix

    def setSpecialValueText(self, text: str) -> None:
        self._special_text = text

    def setSingleStep(self, step: Any) -> None:
        self._step = step

    def setDecimals(self, decimals: int) -> None:
        self._decimals = decimals


class QSpinBox(_NumericBase):
    pass


class QDoubleSpinBox(_NumericBase):
    pass


class QProgressBar(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._range = (0, 100)
        self._value = 0
        self._format = "%p%"

    def setRange(self, minimum: int, maximum: int) -> None:
        self._range = (minimum, maximum)

    def setValue(self, value: int) -> None:
        self._value = value

    def value(self) -> int:
        return self._value

    def setFormat(self, fmt: str) -> None:
        self._format = fmt

    def setMaximum(self, maximum: int) -> None:
        self._range = (self._range[0], maximum)


# --------------------------------------------------------------------------- #
#  Контейнеры и раскладки
# --------------------------------------------------------------------------- #


class QFrame(QWidget):
    class Shape:
        NoFrame = 0
        HLine = 4
        VLine = 5
        Box = 1
        StyledPanel = 2

    HLine = Shape.HLine
    VLine = Shape.VLine
    NoFrame = Shape.NoFrame

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)

    def setFrameShape(self, shape: Any) -> None:
        self._shape = shape


class QGroupBox(QFrame):
    def __init__(self, title: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self._title = title

    def setTitle(self, title: str) -> None:
        self._title = title

    def title(self) -> str:
        return self._title


class _LayoutItem:
    def __init__(self, obj: Any, stretch: int = 0) -> None:
        self._obj = obj
        self._stretch = stretch

    def widget(self) -> Any:
        return self._obj if isinstance(self._obj, QWidget) else None

    def layout(self) -> Any:
        return self._obj if isinstance(self._obj, _Layout) else None


class _Layout(QObject):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent if isinstance(parent, QWidget) else None)
        self._items: List[_LayoutItem] = []
        self._owner: Optional[QWidget] = None
        if isinstance(parent, QWidget):
            parent.setLayout(self)
        elif isinstance(parent, _Layout):
            # вложенная раскладка: наследуем владельца
            self._owner = parent.owner_widget()

    def owner_widget(self) -> Optional[QWidget]:
        owner = self._owner
        if owner is not None:
            return owner
        parent = self.parent()
        if isinstance(parent, _Layout):
            return parent.owner_widget()
        return parent if isinstance(parent, QWidget) else None

    def addWidget(self, widget: Any, stretch: int = 0, alignment: Any = None) -> None:
        self._items.append(_LayoutItem(widget, stretch))
        self._register_child(widget)

    def addLayout(self, layout: Any, stretch: int = 0) -> None:
        self._items.append(_LayoutItem(layout, stretch))
        self._register_child(layout)

    def _register_child(self, obj: Any) -> None:
        """Регистрируем виджет как ребёнка виджета-владельца — для findChildren."""
        if obj is None:
            return
        owner = self.owner_widget()
        if isinstance(obj, _Layout):
            if obj.owner_widget() is None:
                obj._owner = owner
            for nested in obj._items:
                self._register_child(nested.widget() or nested.layout())
            return
        if owner is None or not isinstance(obj, QWidget):
            return
        if not hasattr(owner, "_children"):
            owner._children = []
        if obj not in owner._children:
            owner._children.append(obj)
        if obj.parent() is None:
            obj.setParent(owner)

    def addStretch(self, stretch: int = 0) -> None:
        self._items.append(_LayoutItem(None, stretch))

    def addSpacing(self, space: int) -> None:
        self._items.append(_LayoutItem(None, 0))

    def addRow(self, *args) -> None:
        for arg in args:
            if arg is None:
                continue
            self._items.append(_LayoutItem(arg))

    def insertWidget(self, index: int, widget: Any, stretch: int = 0) -> None:
        self._items.insert(index, _LayoutItem(widget, stretch))
        self._register_child(widget)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[_LayoutItem]:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> Optional[_LayoutItem]:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def setContentsMargins(self, *args) -> None:
        self._margins = args

    def setSpacing(self, spacing: int) -> None:
        self._spacing = spacing

    def setAlignment(self, *args) -> None:
        self._alignment = args

    def widgets(self) -> List[QWidget]:
        result: List[QWidget] = []
        for item in self._items:
            widget = item.widget()
            if widget is not None:
                result.append(widget)
            nested = item.layout()
            if nested is not None:
                result.extend(nested.widgets())
        return result


class QVBoxLayout(_Layout):
    pass


class QHBoxLayout(_Layout):
    pass


class QGridLayout(_Layout):
    def setHorizontalSpacing(self, spacing: int) -> None:
        self._h_spacing = spacing

    def setVerticalSpacing(self, spacing: int) -> None:
        self._v_spacing = spacing

    def setColumnStretch(self, column: int, stretch: int) -> None:
        self._column_stretch = getattr(self, "_column_stretch", {})
        self._column_stretch[column] = stretch

    def setRowStretch(self, row: int, stretch: int) -> None:
        self._row_stretch = getattr(self, "_row_stretch", {})
        self._row_stretch[row] = stretch

    def addWidget(self, widget: Any, row: int = 0, column: int = 0, rowspan: int = 1, colspan: int = 1, alignment: Any = None) -> None:  # noqa: D102
        self._items.append(_LayoutItem(widget))
        self._register_child(widget)

    def addLayout(self, layout: Any, row: int = 0, column: int = 0, rowspan: int = 1, colspan: int = 1) -> None:  # noqa: D102
        self._items.append(_LayoutItem(layout))
        self._register_child(layout)


class QFormLayout(_Layout):
    def addRow(self, *args) -> None:
        for arg in args:
            if arg is None:
                continue
            self._items.append(_LayoutItem(arg))
            self._register_child(arg)

    def setFieldGrowthPolicy(self, policy: Any) -> None:
        self._growth = policy


class QScrollArea(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._widget: Optional[QWidget] = None

    def setWidget(self, widget: QWidget) -> None:
        self._widget = widget
        if widget is not None:
            self._children.append(widget)
            widget.setParent(self)
            layout = getattr(widget, "_layout", None)
            if isinstance(layout, _Layout) and layout.owner_widget() is None:
                layout._owner = widget

    def widget(self) -> Optional[QWidget]:
        return self._widget

    def setWidgetResizable(self, resizable: bool) -> None:
        self._resizable = resizable

    def setFrameShape(self, shape: Any) -> None:
        self._shape = shape

    def setHorizontalScrollBarPolicy(self, policy: Any) -> None:
        self._h_policy = policy


class QSplitter(QWidget):
    def __init__(self, orientation: Any = None, parent: Any = None) -> None:
        super().__init__(parent)
        self._widgets: List[QWidget] = []
        self._sizes: List[int] = []

    def addWidget(self, widget: QWidget) -> None:
        self._widgets.append(widget)

    def setSizes(self, sizes: List[int]) -> None:
        self._sizes = list(sizes)

    def sizes(self) -> List[int]:
        return list(self._sizes)

    def setStretchFactor(self, index: int, stretch: int) -> None:
        self._stretch = getattr(self, "_stretch", {})
        self._stretch[index] = stretch

    def setHandleWidth(self, width: int) -> None:
        self._handle = width

    def setChildrenCollapsible(self, collapsible: bool) -> None:
        self._collapsible = collapsible


class QTabWidget(QWidget):
    currentChanged = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._tabs: List[tuple[QWidget, str]] = []

    def addTab(self, widget: QWidget, title: str) -> int:
        self._tabs.append((widget, title))
        return len(self._tabs) - 1

    def count(self) -> int:
        return len(self._tabs)

    def widget(self, index: int) -> QWidget:
        return self._tabs[index][0]

    def setCurrentIndex(self, index: int) -> None:
        self._index = index


class QDockWidget(QWidget):
    DockWidgetClosable = 1
    DockWidgetMovable = 2
    DockWidgetFloatable = 4

    def __init__(self, title: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self._title = title
        self._widget: Optional[QWidget] = None

    def setWidget(self, widget: QWidget) -> None:
        self._widget = widget
        if widget is not None:
            self._children.append(widget)
            widget.setParent(self)
            layout = getattr(widget, "_layout", None)
            if isinstance(layout, _Layout) and layout.owner_widget() is None:
                layout._owner = widget

    def widget(self) -> Optional[QWidget]:
        return self._widget

    def setAllowedAreas(self, areas: Any) -> None:
        self._areas = areas

    def setFeatures(self, features: Any) -> None:
        self._features = features


class QStatusBar(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._widgets: List[QWidget] = []
        self._message = ""

    def addWidget(self, widget: QWidget, stretch: int = 0) -> None:
        self._widgets.append(widget)

    def showMessage(self, message: str, timeout: int = 0) -> None:
        self._message = message

    def message(self) -> str:
        return self._message


class QMenuBar(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._menus: List["QMenu"] = []

    def addMenu(self, title: str) -> "QMenu":
        menu = QMenu(title, self)
        self._menus.append(menu)
        return menu


class QMenu(QWidget):
    def __init__(self, title: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self._title = title
        self.actions_list: List[QAction] = []

    def addAction(self, action: QAction) -> None:
        self.actions_list.append(action)

    def addSeparator(self) -> QAction:
        action = QAction("---", self)
        self.actions_list.append(action)
        return action

    def title(self) -> str:
        return self._title


# --------------------------------------------------------------------------- #
#  Окна
# --------------------------------------------------------------------------- #


class QMainWindow(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._init_signals()
        self._central: Optional[QWidget] = None
        self._docks: List[QDockWidget] = []
        self._menubar = QMenuBar(self)
        self._statusbar: Optional[QStatusBar] = None
        self._geometry = b""
        self._state = b""

    def setCentralWidget(self, widget: QWidget) -> None:
        self._central = widget
        if widget is not None:
            self._children.append(widget)
            widget.setParent(self)

    def centralWidget(self) -> Optional[QWidget]:
        return self._central

    def addDockWidget(self, area: Any, dock: QDockWidget) -> None:
        self._docks.append(dock)
        self._children.append(dock)

    def menuBar(self) -> QMenuBar:  # noqa: N802
        return self._menubar

    def setStatusBar(self, bar: QStatusBar) -> None:
        self._statusbar = bar

    def statusBar(self) -> Optional[QStatusBar]:  # noqa: N802
        return self._statusbar

    def saveGeometry(self) -> bytes:
        return b"geometry"

    def restoreGeometry(self, data: bytes) -> bool:
        self._geometry = data
        return True

    def saveState(self) -> bytes:
        return b"state"

    def restoreState(self, data: bytes) -> bool:
        self._state = data
        return True


class QDialogButtonBox(QWidget):
    Ok = 1024
    Cancel = 4194304
    Close = 2097152
    Save = 2048
    Yes = 16384
    No = 65536
    Apply = 32

    class StandardButton:
        Ok = 1024
        Cancel = 4194304
        Close = 2097152
        Save = 2048
        Yes = 16384
        No = 65536
        Apply = 32

    Ok = StandardButton.Ok
    Cancel = StandardButton.Cancel
    Close = StandardButton.Close
    Save = StandardButton.Save
    Yes = StandardButton.Yes
    No = StandardButton.No

    accepted = Signal()
    rejected = Signal()

    def __init__(self, buttons: int = 0, parent: Any = None) -> None:
        super().__init__(parent)
        self._buttons: Dict[int, QPushButton] = {}
        titles = {
            self.StandardButton.Ok: "OK",
            self.StandardButton.Cancel: "Cancel",
            self.StandardButton.Close: "Close",
            self.StandardButton.Save: "Save",
            self.StandardButton.Yes: "Yes",
            self.StandardButton.No: "No",
        }
        for flag, title in titles.items():
            if buttons & flag:
                self._buttons[flag] = QPushButton(title, self)

    def button(self, which: int) -> Optional[QPushButton]:
        return self._buttons.get(which)

    def buttons(self) -> List[QPushButton]:
        return list(self._buttons.values())


class QDialog(QWidget):
    Accepted = 1
    Rejected = 0

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._init_signals()
        self._result = QDialog.Rejected
        self._modal = False

    def setModal(self, modal: bool) -> None:
        self._modal = bool(modal)

    def exec(self) -> int:  # noqa: A003
        return self._result

    def accept(self) -> None:
        self._result = QDialog.Accepted

    def reject(self) -> None:
        self._result = QDialog.Rejected

    def done(self, result: int) -> None:
        self._result = result


class QMessageBox(QDialog):
    Yes = 16384
    No = 65536
    Ok = 1024
    Cancel = 4194304

    class StandardButton:
        Ok = 1024
        Cancel = 4194304
        Yes = 16384
        No = 65536

    Yes = StandardButton.Yes
    No = StandardButton.No
    Ok = StandardButton.Ok
    Cancel = StandardButton.Cancel

    #: результат, который вернёт следующий вызов question() (для тестов)
    next_answer = StandardButton.Yes
    #: лог вызовов
    calls: List[tuple] = []

    @classmethod
    def information(cls, parent, title, text, *args, **kwargs) -> int:
        cls.calls.append(("information", title, text))
        return cls.Ok

    @classmethod
    def warning(cls, parent, title, text, *args, **kwargs) -> int:
        cls.calls.append(("warning", title, text))
        return cls.Ok

    @classmethod
    def critical(cls, parent, title, text, *args, **kwargs) -> int:
        cls.calls.append(("critical", title, text))
        return cls.Ok

    @classmethod
    def question(cls, parent, title, text, *args, **kwargs) -> int:
        cls.calls.append(("question", title, text))
        return cls.next_answer

    @classmethod
    def about(cls, parent, title, text) -> None:
        cls.calls.append(("about", title, text))


class QFileDialog(QDialog):
    next_files: List[str] = []
    next_dir: str = ""

    @classmethod
    def getOpenFileNames(cls, parent, caption, directory, filter_text="") -> tuple:  # noqa: N805
        return list(cls.next_files), filter_text

    @classmethod
    def getOpenFileName(cls, parent, caption, directory, filter_text="") -> tuple:  # noqa: N805
        return (cls.next_files[0] if cls.next_files else ""), filter_text

    @classmethod
    def getExistingDirectory(cls, parent, caption, directory, *args) -> str:  # noqa: N805
        return cls.next_dir

    @classmethod
    def getSaveFileName(cls, parent, caption, directory, filter_text="") -> tuple:  # noqa: N805
        return (cls.next_files[0] if cls.next_files else ""), filter_text


class QApplication(QObject):
    _instance: Optional["QApplication"] = None

    class Attribute:
        AA_ShareOpenGLContexts = 1
        AA_DontCreateNativeWidgetSiblings = 2
        AA_EnableHighDpiScaling = 3

    def __init__(self, argv: List[str] | None = None) -> None:
        super().__init__(None)
        self.argv = argv or []
        self._style_sheet = ""
        QApplication._instance = self

    @staticmethod
    def instance() -> Optional["QApplication"]:
        return QApplication._instance

    @staticmethod
    def setAttribute(attribute: Any, value: bool = True) -> None:
        return None

    def setStyleSheet(self, sheet: str) -> None:
        self._style_sheet = sheet

    def styleSheet(self) -> str:
        return self._style_sheet

    def setApplicationName(self, name: str) -> None:
        self._app_name = name

    def setApplicationVersion(self, version: str) -> None:
        self._app_version = version

    def setOrganizationName(self, name: str) -> None:
        self._org = name

    def setWindowIcon(self, icon: QIcon) -> None:
        self._icon = icon

    def exec(self) -> int:  # noqa: A003
        return 0

    def quit(self) -> None:
        return None


# --------------------------------------------------------------------------- #
#  Списки и таблицы
# --------------------------------------------------------------------------- #


class QListWidgetItem:
    def __init__(self, text: str = "") -> None:
        self._text = text
        self._data: Dict[Any, Any] = {}
        self._widget: Optional[QWidget] = None
        self._size_hint: Optional[QSize] = None
        self._tooltip = ""
        self._selected = False

    def setData(self, role: Any, value: Any) -> None:
        self._data[role] = value

    def data(self, role: Any) -> Any:
        return self._data.get(role)

    def setText(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setToolTip(self, text: str) -> None:
        self._tooltip = text

    def setSizeHint(self, size: QSize) -> None:
        self._size_hint = size

    def sizeHint(self) -> QSize:
        return self._size_hint or QSize(100, 40)

    def setSelected(self, selected: bool) -> None:
        self._selected = bool(selected)

    def isSelected(self) -> bool:
        return self._selected


class QListWidget(QWidget):
    ExtendedSelection = 3
    NoSelection = 0
    SingleSelection = 1
    MultiSelection = 2
    ExtendedSelection = 3
    ScrollBarAlwaysOff = 1

    itemClicked = Signal()
    itemDoubleClicked = Signal()
    currentItemChanged = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._items: List[QListWidgetItem] = []
        self._widgets: Dict[int, QWidget] = {}
        self._spacing = 0

    def addItem(self, item: QListWidgetItem) -> None:
        self._items.append(item)

    def addItems(self, texts: List[str]) -> None:
        for text in texts:
            self._items.append(QListWidgetItem(text))

    def item(self, index: int) -> Optional[QListWidgetItem]:
        return self._items[index] if 0 <= index < len(self._items) else None

    def count(self) -> int:
        return len(self._items)

    def row(self, item: QListWidgetItem) -> int:
        try:
            return self._items.index(item)
        except ValueError:
            return -1

    def takeItem(self, index: int) -> Optional[QListWidgetItem]:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def clear(self) -> None:
        self._items = []
        self._widgets = {}

    def setItemWidget(self, item: QListWidgetItem, widget: QWidget) -> None:
        self._widgets[id(item)] = widget

    def itemWidget(self, item: QListWidgetItem) -> Optional[QWidget]:
        return self._widgets.get(id(item))

    def selectedItems(self) -> List[QListWidgetItem]:
        return [item for item in self._items if item.isSelected()]

    def setSelectionMode(self, mode: Any) -> None:
        self._selection_mode = mode

    def setUniformItemSizes(self, uniform: bool) -> None:
        self._uniform = uniform

    def setSpacing(self, spacing: int) -> None:
        self._spacing = spacing

    def setHorizontalScrollBarPolicy(self, policy: Any) -> None:
        self._h_policy = policy

    def setCurrentItem(self, item: QListWidgetItem) -> None:
        self._current = item

    def currentItem(self) -> Optional[QListWidgetItem]:
        return getattr(self, "_current", None)


class QTableWidgetItem:
    def __init__(self, text: str = "") -> None:
        self._text = str(text)
        self._data: Dict[Any, Any] = {}
        self._foreground = None

    def setText(self, text: str) -> None:
        self._text = str(text)

    def text(self) -> str:
        return self._text

    def setData(self, role: Any, value: Any) -> None:
        self._data[role] = value

    def data(self, role: Any) -> Any:
        return self._data.get(role)

    def setForeground(self, color: Any) -> None:
        self._foreground = color

    def setToolTip(self, text: str) -> None:
        self._tooltip = text


class _Header(QWidget):
    Stretch = 1
    Interactive = 0
    Fixed = 2
    ResizeToContents = 3

    class ResizeMode:
        Interactive = 0
        Stretch = 1
        Fixed = 2
        ResizeToContents = 3

    Stretch = ResizeMode.Stretch

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._modes: Dict[int, Any] = {}

    def setStretchLastSection(self, enabled: bool) -> None:
        self._stretch_last = enabled

    def setSectionResizeMode(self, *args) -> None:
        self._modes[args[0]] = args[1] if len(args) > 1 else None


class QTableWidget(QWidget):
    itemDoubleClicked = Signal()
    itemSelectionChanged = Signal()

    class SelectionBehavior:
        SelectRows = 1
        SelectItems = 0

    class EditTrigger:
        NoEditTriggers = 0

    SelectRows = SelectionBehavior.SelectRows
    NoEditTriggers = EditTrigger.NoEditTriggers

    def __init__(self, rows: int = 0, columns: int = 0, parent: Any = None) -> None:
        super().__init__(parent)
        self._init_signals()
        self._rows = rows
        self._columns = columns
        self._items: Dict[tuple, QTableWidgetItem] = {}
        self._header_labels: List[str] = []
        self._current = (-1, -1)

    def setHorizontalHeaderLabels(self, labels: List[str]) -> None:
        self._header_labels = list(labels)
        self._columns = max(self._columns, len(labels))

    def setColumnCount(self, count: int) -> None:
        self._columns = count

    def horizontalHeader(self) -> _Header:
        if not hasattr(self, "_h_header"):
            self._h_header = _Header(self)
        return self._h_header

    def verticalHeader(self) -> _Header:
        if not hasattr(self, "_v_header"):
            self._v_header = _Header(self)
        return self._v_header

    def setSelectionBehavior(self, behavior: Any) -> None:
        self._behavior = behavior

    def setEditTriggers(self, triggers: Any) -> None:
        self._triggers = triggers

    def setAlternatingRowColors(self, enabled: bool) -> None:
        self._alternating = enabled

    def rowCount(self) -> int:
        return self._rows

    def columnCount(self) -> int:
        return self._columns

    def insertRow(self, row: int) -> None:
        self._rows += 1

    def setItem(self, row: int, column: int, item: QTableWidgetItem) -> None:
        self._items[(row, column)] = item

    def item(self, row: int, column: int) -> Optional[QTableWidgetItem]:
        return self._items.get((row, column))

    def currentRow(self) -> int:
        return self._current[0]

    def setCurrentCell(self, row: int, column: int) -> None:
        self._current = (row, column)


class QAbstractItemView(QWidget):
    SelectRows = 1
    SelectItems = 0
    NoEditTriggers = 0

    class SelectionBehavior:
        SelectRows = 1
        SelectItems = 0

    class EditTrigger:
        NoEditTriggers = 0


# --------------------------------------------------------------------------- #
#  Эффекты и прочее
# --------------------------------------------------------------------------- #


class QGraphicsDropShadowEffect(QObject):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)

    def setBlurRadius(self, radius: float) -> None:
        self._blur = radius

    def setOffset(self, x: float, y: float = 0) -> None:
        self._offset = (x, y)

    def setColor(self, color: Any) -> None:
        self._color = color


class QSizePolicy:
    Expanding = 3
    Preferred = 0
    Fixed = 2
    Minimum = 1


class QStyle:
    pass


class QCompleter(QObject):
    def __init__(self, items: List[str] | None = None, parent: Any = None) -> None:
        super().__init__(parent)
        self._items = items or []


# часто используемые псевдонимы
QHeaderView = _Header
