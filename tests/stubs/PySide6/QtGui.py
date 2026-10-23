"""Стаб PySide6.QtGui для тестов."""

from __future__ import annotations

from typing import Any, List

from .QtCore import QObject, Signal


class QFont:
    def __init__(self, family: str = "", size: int = -1) -> None:
        self._family = family
        self._size = size

    def setPointSize(self, size: int) -> None:
        self._size = size

    def pointSize(self) -> int:
        return self._size

    def setBold(self, bold: bool) -> None:
        self._bold = bold

    def family(self) -> str:
        return self._family


class QFontMetrics:
    def __init__(self, font: QFont | None = None) -> None:
        self.font = font or QFont()

    def elidedText(self, text: str, mode: Any = None, width: int = 100) -> str:
        return text if len(text) <= width else text[: max(0, width - 1)] + "…"


class QColor:
    def __init__(self, *args) -> None:
        self.args = args


class QPalette:
    def __init__(self) -> None:
        self._data: dict = {}

    def setColor(self, *args) -> None:
        self._data[args[:1]] = args[-1]


class QIcon:
    def __init__(self, pixmap: Any = None) -> None:
        self._pixmap = pixmap
        self._empty = pixmap is None

    def addPixmap(self, pixmap: Any, *_args) -> None:
        self._pixmap = pixmap
        self._empty = False

    def isNull(self) -> bool:
        return self._empty


class QPixmap:
    def __init__(self, *args) -> None:
        self.args = args

    @staticmethod
    def fromImage(image: "QImage") -> "QPixmap":
        return QPixmap(image)

    def isNull(self) -> bool:
        return True


class QImage:
    def __init__(self, data: bytes | None = None) -> None:
        self.data = data

    @staticmethod
    def fromData(data: bytes) -> "QImage":
        return QImage(data)

    def isNull(self) -> bool:
        return not self.data


class QKeySequence:
    def __init__(self, text: str = "") -> None:
        self.text = text


class QAction(QObject):
    triggered = Signal()

    def __init__(self, text: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self._text = text
        self._shortcut = ""
        self._checkable = False

    def setText(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setShortcut(self, shortcut: Any) -> None:
        self._shortcut = shortcut

    def setCheckable(self, checkable: bool) -> None:
        self._checkable = checkable

    def trigger(self) -> None:
        self.triggered.emit()


class QDesktopServices:
    opened: List[str] = []

    @staticmethod
    def openUrl(url: Any) -> bool:
        QDesktopServices.opened.append(str(getattr(url, "_url", url)))
        return True


class QSurfaceFormat:
    def __init__(self, *args) -> None:
        self.args = args

    @staticmethod
    def setDefaultFormat(fmt: Any) -> None:
        return None


class QMouseEvent:
    def __init__(self, *args) -> None:
        self.args = args


class QDragEnterEvent:
    def __init__(self, mime: Any = None) -> None:
        self._mime = mime

    def mimeData(self) -> Any:
        return self._mime

    def acceptProposedAction(self) -> None:
        self.accepted = True
