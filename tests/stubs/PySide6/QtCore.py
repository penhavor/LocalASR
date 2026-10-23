"""Стаб PySide6.QtCore для тестов без установленного Qt."""

from __future__ import annotations

import base64
import threading
from typing import Any, Callable, List

from . import Qt  # noqa: F401  (реэкспорт)


# --------------------------------------------------------------------------- #
#  Сигналы
# --------------------------------------------------------------------------- #


class BoundSignal:
    def __init__(self, owner: Any, name: str) -> None:
        self.owner = owner
        self.name = name
        self.slots: List[Callable] = []
        self.emissions: List[tuple] = []

    def connect(self, slot: Callable) -> None:
        if slot is None:
            raise ValueError("slot is None")
        if not callable(slot):
            raise TypeError(f"слот {slot!r} не вызываемый")
        self.slots.append(slot)

    def disconnect(self, slot: Callable | None = None) -> None:
        if slot is None:
            self.slots.clear()
            return
        try:
            self.slots.remove(slot)
        except ValueError:
            pass

    def emit(self, *args) -> None:
        self.emissions.append(args)
        for slot in list(self.slots):
            slot(*args)

    def __call__(self, *args) -> None:
        """Signal-to-signal connection: в PySide6 сигнал можно передать как слот."""
        self.emit(*args)

    # для проверки «есть ли подключения»
    def receivers(self, *_args) -> int:
        return len(self.slots)


class Signal:
    """Дескриптор: на классе — Signal, на экземпляре — BoundSignal."""

    def __init__(self, *types, name: str = "") -> None:
        self.types = types
        self._name = name

    def __set_name__(self, owner, name) -> None:
        self._name = name
        owner.__signals__ = getattr(owner, "__signals__", {})
        owner.__signals__[name] = self

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        cache = instance.__dict__.setdefault("__bound_signals__", {})
        if self._name not in cache:
            cache[self._name] = BoundSignal(instance, self._name)
        return cache[self._name]


class Slot:  # noqa: N801 - имитируем декоратор PySide6
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, func):
        return func


# --------------------------------------------------------------------------- #
#  QObject / QThread
# --------------------------------------------------------------------------- #


class QObject:
    def __init__(self, parent: Any = None) -> None:
        self._parent = parent
        self._object_name = ""
        self._properties: dict = {}
        self._visible = True
        self._enabled = True
        self._children: List[Any] = []
        self._deleted = False
        if parent is not None and hasattr(parent, "_children"):
            parent._children.append(self)
        self._init_signals()

    # -- сигналы ---------------------------------------------------------- #
    def _collect_signals(self) -> dict:
        signals: dict = {}
        for klass in reversed(type(self).__mro__):
            signals.update(getattr(klass, "__signals__", {}) or {})
        return signals

    def _init_signals(self) -> None:
        """Создаёт BoundSignal для всех Signal-дескрипторов класса.

        Нужно потому, что подклассы (особенно QThread-воркеры) часто не вызывают
        super().__init__() — а сигналы у них всё равно должны работать.
        """
        cache = self.__dict__.setdefault("__bound_signals__", {})
        for name in self._collect_signals():
            cache.setdefault(name, BoundSignal(self, name))

    def signal(self, name: str) -> BoundSignal:
        cache = self.__dict__.setdefault("__bound_signals__", {})
        if name not in cache:
            cache[name] = BoundSignal(self, name)
        return cache[name]

    # -- поиск детей ------------------------------------------------------ #
    def findChild(self, type_: Any = None, name: str = "") -> Any:
        for child in self.findChildren(type_, name):
            return child
        return None

    def findChildren(self, type_: Any = None, name: str = "") -> List[Any]:
        found: List[Any] = []
        stack = list(getattr(self, "_children", []))
        while stack:
            child = stack.pop()
            stack.extend(getattr(child, "_children", []) or [])
            if type_ is not None and not isinstance(child, type_):
                continue
            if name and getattr(child, "objectName", lambda: "")() != name:
                continue
            found.append(child)
        return found

    # имя/свойства
    def setObjectName(self, name: str) -> None:
        self._object_name = name

    def objectName(self) -> str:
        return self._object_name

    def setProperty(self, key: str, value: Any) -> None:
        self._properties[key] = value

    def property(self, key: str) -> Any:
        return self._properties.get(key)

    # видимость/доступность
    def setVisible(self, visible: bool) -> None:
        self._visible = bool(visible)

    def isVisible(self) -> bool:
        return self._visible

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self._enabled

    # родитель/дети
    def setParent(self, parent: Any) -> None:
        self._parent = parent

    def parent(self) -> Any:
        return self._parent

    def parentWidget(self) -> Any:
        return self._parent

    def children(self) -> List[Any]:
        return list(self._children)

    def deleteLater(self) -> None:
        self._deleted = True

    # прочее
    def setToolTip(self, text: str) -> None:
        self._tooltip = text

    def toolTip(self) -> str:
        return getattr(self, "_tooltip", "")

    def updateGeometry(self) -> None:
        return None

    def blockSignals(self, block: bool) -> bool:
        previous = getattr(self, "_blocked", False)
        self._blocked = bool(block)
        return previous

    def style(self) -> "_Style":
        return _Style()

    def metaObject(self) -> Any:
        return None

    def tr(self, text: str) -> str:  # noqa: A003
        return text


class _Style:
    def unpolish(self, *_args) -> None:
        return None

    def polish(self, *_args) -> None:
        return None


class QThread(QObject):
    started = Signal()
    finished = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._running = False
        self._thread: threading.Thread | None = None

    def _ensure_ready(self) -> None:
        if "_running" not in self.__dict__:
            self._running = False
            self._thread = None
        self._init_signals()

    def start(self, *_args) -> None:
        self._ensure_ready()
        self._running = True
        self._thread = threading.Thread(target=self._wrapper, daemon=True)
        self._thread.start()

    def _wrapper(self) -> None:
        try:
            self.started.emit()
            self.run()
        finally:
            self._running = False
            self.finished.emit()

    def run(self) -> None:  # pragma: no cover - переопределяется в коде
        return None

    def isRunning(self) -> bool:
        self._ensure_ready()
        return self._running

    def wait(self, timeout: int = 0) -> bool:
        if self._thread is not None:
            self._thread.join(timeout=(timeout or 0) / 1000 or None)
        return True

    def quit(self) -> None:
        self._running = False

    def requestInterruption(self) -> None:
        self._running = False

    def isInterruptionRequested(self) -> bool:
        return not self._running


class QTimer(QObject):
    timeout = Signal()

    _pending: List[Callable] = []

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._interval = 0

    def start(self, msec: int = 0) -> None:
        self._interval = msec

    def stop(self) -> None:
        self._interval = 0

    def setInterval(self, msec: int) -> None:
        self._interval = msec

    @staticmethod
    def singleShot(msec: int, callback: Callable) -> None:
        """В тестах не исполняем автоматически — складываем, чтобы вызвать руками."""
        QTimer._pending.append(callback)

    @staticmethod
    def flush_pending() -> None:
        pending, QTimer._pending = QTimer._pending, []
        for callback in pending:
            callback()


# --------------------------------------------------------------------------- #
#  Прочие типы QtCore
# --------------------------------------------------------------------------- #


class QByteArray(bytes):
    @staticmethod
    def fromBase64(data: bytes | str) -> "QByteArray":
        if isinstance(data, str):
            data = data.encode("ascii")
        return QByteArray(base64.b64decode(data))

    def toBase64(self) -> "QByteArray":
        return QByteArray(base64.b64encode(bytes(self)))


class QUrl:
    def __init__(self, url: str = "") -> None:
        self._url = url

    @staticmethod
    def fromLocalFile(path: str) -> "QUrl":
        return QUrl("file://" + str(path))

    def toString(self) -> str:
        return self._url

    def toLocalFile(self) -> str:
        return self._url.replace("file://", "")


class QMimeData:
    def __init__(self, urls: List[QUrl] | None = None) -> None:
        self._urls = urls or []

    def hasUrls(self) -> bool:
        return bool(self._urls)

    def urls(self) -> List[QUrl]:
        return list(self._urls)


class QEvent:
    def __init__(self, mime: QMimeData | None = None) -> None:
        self._mime = mime or QMimeData()
        self.accepted = False
        self.ignored = False

    def mimeData(self) -> QMimeData:
        return self._mime

    def acceptProposedAction(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True

    def accept(self) -> None:
        self.accepted = True


class QSize:
    def __init__(self, width: int = 0, height: int = 0) -> None:
        self._w, self._h = width, height

    def width(self) -> int:
        return self._w

    def height(self) -> int:
        return self._h


class QPoint:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x, self.y = x, y


class QRect:
    def __init__(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h


class Signal_(Signal):  # алиас на случай импорта под другим именем
    pass
