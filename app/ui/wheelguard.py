"""Guard against accidental value changes with the mouse wheel.

Problem: hovering + wheel scrolling changed the selected value of
QComboBox / QSpinBox / QDoubleSpinBox — scrolling a page could turn
"Russian" into "Romanian" or "beam 5" into "beam 8".

Solution: an application-level event filter. The wheel over a combo/spin
does NOT change the value; it is forwarded to the parent scroll area (plain
navigation). Values change only via explicit actions: clicking an item,
spin arrows or keyboard input. An open combo popup is not affected: there
the wheel scrolls the items, which is navigation too.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject, QPointF
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QWidget,
)


def _popup_is_open(combo: QComboBox) -> bool:
    try:
        view = combo.view()
        return view is not None and view.isVisible()
    except RuntimeError:
        return False


def _scroll_target(widget: QWidget) -> Optional[QWidget]:
    """Nearest scrollable area up the tree (its viewport).

    Qt does not propagate programmatically sent wheel events upwards, so we
    find the QScrollArea/QListWidget/… ourselves.
    """
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            try:
                return parent.viewport()
            except RuntimeError:
                return parent
        parent = parent.parentWidget()
    return widget.parentWidget()


def _clone_for_parent(event: QWheelEvent, source: QWidget, target: QWidget) -> Optional[QWheelEvent]:
    """Copy of the wheel event with coordinates remapped for the parent."""
    try:
        global_pos = source.mapToGlobal(event.position().toPoint())
        local = QPointF(target.mapFromGlobal(global_pos))
        return QWheelEvent(
            local,
            QPointF(global_pos),
            event.pixelDelta(),
            event.angleDelta(),
            event.buttons(),
            event.modifiers(),
            event.phase(),
            event.inverted(),
            event.source(),
        )
    except (RuntimeError, TypeError):
        return None


class WheelGuard(QObject):
    """Application event filter: wheel over combo/spin = page scrolling."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        try:
            if event.type() != QEvent.Type.Wheel:
                return False
            if isinstance(obj, QAbstractSpinBox):
                pass    # spin boxes are always blocked
            elif isinstance(obj, QComboBox):
                if _popup_is_open(obj):
                    return False    # an open popup is wheel-scrollable — fine
            else:
                return False

            parent = _scroll_target(obj)
            if parent is not None:
                clone = _clone_for_parent(event, obj, parent)
                if clone is not None:
                    QApplication.sendEvent(parent, clone)
            event.accept()
            return True     # swallow the original event: the value stays
        except RuntimeError:
            return False    # the Qt object is already deleted


def install_wheel_guard(app: QApplication) -> WheelGuard:
    """Install the filter on the app. Keep the reference alive (parent = app)."""
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard
