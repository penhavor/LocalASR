"""Защита от случайной смены значений колёсиком мыши.

Проблема: наведённый курсор + прокрутка колеса меняли выбранное значение
QComboBox / QSpinBox / QDoubleSpinBox. Пользователь, пролистывая страницу
или каталог моделей, случайно превращал «Русский» в «Румынский», а
«beam 5» в «beam 8».

Решение: глобальный (application-level) фильтр событий. Колесо над
комбо/спинбоксом НЕ меняет значение, а передаётся родительскому виджету —
то есть работает как обычная прокрутка страницы (навигация). Значение
меняется только явным действием: кликом по пункту списка, стрелками
спинбокса или вводом с клавиатуры.

Открытый выпадающий список комбо (popup) фильтром не затрагивается: там
колесо листает сами пункты — это навигация, значение меняется только кликом.
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
    """Ближайшая прокручиваемая область вверх по дереву (её viewport).

    Отправленное программно wheel-событие Qt вверх НЕ передаёт (в отличие от
    «спонтанных» событий мыши), поэтому ищем QScrollArea/QListWidget/… сами —
    колесо над комбо прокручивает страницу, а не меняет значение.
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
    """Копия wheel-события с координатами, пересчитанными для родителя."""
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
    """Фильтр событий приложения: колесо над combo/spin = прокрутка страницы."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        try:
            if event.type() != QEvent.Type.Wheel:
                return False
            if isinstance(obj, QAbstractSpinBox):
                pass    # спинбоксы блокируем всегда
            elif isinstance(obj, QComboBox):
                if _popup_is_open(obj):
                    return False    # открытый список листается колесом — ок
            else:
                return False

            parent = _scroll_target(obj)
            if parent is not None:
                clone = _clone_for_parent(event, obj, parent)
                if clone is not None:
                    QApplication.sendEvent(parent, clone)
            event.accept()
            return True     # исходное событие съедаем: значение не меняется
        except RuntimeError:
            return False    # объект Qt уже удалён


def install_wheel_guard(app: QApplication) -> WheelGuard:
    """Ставит фильтр на приложение. Ссылку нужно держать живой (родитель = app)."""
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard
