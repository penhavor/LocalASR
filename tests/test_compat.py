"""Тесты слоя совместимости Qt-перечислений.

Если здесь что-то отвалится — GUI с большой вероятностью «вылетит при открытии»
на машине пользователя, поэтому проверяем, что все константы реально разрешились.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from app.ui import compat  # noqa: E402


@pytest.mark.parametrize(
    "name",
    [
        "ALIGN_LEFT",
        "ALIGN_RIGHT",
        "ALIGN_CENTER",
        "ALIGN_HCENTER",
        "ALIGN_VCENTER",
        "ALIGN_RIGHT_VCENTER",
        "HORIZONTAL",
        "VERTICAL",
        "BOTTOM_DOCK_AREA",
        "TOP_DOCK_AREA",
        "DOCK_CLOSABLE",
        "DOCK_MOVABLE",
        "CURSOR_POINTING_HAND",
        "TEXT_SELECTABLE_BY_MOUSE",
        "TEXT_FORMAT_RICH",
        "SCROLLBAR_ALWAYS_OFF",
        "COLOR_GREEN",
        "COLOR_RED",
        "COLOR_TRANSPARENT",
        "ROLE_USER",
        "WINDOW_CONTEXT_HELP_HINT",
    ],
)
def test_constant_is_resolved(name: str) -> None:
    value = getattr(compat, name)
    assert value is not None, f"{name} не разрешился"
    if name in ("ALIGN_CENTER", "ALIGN_HCENTER", "ALIGN_VCENTER", "ALIGN_LEFT", "ALIGN_RIGHT"):
        assert int(value) != 0 or name == "ALIGN_LEFT", f"{name} подозрительно равен нулю"
    if name in ("DOCK_CLOSABLE", "DOCK_MOVABLE", "BOTTOM_DOCK_AREA", "TOP_DOCK_AREA"):
        assert int(value) != 0, f"{name} = 0, значит enum не найден"


def test_helpers_return_something() -> None:
    assert compat.header_stretch_mode() is not None
    assert compat.dialog_accepted() == 1
    assert compat.elide_mode() is not None
    assert compat.selection_behaviour_rows() is not None
    assert compat.no_edit_triggers() is not None
    assert compat.frame_no_frame() is not None
    assert compat.frame_hline() is not None


def test_dialog_button_flags_are_distinct() -> None:
    ok = int(compat.dialog_buttons("Ok"))
    cancel = int(compat.dialog_buttons("Cancel"))
    both = int(compat.dialog_buttons("Ok", "Cancel"))
    assert ok and cancel and ok != cancel
    assert both == ok | cancel


def test_message_box_buttons() -> None:
    assert compat.message_box_yes() != compat.message_box_no()


def test_app_attributes_are_optional() -> None:
    """Их может не быть в новых Qt — код обязан это переживать."""
    for name in ("AA_SHARE_OPENGL_CONTEXTS", "AA_DONT_CREATE_NATIVE_WIDGET_SIBLINGS", "AA_ENABLE_HIGHDPI_SCALING"):
        getattr(compat, name)  # не должно бросать исключение
