"""Проверка compat-слоя по .pyi-стаbam установленного PySide6.

Именно эта проверка поймала бы ошибку «setFeatures(int)»: она сверяет, что
каждое перечисление, которое compat берёт из Qt, действительно объявлено
в том классе, откуда мы его читаем (DockWidgetFeature живёт в QDockWidget!).
Запускается без дисплея и без импорта Qt-библиотек — читается только текст .pyi.
"""

from __future__ import annotations

import pytest

from app.ui import qt_pyi_check


@pytest.fixture(scope="module")
def report():
    return qt_pyi_check.verify()


def test_pyside6_stubs_present_or_skip(report) -> None:
    if not report.get("pyside6"):
        pytest.skip("PySide6 с .pyi-стабами не установлен — проверка не нужна")


def test_all_enums_live_where_compat_expects(report) -> None:
    if not report.get("pyside6"):
        pytest.skip("нет .pyi")
    problems = report["problems"]
    assert not problems, "compat читает перечисления не оттуда:\n" + "\n".join(problems)


def test_dock_features_are_in_qdockwidget(report) -> None:
    """Регрессия: TypeError setFeatures(int) на PySide6 6.12."""
    if not report.get("pyside6"):
        pytest.skip("нет .pyi")
    checked = set(report["checked"])
    assert "QDockWidget.DockWidgetFeature.DockWidgetClosable" in checked
    assert "QDockWidget.DockWidgetFeature.DockWidgetMovable" in checked


def test_compat_returns_typed_values() -> None:
    """На настоящем PySide6 compat не должен отдавать голые int."""
    problems = qt_pyi_check.verify_compat_values()
    assert not problems, "\n".join(problems)
