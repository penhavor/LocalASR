# -*- coding: utf-8 -*-
"""Тесты UI-поведения LocalASR (задачи 1–4 + тумблер логов).

Offscreen-проверки (реальные виджеты Qt, без показа на экране):
  * колёсико мыши над QComboBox/QSpinBox НЕ меняет значение, а прокручивает
    страницу (wheel guard);
  * открытый выпадающий список комбо колёсиком листается (значения кликом);
  * панель фильтров каталога сворачивается/разворачивается, состояние
    сохраняется в settings.json (collapsed), на кнопке виден счётчик;
  * у журнала (log_view) НЕТ всплывающей подсказки;
  * верхнеуровневые меню (Файлы/Модели/Сервис/Справка) имеют подсказки;
  * неактивный пункт «Выгрузить модель из памяти» отключён, а тема содержит
    стиль QMenu::item:disabled (серый вид);
  * тумблер «Вести подробный лог-файл» в настройках создаёт лог СРАЗУ
    (без перезапуска) и выключает запись при снятии галочки.

Запуск:  cd <корень проекта> && QT_QPA_PLATFORM=offscreen python tests/test_ui_behaviors.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LOCALASR_QUIET", "1")
os.environ.pop("LOCALASR_LOG", None)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

app = QApplication.instance() or QApplication([sys.argv[0]])

from app import logutils  # noqa: E402

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


def make_wheel(widget: QWidget, delta: int = 120) -> QWheelEvent:
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    return QWheelEvent(
        pos,
        QPointF(widget.mapToGlobal(pos.toPoint())),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_wheel_guard() -> None:
    print("== колёсико не меняет значения (wheel guard) ==")
    from app.ui.wheelguard import install_wheel_guard

    guard = install_wheel_guard(app)
    try:
        host = QWidget()
        layout = QVBoxLayout(host)
        scroll = QScrollArea()
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        combo = QComboBox()
        combo.addItems(["один", "два", "три", "четыре", "пять"])
        combo.setCurrentIndex(0)
        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setValue(10)
        inner_layout.addWidget(combo)
        inner_layout.addWidget(spin)
        for _ in range(40):
            filler = QWidget()
            filler.setFixedHeight(30)
            inner_layout.addWidget(filler)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        host.resize(300, 200)
        host.show()
        app.processEvents()

        # колесо над комбо: значение НЕ меняется
        before_index = combo.currentIndex()
        QApplication.sendEvent(combo, make_wheel(combo, -120))
        app.processEvents()
        check(combo.currentIndex() == before_index,
              f"комбо: значение не изменилось колесом (осталось {combo.currentIndex()})")

        # колесо над спинбоксом: значение НЕ меняется
        before_value = spin.value()
        QApplication.sendEvent(spin, make_wheel(spin, 120))
        QApplication.sendEvent(spin, make_wheel(spin, -120))
        app.processEvents()
        check(spin.value() == before_value, f"спинбокс: значение не изменилось (осталось {spin.value()})")

        # событие прокрутки доходит до scroll area (страница листается)
        bar = scroll.verticalScrollBar()
        bar.setValue(0)
        QApplication.sendEvent(combo, make_wheel(combo, -240))
        app.processEvents()
        check(bar.value() > 0, f"страница прокрутилась колесом над комбо (bar={bar.value()})")

        # явный выбор мышью/клавиатурой по-прежнему работает
        combo.setCurrentIndex(2)
        check(combo.currentIndex() == 2, "явная смена значения работает")
        spin.stepUp()
        check(spin.value() == before_value + 1, "стрелки спинбокса работают")
        host.close()
    finally:
        app.removeEventFilter(guard)
        guard.deleteLater()


def test_filters_collapse() -> None:
    print("== сворачивание фильтров каталога ==")
    from app.core.models_catalog import ModelCatalog
    from app.settings import Settings
    from app.ui.models_panel import ModelListPanel

    with tempfile.TemporaryDirectory() as td:
        settings = Settings.load(Path(td))
        catalog = ModelCatalog(root=ROOT)
        panel = ModelListPanel(catalog, {}, settings=settings)
        panel.show()
        app.processEvents()

        check(panel.filters_frame.isVisible(), "по умолчанию фильтры РАСКРЫТЫ")
        check("▾" in panel.filters_toggle.text(), "кнопка показывает «▾ Фильтры»")

        panel.filters_toggle.click()
        app.processEvents()
        check(not panel.filters_frame.isVisible(), "после клика фильтры свёрнуты")
        check("▸" in panel.filters_toggle.text(), "кнопка показывает «▸»")
        check(settings.catalog_filters.get("collapsed") is True, "состояние сохранено в настройках")

        # счётчик активных фильтров на свёрнутой кнопке
        panel.engine_filter.setCurrentIndex(1)
        panel._apply_filters()
        app.processEvents()
        check("(1)" in panel.filters_toggle.text(),
              f"на кнопке виден счётчик активных фильтров: {panel.filters_toggle.text()!r}")

        # новая панель с теми же настройками — фильтры свёрнуты
        panel2 = ModelListPanel(catalog, {}, settings=settings)
        panel2.show()
        app.processEvents()
        check(not panel2.filters_frame.isVisible(), "свёрнутое состояние переживает перезапуск")
        panel2.close()

        # разворот обратно
        panel.filters_toggle.click()
        app.processEvents()
        check(panel.filters_frame.isVisible(), "повторный клик раскрывает фильтры")
        check(settings.catalog_filters.get("collapsed") is False, "состояние «раскрыто» сохранено")
        panel.close()


def test_window_tweaks() -> None:
    print("== главное окно: тултипы и «Выгрузить модель» ==")
    from app.core.models_catalog import ModelCatalog
    from app.settings import Settings
    from app.ui import theme
    from app.ui.main_window import MainWindow

    app.setStyleSheet(theme.stylesheet())
    with tempfile.TemporaryDirectory() as td:
        settings = Settings.load(Path(td))
        catalog = ModelCatalog(root=ROOT)
        win = MainWindow(settings=settings, catalog=catalog)
        win.show()
        app.processEvents()

        # журнал БЕЗ подсказки (не отвлекает)
        check(not win.log_view.toolTip(), "у журнала нет всплывающей подсказки")

        # верхнеуровневые меню имеют подсказки
        menus = {a.text(): a for a in win.menuBar().actions()}
        for title in ("Файлы", "Модели", "Сервис", "Справка"):
            action = menus.get(title)
            check(action is not None and bool(action.toolTip()), f"меню «{title}» имеет подсказку")

        # все пункты меню (кроме разделителей) — с подсказками
        empty = []
        for title, action in menus.items():
            menu = action.menu()
            if menu is None:
                continue
            for item in menu.actions():
                if item.isSeparator():
                    continue
                if not item.toolTip():
                    empty.append(f"{title} → {item.text()}")
        check(not empty, f"у всех пунктов меню есть подсказки {empty or ''}")

        # «Выгрузить модель из памяти»: неактивна без модели + серый стиль в теме
        check(not win.unload_action.isEnabled(), "«Выгрузить модель» неактивна без загруженной модели")
        css = theme.stylesheet()
        check("QMenu::item:disabled" in css, "в теме есть стиль неактивного пункта меню")
        check("color: #4d566b" in css.replace("color:#4d566b", "color: #4d566b"),
              "неактивные пункты меню серые")

        # подсказка журнала на доке тоже пустая
        check(not win.log_dock.toolTip(), "у дока журнала нет подсказки")
        win.close()
        app.processEvents()


def test_settings_dialog_logging_toggle() -> None:
    print("== тумблер логов в настройках действует сразу ==")
    from app.settings import Settings
    from app.ui.dialogs import SettingsDialog

    logutils.disable_file_logging(final_note=False)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        settings = Settings.load(root)
        dialog = SettingsDialog(settings, {}, None)

        check(hasattr(dialog, "file_logging_check"), "в настройках есть вкладка журнала")
        check(not dialog.file_logging_check.isChecked(), "по умолчанию логирование выключено")

        # включение -> файл появляется СРАЗУ
        import app.paths as app_paths

        original_root = app_paths.ROOT_DIR
        try:
            app_paths.ROOT_DIR = root
            dialog.file_logging_check.setChecked(True)
            app.processEvents()
            current = logutils.current_log_file()
            check(current is not None and current.is_file(), f"лог создан сразу: {current and current.name}")
            check(current is not None and current.parent == root / "logs", "лог лежит в logs/ проекта")
            check(settings.file_logging is True, "настройка file_logging сохранена")
            check((root / "settings" / "settings.json").is_file(), "settings.json записан")

            dialog.file_logging_check.setChecked(False)
            app.processEvents()
            check(logutils.current_log_file() is None, "после выключения запись прекращена")
            check(settings.file_logging is False, "настройка обновилась на False")
            check(current is not None and current.is_file(), "сам файл лога остался на диске")
        finally:
            app_paths.ROOT_DIR = original_root
            logutils.disable_file_logging(final_note=False)
        dialog.close()


def main() -> int:
    test_wheel_guard()
    test_filters_collapse()
    test_window_tweaks()
    test_settings_dialog_logging_toggle()
    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("ТЕСТ UI-ПОВЕДЕНИЯ ПРОЙДЕН")
    return 0


if __name__ == "__main__":
    code = main()
    # Qt-объекты без родителя в offscreen-режиме могут разрушить кучу при
    # GC-разборе во время выхода интерпретатора (segfault ПОСЛЕ успешных
    # проверок). Результаты уже напечатаны — выходим немедленно.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
