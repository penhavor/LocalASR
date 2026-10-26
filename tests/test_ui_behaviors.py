# -*- coding: utf-8 -*-
"""LocalASR UI behavior tests (offscreen, real Qt widgets).

  * the mouse wheel over QComboBox/QSpinBox does NOT change the value but
    scrolls the page (wheel guard);
  * an open combo popup is scrolled by the wheel (values change on click);
  * the catalog filter panel collapses/expands, the state is saved in
    settings.json (collapsed) and the button shows an active-filter counter;
  * the journal (log_view) has NO tooltip;
  * top-level menus (File/Models/Tools/Help) have tooltips;
  * the disabled "Unload the model from memory" item is inactive and the
    theme contains the QMenu::item:disabled style (gray look);
  * the "Keep a verbose log file" toggle in settings creates the log
    IMMEDIATELY (no restart) and stops writing when unchecked;
  * resetting settings while the ACTIVE interface language differs from the
    post-reset one offers a restart (and does not when they are equal);
  * the result-format combo (txt/srt/vtt/json/md) drives settings, the
    output hint and the "Timecodes in .txt" availability.

Run:  cd <project root> && QT_QPA_PLATFORM=offscreen python tests/test_ui_behaviors.py
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

from app import i18n  # noqa: E402

i18n.init("ru")  # stable UI language for widget-text assertions

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
    print("== the wheel does not change values (wheel guard) ==")
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

        # wheel over the combo: the value does NOT change
        before_index = combo.currentIndex()
        QApplication.sendEvent(combo, make_wheel(combo, -120))
        app.processEvents()
        check(combo.currentIndex() == before_index,
              f"combo: wheel did not change the value (still {combo.currentIndex()})")

        # wheel over the spinbox: the value does NOT change
        before_value = spin.value()
        QApplication.sendEvent(spin, make_wheel(spin, 120))
        QApplication.sendEvent(spin, make_wheel(spin, -120))
        app.processEvents()
        check(spin.value() == before_value, f"spinbox: value unchanged (still {spin.value()})")

        # the scroll event reaches the scroll area (the page scrolls)
        bar = scroll.verticalScrollBar()
        bar.setValue(0)
        QApplication.sendEvent(combo, make_wheel(combo, -240))
        app.processEvents()
        check(bar.value() > 0, f"the page scrolled via the wheel over the combo (bar={bar.value()})")

        # explicit mouse/keyboard selection still works
        combo.setCurrentIndex(2)
        check(combo.currentIndex() == 2, "explicit value change works")
        spin.stepUp()
        check(spin.value() == before_value + 1, "spinbox arrows work")
        host.close()
    finally:
        app.removeEventFilter(guard)
        guard.deleteLater()


def test_filters_collapse() -> None:
    print("== catalog filter collapsing ==")
    from app.core.models_catalog import ModelCatalog
    from app.settings import Settings
    from app.ui.models_panel import ModelListPanel

    with tempfile.TemporaryDirectory() as td:
        settings = Settings.load(Path(td))
        catalog = ModelCatalog(root=ROOT)
        panel = ModelListPanel(catalog, {}, settings=settings)
        panel.show()
        app.processEvents()

        check(panel.filters_frame.isVisible(), "filters are EXPANDED by default")
        check("▾" in panel.filters_toggle.text(), "the button shows ▾ Filters")

        panel.filters_toggle.click()
        app.processEvents()
        check(not panel.filters_frame.isVisible(), "after the click filters are collapsed")
        check("▸" in panel.filters_toggle.text(), "the button shows ▸")
        check(settings.catalog_filters.get("collapsed") is True, "state saved in settings")

        # active-filter counter on the collapsed button
        panel.engine_filter.setCurrentIndex(1)
        panel._apply_filters()
        app.processEvents()
        check("(1)" in panel.filters_toggle.text(),
              f"the button shows the active-filter counter: {panel.filters_toggle.text()!r}")

        # a new panel with the same settings — filters collapsed
        panel2 = ModelListPanel(catalog, {}, settings=settings)
        panel2.show()
        app.processEvents()
        check(not panel2.filters_frame.isVisible(), "the collapsed state survives a restart")
        panel2.close()

        # expand back
        panel.filters_toggle.click()
        app.processEvents()
        check(panel.filters_frame.isVisible(), "a second click expands the filters")
        check(settings.catalog_filters.get("collapsed") is False, "the expanded state is saved")
        panel.close()


def test_window_tweaks() -> None:
    print("== main window: tooltips and Unload model ==")
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

        # the journal has NO tooltip (does not distract)
        check(not win.log_view.toolTip(), "the journal has no tooltip")

        # top-level menus have tooltips
        menus = {a.text(): a for a in win.menuBar().actions()}
        for title in ("Файлы", "Модели", "Сервис", "Справка"):
            action = menus.get(title)
            check(action is not None and bool(action.toolTip()), f"menu {title} has a tooltip")

        # all menu items (except separators) have tooltips
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
        check(not empty, f"all menu items have tooltips {empty or ''}")

        # "Unload the model from memory": inactive without a model + gray theme style
        check(not win.unload_action.isEnabled(), "Unload model is inactive without a loaded model")
        css = theme.stylesheet()
        check("QMenu::item:disabled" in css, "the theme has the disabled-menu-item style")
        check("color: #4d566b" in css.replace("color:#4d566b", "color: #4d566b"),
              "disabled menu items are gray")

        # the journal dock tooltip is empty too
        check(not win.log_dock.toolTip(), "the journal dock has no tooltip")
        win.close()
        app.processEvents()


def test_settings_dialog_logging_toggle() -> None:
    print("== the settings log toggle applies immediately ==")
    from app.settings import Settings
    from app.ui.dialogs import SettingsDialog

    logutils.disable_file_logging(final_note=False)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        settings = Settings.load(root)
        dialog = SettingsDialog(settings, {}, None)

        check(hasattr(dialog, "file_logging_check"), "settings have the logging tab")
        check(not dialog.file_logging_check.isChecked(), "logging is off by default")

        # enabling -> the file appears IMMEDIATELY
        import app.paths as app_paths

        original_root = app_paths.ROOT_DIR
        try:
            app_paths.ROOT_DIR = root
            dialog.file_logging_check.setChecked(True)
            app.processEvents()
            current = logutils.current_log_file()
            check(current is not None and current.is_file(), f"log created immediately: {current and current.name}")
            check(current is not None and current.parent == root / "logs", "the log lives in the project logs/")
            check(settings.file_logging is True, "the file_logging setting is saved")
            check((root / "settings" / "settings.json").is_file(), "settings.json written")

            dialog.file_logging_check.setChecked(False)
            app.processEvents()
            check(logutils.current_log_file() is None, "writing stops after disabling")
            check(settings.file_logging is False, "the setting updated to False")
            check(current is not None and current.is_file(), "the log file itself stays on disk")
        finally:
            app_paths.ROOT_DIR = original_root
            logutils.disable_file_logging(final_note=False)
        dialog.close()


def test_reset_language_offers_restart() -> None:
    print("== reset settings: interface language changes -> restart is offered ==")
    import os as _os

    from app import i18n as i18n_mod
    from app.settings import Settings
    from app.ui import compat, dialogs as dialogs_mod

    # remember the environment so detect_system_language() is deterministic
    env_backup = {var: _os.environ.get(var) for var in ("LANGUAGE", "LC_ALL", "LC_CTYPE", "LANG")}
    _os.environ["LANGUAGE"] = "ru"
    for var in ("LC_ALL", "LC_CTYPE", "LANG"):
        _os.environ.pop(var, None)

    real_message_box = dialogs_mod.QMessageBox
    asked: list[str] = []
    restarts: list[bool] = []

    class FakeMessageBox:
        """Records every question and always answers YES."""

        @staticmethod
        def question(parent, title, text, *args, **kwargs):
            asked.append(str(title))
            return compat.message_box_yes()

        def __getattr__(self, name):   # everything else -> the real class
            return getattr(real_message_box, name)

    original_restart = dialogs_mod.SettingsDialog._restart_application
    dialogs_mod.SettingsDialog._restart_application = lambda self: restarts.append(True)
    dialogs_mod.QMessageBox = FakeMessageBox
    try:
        # --- case A: active language EN, system RU, no explicit preference ---
        # (e.g. the user had chosen English explicitly; the reset returns the
        #  preference to AUTO -> the system Russian must be offered a restart)
        i18n_mod.init("en")
        with tempfile.TemporaryDirectory() as td:
            settings = Settings.load(Path(td))
            settings.ui_language = "en"
            dialog = dialogs_mod.SettingsDialog(settings, {}, None)
            dialog._reset_defaults()
            app.processEvents()   # fire the scheduled restart
            check(settings.ui_language == "", "reset returned ui_language to AUTO")
            check(len(asked) >= 2 and asked[-1] == i18n_mod.t("Interface language"),
                  f"the restart question was asked: {asked}")
            check(restarts == [True], "restart was requested after YES")
            dialog.close()

        # --- case B: active language already equals the post-reset language ---
        asked.clear()
        restarts.clear()
        i18n_mod.init("ru")
        with tempfile.TemporaryDirectory() as td:
            settings = Settings.load(Path(td))
            settings.ui_language = ""
            dialog = dialogs_mod.SettingsDialog(settings, {}, None)
            dialog._reset_defaults()
            app.processEvents()
            check(len(asked) == 1, f"only the reset confirmation, no restart offer: {asked}")
            check(restarts == [], "no restart when the language does not change")
            dialog.close()

        # --- case C: the user declines the restart -> no restart, no crash -----
        asked.clear()
        restarts.clear()

        class NoMessageBox(FakeMessageBox):
            @staticmethod
            def question(parent, title, text, *args, **kwargs):
                asked.append(str(title))
                # first question (the reset confirmation) — YES,
                # second (restart) — NO
                return compat.message_box_yes() if len(asked) == 1 else compat.message_box_no()

        dialogs_mod.QMessageBox = NoMessageBox
        i18n_mod.init("en")
        with tempfile.TemporaryDirectory() as td:
            settings = Settings.load(Path(td))
            settings.ui_language = "en"
            dialog = dialogs_mod.SettingsDialog(settings, {}, None)
            dialog._reset_defaults()
            app.processEvents()
            check(len(asked) == 2 and restarts == [], "NO answer -> the app is not restarted")
            dialog.close()
    finally:
        dialogs_mod.QMessageBox = real_message_box
        dialogs_mod.SettingsDialog._restart_application = original_restart
        for var, value in env_backup.items():
            if value is None:
                _os.environ.pop(var, None)
            else:
                _os.environ[var] = value
        i18n_mod.init("ru")   # the stable language of this test module


def test_result_format_combo() -> None:
    print("== result format combo (txt/srt/vtt/json/md) ==")
    from app.settings import Settings
    from app.ui.main_window import MainWindow

    check(MainWindow._ui_format("txt-time") == "txt", "txt-time maps to the TXT combo item")
    check(MainWindow._ui_format("srt") == "srt", "known formats pass through")
    check(MainWindow._ui_format("garbage") == "txt", "unknown formats fall back to txt")
    check(MainWindow._ui_format("") == "txt", "empty falls back to txt")

    with tempfile.TemporaryDirectory() as td:
        settings = Settings.load(Path(td))
        win = MainWindow(settings=settings)
        try:
            # default state
            check(win.format_combo.currentData() == "txt", "default format is txt")
            check(win.timestamps_check.isEnabled(), "timecodes are enabled for txt")

            # switching to srt: settings, hint and the timecodes box follow
            index = win.format_combo.findData("srt")
            win.format_combo.setCurrentIndex(index)
            app.processEvents()
            check(settings.output_format == "srt", "settings.output_format follows the combo")
            check(not win.timestamps_check.isEnabled(), "timecodes are disabled for srt")
            check(".srt" in win.output_hint.text(), f"the hint shows the .srt name: {win.output_hint.text()[:80]}")

            # vtt / json / md round-trip through _collect_settings
            for fmt in ("vtt", "json", "md"):
                win.format_combo.setCurrentIndex(win.format_combo.findData(fmt))
                app.processEvents()
                check(settings.output_format == fmt, f"format {fmt} is saved")

            # back to txt + timecodes checked -> the job upgrades to txt-time
            win.format_combo.setCurrentIndex(win.format_combo.findData("txt"))
            app.processEvents()
            check(settings.output_format == "txt" and win.timestamps_check.isEnabled(),
                  "txt restores the timecodes checkbox")
            win.timestamps_check.setChecked(True)
            app.processEvents()
            check(settings.add_timestamps_to_txt is True, "the timecode flag is saved")
            check(".txt" in win.output_hint.text(), "the hint is back to .txt")

            # a settings.json with "txt-time" (CLI compatibility) normalizes:
            # combo=txt + the timecodes box checked
            settings.output_format = "txt-time"
            win._apply_settings_to_ui()
            app.processEvents()
            check(win.format_combo.currentData() == "txt", "txt-time shows as TXT in the combo")
            check(win.timestamps_check.isChecked(), "txt-time turns the timecodes box on")
        finally:
            win.close()


def main() -> int:
    test_wheel_guard()
    test_filters_collapse()
    test_window_tweaks()
    test_settings_dialog_logging_toggle()
    test_reset_language_offers_restart()
    test_result_format_combo()
    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("UI BEHAVIOR TEST PASSED")
    return 0


if __name__ == "__main__":
    code = main()
    # Parentless Qt objects in offscreen mode may corrupt the heap during
    # GC teardown at interpreter exit (a segfault AFTER successful checks).
    # The results are printed — exit immediately.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
