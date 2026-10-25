# -*- coding: utf-8 -*-
"""LocalASR GUI smoke test (offscreen): the window and dialogs build without crashes.

Run:
    cd <project root> && QT_QPA_PLATFORM=offscreen python tests/test_gui_smoke.py

No real models are downloaded/loaded — only the interface is checked:
  * MainWindow builds; menus/panels/buttons are in place;
  * "Unload the model from memory" is gray until a model is loaded;
  * numeric ranges and the sort modes work in the catalog;
  * resetting settings returns the default values;
  * DiagnosticsDialog/ModelInfoDialog/SettingsDialog build;
  * the model card shows parameters/quantization/year.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LOCALASR_QUIET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import i18n  # noqa: E402

i18n.init("ru")  # stable UI language for widget-text assertions

from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([sys.argv[0]])

from app.core.models_catalog import ModelCatalog  # noqa: E402
from app.settings import Settings  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

app.setStyleSheet(theme.stylesheet())

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


def main() -> int:
    import tempfile

    tmp = tempfile.mkdtemp(prefix="localasr-gui-")
    settings = Settings.load(tmp)
    catalog = ModelCatalog()

    print("== MainWindow ==")
    win = MainWindow(settings=settings, catalog=catalog)
    win.show()
    app.processEvents()

    # "Unload model" is gray until something is loaded
    check(win.unload_action is not None, "the Unload model item exists")
    check(not win.unload_action.isEnabled(), "Unload model is inactive without a loaded model")

    # Tools menu: no manual engine/CUDA installs, diagnostics present
    menus = {a.menu().title(): a.menu() for a in win.menuBar().actions() if a.menu()}
    service = menus.get("Сервис")
    service_texts = [a.text() for a in service.actions() if a.text()] if service else []
    check(not any("Установить движок" in t for t in service_texts),
          "manual engine installs are absent from Tools")
    check(not any("CUDA-библиотеки" in t for t in service_texts),
          "manual CUDA installation is absent from Tools")
    check(any("Диагностика" in t for t in service_texts),
          "Tools contains Diagnostics & recovery")
    check(not any("Как выбрать модель" in t for t in
                  [a.text() for a in (menus.get("Справка").actions() if menus.get("Справка") else [])]),
          "Help has no extra promo items")

    # the RUN button is inactive without files and a model
    check(not win.work_button.isEnabled(), "RUN is inactive without files/model")

    print("== models panel: ranges and sorts ==")
    panel = win.models_panel
    sort_keys = {panel.sort_combo.itemData(i) for i in range(panel.sort_combo.count())}
    for expected in ("year", "params", "ram", "size", "quality", "speed", "name"):
        check(expected in sort_keys, f"sort {expected} is available")
    check(hasattr(panel, "year_lo") and hasattr(panel, "params_hi"),
          "numeric ranges (year, parameters) are present")

    # the year range cuts off everything older
    panel.year_lo.setValue(2026)
    panel.year_hi.setValue(panel.year_hi.maximum())
    panel._apply_filters()
    visible_after = sum(1 for c in panel._cards.values() if c.isVisible())
    check(visible_after > 0, f"the 2026+ year filter shows models ({visible_after})")
    panel.reset_filters_to_defaults()
    panel._apply_filters()

    print("== settings reset ==")
    settings.language = "en"
    settings.beam_size = 9
    settings.advanced_open = True
    settings.save()
    # file logging is on — the factory reset must turn it off
    from pathlib import Path as _Path
    from app import logutils
    logutils.enable_file_logging(root=_Path(tmp))
    settings.file_logging = True
    settings.save()
    check(logutils.current_log_file() is not None, "the log file is active before the reset")
    # the real flow: the dialog resets the settings object, then calls the window callback
    settings.reset_to_defaults()
    win._reset_settings_to_defaults()
    app.processEvents()
    check(settings.language == "ru", "after the reset language = ru")
    check(settings.file_logging is False, "after the reset file logging is off")
    check(logutils.current_log_file() is None, "the log is disabled immediately (no restart)")
    check(settings.beam_size == 5, "after the reset beam = 5")
    check(not settings.advanced_open, "after the reset advanced options are collapsed")
    check(not win.advanced_frame.isVisible(), "the advanced panel is hidden after the reset")
    # the speech-language combo returned to ru too
    check(win.language_combo.currentData() == "ru", "the speech-language combo after reset = Russian")

    print("== dialogs build ==")
    entry = catalog.get("whisper-large-v3-turbo") or catalog.all()[0]
    from app.ui.dialogs import DiagnosticsDialog, ModelInfoDialog, SettingsDialog

    info = ModelInfoDialog(entry, win)
    # the info dialog must contain rows about parameters/quantization/year
    from PySide6.QtWidgets import QLabel
    texts = " ".join(l.text() for l in info.findChildren(QLabel))
    check("Число параметров" in texts, "ModelInfoDialog shows the parameter count")
    check("Квантование" in texts, "ModelInfoDialog shows quantization")
    check("Год релиза" in texts, "ModelInfoDialog shows the release year")
    info.deleteLater()

    reset_called = {"v": False}
    sd = SettingsDialog(settings, win.hardware, win, on_reset_defaults=lambda: reset_called.__setitem__("v", True))
    check(hasattr(sd, "reset_button"), "settings have the reset button")
    # the engines tab contains no manual install buttons
    engine_buttons = [b.text() for b in sd.findChildren(type(sd.reset_button)) if "Установить движок" in b.text()]
    check(not engine_buttons, "the Engines tab has no manual install buttons")
    sd.deleteLater()

    dd = DiagnosticsDialog(win)
    # wait for the background audit so the thread ends cleanly (no QThread warning)
    if dd.audit_worker is not None:
        dd.audit_worker.wait(15000)
    app.processEvents()
    dd.close()
    dd.deleteLater()
    app.processEvents()
    check(True, "DiagnosticsDialog builds and runs the audit")

    print("== model card: parameters/quantization/year ==")
    from PySide6.QtWidgets import QLabel
    card = panel.card(entry.id)
    if card is not None:
        ctexts = " ".join(l.text() for l in card.findChildren(QLabel))
        check(entry.params_text in ctexts or "параметров" in ctexts,
              "the card shows the parameter count")
        check((entry.quantization or "—") in ctexts or not entry.quantization,
              "the card shows quantization")

    win.close()
    app.processEvents()

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print("  -", f)
        return 1
    print("GUI SMOKE: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
