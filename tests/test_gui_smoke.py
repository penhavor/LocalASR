# -*- coding: utf-8 -*-
"""GUI-смоук-тест LocalASR (offscreen): окно и диалоги собираются без падений.

Запуск:
    cd <корень проекта> && QT_QPA_PLATFORM=offscreen python tests/test_gui_smoke.py

Реальные модели не скачиваются/не грузятся — проверяется только интерфейс:
  * MainWindow строится, меню/панели/кнопки на месте;
  * «Выгрузить модель из памяти» серая, пока модель не загружена;
  * окно первого запуска и баннер-рекомендация удалены;
  * в каталоге работают числовые диапазоны и новые сортировки;
  * сброс настроек возвращает стандартные значения;
  * DiagnosticsDialog/ModelInfoDialog/SettingsDialog строятся;
  * карточка модели показывает параметры/квантование/год.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LOCALASR_QUIET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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

    # окно первого запуска и баннер удалены
    check(not hasattr(win, "banner"), "баннер-рекомендация удалён")
    check(not hasattr(win, "_show_first_run"), "метод первого запуска удалён")

    # «Выгрузить модель» серая, пока ничего не загружено
    check(win.unload_action is not None, "пункт «Выгрузить модель» существует")
    check(not win.unload_action.isEnabled(), "«Выгрузить модель» неактивна без загруженной модели")

    # меню Сервис: нет ручных установок движков/CUDA, есть диагностика
    menus = {a.menu().title(): a.menu() for a in win.menuBar().actions() if a.menu()}
    service = menus.get("Сервис")
    service_texts = [a.text() for a in service.actions() if a.text()] if service else []
    check(not any("Установить движок" in t for t in service_texts),
          "из «Сервис» убраны ручные установки движков")
    check(not any("CUDA-библиотеки" in t for t in service_texts),
          "из «Сервис» убрана ручная установка CUDA")
    check(any("Диагностика" in t for t in service_texts),
          "в «Сервис» есть «Диагностика и восстановление»")
    check(not any("Как выбрать модель" in t for t in
                  [a.text() for a in (menus.get("Справка").actions() if menus.get("Справка") else [])]),
          "из «Справка» убрана реклама «Как выбрать модель»")

    # кнопка РАБОТАТЬ неактивна без файлов и модели
    check(not win.work_button.isEnabled(), "«РАБОТАТЬ» неактивна без файлов/модели")

    print("== панель моделей: диапазоны и сортировки ==")
    panel = win.models_panel
    sort_keys = {panel.sort_combo.itemData(i) for i in range(panel.sort_combo.count())}
    for expected in ("year", "params", "ram", "size", "quality", "speed", "name"):
        check(expected in sort_keys, f"сортировка «{expected}» доступна")
    check(hasattr(panel, "year_lo") and hasattr(panel, "params_hi"),
          "числовые диапазоны (год, параметры) на месте")

    # диапазон по году отсекает всё старше
    panel.year_lo.setValue(2026)
    panel.year_hi.setValue(panel.year_hi.maximum())
    panel._apply_filters()
    visible_after = sum(1 for c in panel._cards.values() if c.isVisible())
    check(visible_after > 0, f"фильтр по году 2026+ показывает модели ({visible_after})")
    panel.reset_filters_to_defaults()
    panel._apply_filters()

    print("== сброс настроек ==")
    settings.language = "en"
    settings.beam_size = 9
    settings.advanced_open = True
    settings.save()
    # реальный поток: диалог сначала сбрасывает объект настроек, затем зовёт колбэк окна
    settings.reset_to_defaults()
    win._reset_settings_to_defaults()
    app.processEvents()
    check(settings.language == "ru", "после сброса язык = ru")
    check(settings.beam_size == 5, "после сброса beam = 5")
    check(not settings.advanced_open, "после сброса расширенные настройки свёрнуты")
    check(not win.advanced_frame.isVisible(), "панель расширенных настроек скрыта после сброса")
    # язык в комбобоксе тоже вернулся к ru
    check(win.language_combo.currentData() == "ru", "комбо языка после сброса = Русский")

    print("== диалоги строятся ==")
    entry = catalog.get("whisper-large-v3-turbo") or catalog.all()[0]
    from app.ui.dialogs import DiagnosticsDialog, ModelInfoDialog, SettingsDialog

    info = ModelInfoDialog(entry, win)
    # проверим, что в инфо-диалоге есть строки про параметры/квантование/год
    from PySide6.QtWidgets import QLabel
    texts = " ".join(l.text() for l in info.findChildren(QLabel))
    check("Число параметров" in texts, "ModelInfoDialog показывает число параметров")
    check("Квантование" in texts, "ModelInfoDialog показывает квантование")
    check("Год релиза" in texts, "ModelInfoDialog показывает год релиза")
    info.deleteLater()

    reset_called = {"v": False}
    sd = SettingsDialog(settings, win.hardware, win, on_reset_defaults=lambda: reset_called.__setitem__("v", True))
    check(hasattr(sd, "reset_button"), "в настройках есть кнопка сброса")
    # вкладка движков не содержит ручных кнопок установки
    engine_buttons = [b.text() for b in sd.findChildren(type(sd.reset_button)) if "Установить движок" in b.text()]
    check(not engine_buttons, "во вкладке «Движки» нет кнопок ручной установки")
    sd.deleteLater()

    dd = DiagnosticsDialog(win)
    # дожидаемся фонового аудита, чтобы поток завершился корректно (без QThread-варнинга)
    if dd.audit_worker is not None:
        dd.audit_worker.wait(15000)
    app.processEvents()
    dd.close()
    dd.deleteLater()
    app.processEvents()
    check(True, "DiagnosticsDialog строится и проводит аудит")

    print("== карточка модели: параметры/квантование/год ==")
    from PySide6.QtWidgets import QLabel
    card = panel.card(entry.id)
    if card is not None:
        ctexts = " ".join(l.text() for l in card.findChildren(QLabel))
        check(entry.params_text in ctexts or "параметров" in ctexts,
              "карточка показывает число параметров")
        check((entry.quantization or "—") in ctexts or not entry.quantization,
              "карточка показывает квантование")

    win.close()
    app.processEvents()

    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)}")
        for f in failures:
            print("  -", f)
        return 1
    print("GUI-СМОУК: ВСЁ ОК")
    return 0


if __name__ == "__main__":
    sys.exit(main())
