"""Живой smoke-тест GUI на НАСТОЯЩЕМ PySide6 (offscreen-рендер, без дисплея).

Запуск:
    QT_QPA_PLATFORM=offscreen python scripts/dev_gui_smoke.py

Проверяет то, на чём обычно всё и падает у пользователя: создание окна,
построение всех панелей, карточки моделей, диалоги, добавление файлов,
выбор модели, запуск воркера (без реальной модели — ожидаем понятную ошибку).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, details: str = "") -> None:
    CHECKS.append((name, bool(condition), details))
    mark = "OK " if condition else "FAIL"
    print(f"[{mark}] {name}" + (f" — {details}" if details else ""))


def run() -> int:
    app = QApplication.instance() or QApplication(["LocalASR-smoke"])

    from app.settings import Settings
    from app.core.models_catalog import ModelCatalog
    from app.ui import theme
    from app.ui.main_window import MainWindow, app_icon

    app.setStyleSheet(theme.stylesheet())
    app.setWindowIcon(app_icon())
    check("stylesheet применён", len(app.styleSheet()) > 1000)
    check("иконка собрана", not app_icon().isNull())

    tmp = Path(tempfile.mkdtemp(prefix="localasr-smoke-"))
    settings = Settings.load(root=tmp)
    catalog = ModelCatalog(root=tmp)
    check("каталог загружен", len(catalog.all()) >= 10, f"{len(catalog.all())} моделей")

    window = MainWindow(settings=settings, catalog=catalog)
    window.show()
    app.processEvents()
    check("главное окно создано", window.isVisible())
    check("заголовок", "LocalASR" in window.windowTitle(), window.windowTitle())

    window._startup_checks()
    app.processEvents()
    check("startup-проверки прошли", isinstance(window.hardware, dict))
    check("карточки моделей построены", len(window.models_panel._cards) == len(catalog.all()))
    check("баннер рекомендации виден", window.banner.isVisible())
    check("кнопка РАБОТАТЬ заблокирована", not window.work_button.isEnabled())

    # --- файлы
    media = tmp / "лекция 1.mp4"
    media.write_bytes(b"\x00" * 2048)
    added = window.file_list.add_paths([str(media)])
    app.processEvents()
    check("файл добавлен", added == 1 and window.file_list.count() == 1)

    # --- имитация установленной модели
    entry = catalog.get("whisper-base") or catalog.all()[0]
    folder = catalog.models_root / entry.id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.bin").write_bytes(b"\x00" * 32)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "installed.json").write_text(
        '{"id": "%s", "engine": "%s"}' % (entry.id, entry.engine), encoding="utf-8"
    )
    catalog.refresh_installed()
    window.models_panel.rebuild()
    window.on_model_selected(catalog.get(entry.id))
    app.processEvents()
    check("модель выбрана", window.selected_model is not None)
    check("кнопка РАБОТАТЬ активна", window.work_button.isEnabled())

    # --- диалоги
    from app.ui.dialogs import (
        CustomModelDialog,
        FirstRunDialog,
        HardwareDialog,
        ModelInfoDialog,
        ProgressDialog,
        SettingsDialog,
    )
    from app.ui.main_window import ResultsDialog

    for factory, name in (
        (lambda: ProgressDialog("Тест", "описание"), "ProgressDialog"),
        (lambda: ModelInfoDialog(entry), "ModelInfoDialog"),
        (lambda: HardwareDialog("CPU 2", "ffmpeg 7", ["заметка"]), "HardwareDialog"),
        (lambda: FirstRunDialog("CPU 2", catalog.all()[:2]), "FirstRunDialog"),
        (lambda: SettingsDialog(settings, window.hardware, window), "SettingsDialog"),
        (lambda: CustomModelDialog(window), "CustomModelDialog"),
        (lambda: ResultsDialog({}, window), "ResultsDialog"),
    ):
        dialog = factory()
        dialog.show()
        app.processEvents()
        check(f"диалог {name} открывается", True)
        dialog.close()

    # --- настройки
    window.language_combo.setCurrentIndex(window.language_combo.findData("kk"))
    window.format_combo.setCurrentIndex(window.format_combo.findData("srt"))
    window._collect_settings()
    check("настройки собираются из UI", settings.language == "kk" and settings.output_format == "srt")

    # --- запуск работы без настоящей модели: должна быть понятная ошибка, не креш
    errors: list[str] = []
    QMessageBox.critical = classmethod(lambda cls, parent, title, text, *a, **k: errors.append(text))
    QMessageBox.warning = classmethod(lambda cls, parent, title, text, *a, **k: errors.append(text))
    window.on_work()
    for _ in range(60):
        app.processEvents()
        if window.worker is None or not window.worker.isRunning():
            break
        import time

        time.sleep(0.1)
    app.processEvents()
    check("воркер отработал без падения приложения", window.isVisible())
    check("сообщение об ошибке получено", bool(errors) or window.file_label.text() != "",
          (errors or [window.file_label.text()])[0][:160])

    window.close()
    app.processEvents()

    failed = [name for name, ok, _ in CHECKS if not ok]
    print("\n" + "=" * 60)
    print(f"Проверок: {len(CHECKS)} · провалено: {len(failed)}")
    if failed:
        for name in failed:
            print("  ✗", name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
