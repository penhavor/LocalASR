"""Smoke-тесты GUI.

Цель — поймать ровно те баги, из-за которых «прога вылетает при открытии окна»:
опечатки в именах, неверные сигнатуры конструкторов, сломанные connect(),
обращения к несуществующим атрибутам. Тесты работают и с настоящим PySide6,
и со стабом из tests/stubs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.models_catalog import ModelCatalog
from app.settings import Settings

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QTimer, QUrl  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(["LocalASR-test"])
    yield app


@pytest.fixture()
def window(qapp, temp_root: Path):
    from app.ui.main_window import MainWindow

    settings = Settings.load(root=temp_root)
    catalog = ModelCatalog(root=temp_root)
    win = MainWindow(settings=settings, catalog=catalog)
    yield win
    win.close()
    win.deleteLater()


# --------------------------------------------------------------------------- #
#  Создание окна
# --------------------------------------------------------------------------- #


def test_window_creates_and_has_all_key_widgets(window) -> None:
    for name in (
        "file_list",
        "work_button",
        "cancel_button",
        "models_panel",
        "language_combo",
        "format_combo",
        "output_edit",
        "overall_progress",
        "file_progress",
        "preview",
        "log_view",
        "banner",
    ):
        assert hasattr(window, name), f"в окне нет атрибута {name}"
        assert getattr(window, name) is not None


def test_window_title_and_default_state(window) -> None:
    assert "LocalASR" in window.windowTitle()
    assert window.work_button.isEnabled() is False, "кнопка РАБОТАТЬ не должна быть активной без файлов и модели"
    assert window.file_list.count() == 0


def test_startup_checks_do_not_crash(window) -> None:
    """_startup_checks в бою вызывается через QTimer.singleShot — зовём напрямую."""
    window._startup_checks()
    flush = getattr(QTimer, "flush_pending", None)  # есть только в тестовом стабе
    if flush is not None:
        flush()
    assert isinstance(window.hardware, dict)
    assert window.models_panel.count_label.text()


def test_statusbar_filled_after_startup(window) -> None:
    window._startup_checks()
    assert window.status_model.text()
    assert "ffmpeg" in window.status_ffmpeg.text().lower()


# --------------------------------------------------------------------------- #
#  Файлы
# --------------------------------------------------------------------------- #


def test_add_files_and_update_counter(window, sample_media: list[Path]) -> None:
    media = [str(p) for p in sample_media if p.suffix != ".txt"]
    added = window.file_list.add_paths(media)
    assert added == 3
    assert window.file_list.count() == 3
    assert "3" in window.files_count_label.text()
    assert window.drop_hint.isVisible() is False


def test_non_media_files_are_rejected(window, sample_media: list[Path]) -> None:
    window.file_list.add_paths([str(p) for p in sample_media])
    names = {p.name for p in window.file_list.files()}
    assert "notes.txt" not in names


def test_duplicates_are_ignored(window, sample_media: list[Path]) -> None:
    first = str(sample_media[0])
    window.file_list.add_paths([first, first, first])
    assert window.file_list.count() == 1


def test_remove_and_clear(window, sample_media: list[Path]) -> None:
    window.file_list.add_paths([str(p) for p in sample_media])
    assert window.file_list.count() == 3
    window.file_list._remove_one(window.file_list.files()[0])
    assert window.file_list.count() == 2
    window.file_list.clear_all()
    assert window.file_list.count() == 0
    assert window.drop_hint.isVisible() is True


class _FakeDropEvent:
    """Мини-замена QDropEvent: достаточно для проверки наших обработчиков."""

    def __init__(self, mime) -> None:
        self._mime = mime
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def test_drag_and_drop_event(window, sample_media: list[Path]) -> None:
    mime = QMimeData([QUrl.fromLocalFile(str(sample_media[0]))])
    enter = _FakeDropEvent(mime)
    window.dragEnterEvent(enter)
    assert enter.accepted
    drop = _FakeDropEvent(mime)
    window.dropEvent(drop)
    assert drop.accepted
    assert window.file_list.count() == 1


def test_drag_and_drop_ignores_non_media(window, tmp_path: Path) -> None:
    text_file = tmp_path / "документ.docx"
    text_file.write_text("x", encoding="utf-8")
    event = _FakeDropEvent(QMimeData([QUrl.fromLocalFile(str(text_file))]))
    window.dropEvent(event)
    assert window.file_list.count() == 0


def test_dialog_add_files(window, monkeypatch, sample_media: list[Path]) -> None:
    from PySide6.QtWidgets import QFileDialog

    picked = [str(p) for p in sample_media[:2]]
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", classmethod(lambda cls, *a, **k: (picked, "")))
    window.on_add_files()
    assert window.file_list.count() == 2


# --------------------------------------------------------------------------- #
#  Модели
# --------------------------------------------------------------------------- #


def test_no_model_installed_shows_banner(window) -> None:
    window._startup_checks()
    assert window.catalog.installed() == []
    assert window.banner.isVisible() is True
    assert window.banner_button.text() == "Скачать рекомендованную"
    assert window._banner_target is not None


def test_selecting_uninstalled_model_offers_download(window, monkeypatch) -> None:
    entry = window.catalog.all()[0]
    answers = []
    monkeypatch.setattr(QMessageBox, "question", classmethod(lambda cls, *a, **k: answers.append(a) or QMessageBox.No))
    monkeypatch.setattr(window, "on_download_model", lambda item: answers.append(("download", item.id)))
    window.on_model_selected(entry)
    assert answers and answers[0][0] != "download", "без подтверждения скачивать нельзя"


def test_installed_model_becomes_selectable(window, temp_root: Path) -> None:
    entry = window.catalog.get("whisper-small") or window.catalog.all()[0]
    folder = temp_root / "models" / entry.id
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"\x00" * 16)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "installed.json").write_text(
        '{"id": "%s", "engine": "%s"}' % (entry.id, entry.engine), encoding="utf-8"
    )
    window.catalog.refresh_installed()
    window.models_panel.rebuild()

    fresh = window.catalog.get(entry.id)
    assert fresh is not None and fresh.installed
    window.on_model_selected(fresh)
    assert window.selected_model is not None
    assert window.settings.selected_model_id == entry.id
    assert window.banner.isVisible() is False


def test_work_button_gates_on_files_and_model(window, sample_media: list[Path], temp_root: Path) -> None:
    entry = window.catalog.all()[0]
    folder = temp_root / "models" / entry.id
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"0" * 8)
    (folder / "installed.json").write_text('{"id": "%s", "engine": "%s"}' % (entry.id, entry.engine), encoding="utf-8")
    window.catalog.refresh_installed()
    window.models_panel.rebuild()
    window.on_model_selected(window.catalog.get(entry.id))

    assert window.work_button.isEnabled() is False, "модель есть, файлов нет"
    window.file_list.add_paths([str(sample_media[0])])
    assert window.work_button.isEnabled() is True, "есть и модель, и файл — кнопка должна работать"
    assert "РАБОТАТЬ" in window.work_button.text()


def test_model_cards_are_created_for_every_entry(window) -> None:
    entries = window.catalog.all()
    assert len(window.models_panel._cards) == len(entries)
    for entry in entries:
        card = window.models_panel.card(entry.id)
        assert card is not None
        assert entry.name in card.title.text()
        assert card.entry.size_text != "—"


def test_model_filters(window) -> None:
    panel = window.models_panel
    panel.only_russian.setChecked(True)
    panel._apply_filters()
    visible = [card for card in panel._cards.values() if card.isVisible()]
    assert visible and all(card.entry.supports_russian for card in visible)

    panel.search.setText("whisper.cpp")
    panel._apply_filters()
    visible = [card for card in panel._cards.values() if card.isVisible()]
    assert all("whispercpp" == card.entry.engine for card in visible)

    panel.search.setText("несуществующаямодельxyz")
    panel._apply_filters()
    assert panel.empty_label.isVisible() is True

    panel.search.setText("")
    panel.only_russian.setChecked(False)
    panel._apply_filters()
    assert panel.empty_label.isVisible() is False


def test_delete_model_removes_folder(window, temp_root: Path, monkeypatch) -> None:
    entry = window.catalog.get("whisper-base") or window.catalog.all()[0]
    folder = temp_root / "models" / entry.id
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"0" * 8)
    (folder / "installed.json").write_text('{"id": "%s", "engine": "%s"}' % (entry.id, entry.engine), encoding="utf-8")
    window.catalog.refresh_installed()
    window.models_panel.rebuild()

    monkeypatch.setattr(QMessageBox, "question", classmethod(lambda cls, *a, **k: QMessageBox.Yes))
    window.on_delete_model(window.catalog.get(entry.id))
    assert not folder.exists()
    assert window.catalog.get(entry.id).installed is False


# --------------------------------------------------------------------------- #
#  Настройки и параметры
# --------------------------------------------------------------------------- #


def test_settings_are_collected_from_ui(window) -> None:
    index = window.language_combo.findData("en")
    window.language_combo.setCurrentIndex(index)
    index = window.format_combo.findData("srt")
    window.format_combo.setCurrentIndex(index)
    window.output_edit.setText("/tmp/custom-out")
    window.hotwords_edit.setText("трансформер, RAG")
    window.clean_check.setChecked(False)
    window._collect_settings()

    assert window.settings.language == "en"
    assert window.settings.output_format == "srt"
    assert window.settings.output_dir == "/tmp/custom-out"
    assert window.settings.hotwords == "трансформер, RAG"
    assert window.settings.suppress_hallucinations is False


def test_output_hint_shows_expected_filename(window) -> None:
    window._update_output_hint()
    hint = window.output_hint.text()
    assert "лекция 1" in hint
    assert hint.endswith(")")


def test_reset_output_dir(window, temp_root: Path) -> None:
    window.output_edit.setText("/somewhere/else")
    window._reset_output_dir()
    assert window.output_edit.text().endswith("outputs")


def test_settings_dialog_opens_and_saves(window) -> None:
    from app.ui.dialogs import SettingsDialog

    dialog = SettingsDialog(window.settings, window.hardware, window)
    assert dialog.tabs.count() == 4
    dialog.language_combo.setCurrentIndex(dialog.language_combo.findData("kk"))
    dialog.beam_spin.setValue(7)
    dialog._save_and_accept()
    assert window.settings.language == "kk"
    assert window.settings.beam_size == 7


# --------------------------------------------------------------------------- #
#  Диалоги
# --------------------------------------------------------------------------- #


def test_progress_dialog_cancel_and_finish(qapp) -> None:
    from app.ui.dialogs import ProgressDialog

    dialog = ProgressDialog("Тест", "описание")
    received = []
    dialog.cancelRequested.connect(lambda *args: received.append(args))
    dialog.set_fraction(0.42)
    dialog.set_status("качаю")
    dialog.append_log("строка лога")
    dialog.cancel()
    assert received == [()], "cancelRequested должен выстрелить один раз"
    assert dialog.cancelled is True

    dialog.finish("готово")
    assert dialog.progress.value() == 1000
    dialog.close()


def test_first_run_dialog(qapp, window) -> None:
    from app.ui.dialogs import FirstRunDialog

    picks = []
    entries = window.catalog.sorted_for_display(window.catalog.all())[:3]
    dialog = FirstRunDialog("CPU 8 · RAM 16 ГБ", entries)
    dialog.downloadRequested.connect(picks.append)
    # имитируем клик по кнопке скачивания первой карточки
    dialog.downloadRequested.emit(entries[0])
    assert picks and picks[0].id == entries[0].id


def test_model_info_dialog(qapp, window) -> None:
    from app.ui.dialogs import ModelInfoDialog

    entry = window.catalog.all()[0]
    dialog = ModelInfoDialog(entry)
    assert entry.name in dialog.windowTitle()


def test_hardware_dialog(qapp) -> None:
    from app.ui.dialogs import HardwareDialog

    dialog = HardwareDialog("CPU 8", "ffmpeg 7.0", ["заметка"])
    assert dialog.windowTitle()


def test_custom_model_dialog_local_folder(qapp, tmp_path: Path) -> None:
    from app.ui.dialogs import CustomModelDialog

    folder = tmp_path / "my-model"
    folder.mkdir()
    (folder / "model.bin").write_bytes(b"0" * 16)
    (folder / "config.json").write_text("{}", encoding="utf-8")

    dialog = CustomModelDialog()
    dialog.source_edit.setText(str(folder))
    dialog._inspect()
    assert "faster-whisper" in dialog.result_label.text().lower() or "CTranslate2" in dialog.result_label.text()

    dialog._accept()
    assert dialog.entry is not None
    assert dialog.entry.engine == "faster-whisper"


def test_custom_model_dialog_rejects_empty(qapp, monkeypatch) -> None:
    from app.ui.dialogs import CustomModelDialog

    dialog = CustomModelDialog()
    monkeypatch.setattr(QMessageBox, "warning", classmethod(lambda cls, *a, **k: QMessageBox.Ok))
    dialog._accept()
    assert dialog.entry is None


# --------------------------------------------------------------------------- #
#  Воркеры (конструирование и связи, без запуска сети/инференса)
# --------------------------------------------------------------------------- #


def test_workers_construct_and_wire(window, sample_media: list[Path]) -> None:
    from app.asr.transcriber import JobOptions
    from app.ui.workers import PackageInstaller, TranscribeWorker

    entry = window.catalog.all()[0]
    options = JobOptions.from_settings(entry, window.settings, window.settings.resolved_output_dir())
    worker = TranscribeWorker([sample_media[0]], options, window.settings)

    received = []
    worker.logLine.connect(received.append)
    worker.logLine.emit("проверка")
    assert received == ["проверка"]

    installer = PackageInstaller(["vosk>=0.3.45"], "Vosk")
    lines = []
    installer.outputLine.connect(lines.append)
    installer.outputLine.emit("pip install …")
    assert lines == ["pip install …"]
    assert installer._find_uv("python") is None or isinstance(installer._find_uv("python"), str)


def test_worker_reports_missing_engine_instead_of_crash(window, sample_media: list[Path], monkeypatch) -> None:
    from app.asr.transcriber import JobOptions
    from app.ui.workers import TranscribeWorker

    entry = window.catalog.get("vosk-ru-small-0.22") or window.catalog.all()[0]
    entry.installed = True
    entry.install_dir = Path("/nonexistent")
    options = JobOptions.from_settings(entry, window.settings, window.settings.resolved_output_dir())

    worker = TranscribeWorker([sample_media[0]], options, window.settings)
    signals: list[tuple] = []
    worker.engineMissing.connect(lambda engine, pkgs: signals.append((engine, pkgs)))
    worker.failed.connect(lambda message: signals.append(("failed", message)))
    worker.finishedWith.connect(lambda ok, fail, secs: signals.append(("finished", ok, fail)))
    worker.run()

    assert signals, "воркер обязан что-то сообщить, а не упасть молча"
    assert signals[0][0] in ("vosk", "failed")


def test_install_packages_interactive_creates_dialog(window, monkeypatch) -> None:
    created = {}

    class FakeWorker:
        def __init__(self, packages, title, parent=None):
            created["packages"] = list(packages)
            self.progress = _FakeSignal()
            self.outputLine = _FakeSignal()
            self.finishedWith = _FakeSignal()

        def start(self):
            created["started"] = True

        def isRunning(self):
            return False

        def cancel(self):
            created["cancelled"] = True

    class _FakeSignal:
        def connect(self, slot):
            return None

    monkeypatch.setattr("app.ui.main_window.PackageInstaller", FakeWorker)
    monkeypatch.setattr("app.ui.dialogs.QDialog.exec", lambda self: 0)
    window.install_packages_interactive(["vosk>=0.3.45"], "Vosk")
    assert created.get("packages") == ["vosk>=0.3.45"]
    assert created.get("started") is True


# --------------------------------------------------------------------------- #
#  Прочее
# --------------------------------------------------------------------------- #


def test_theme_stylesheet_is_valid_css() -> None:
    from app.ui import theme

    css = theme.stylesheet()
    assert css.count("{") == css.count("}"), "несбалансированные скобки в QSS"
    assert "QPushButton" in css and "QProgressBar" in css
    assert theme.palette_hints()["accent"].startswith("#")


def test_app_icon_builds(qapp) -> None:
    from app.ui.main_window import app_icon

    icon = app_icon()
    assert icon is not None


def test_log_sink_writes_to_log_view(window) -> None:
    from app.logutils import get_logger

    get_logger("test").warning("проверка связи с GUI")
    assert "проверка связи" in window.log_view.toPlainText()


def test_results_dialog(window) -> None:
    from app.asr.transcriber import FileResult
    from app.ui.main_window import ResultsDialog

    results = {
        0: FileResult(source=Path("лекция 1.mp4"), ok=True, words=120, outputs=[Path("/tmp/лекция 1.txt")],
                      audio_duration=600, elapsed_seconds=60),
        1: FileResult(source=Path("лекция 2.mp4"), ok=False, error="нет аудиодорожки"),
    }
    dialog = ResultsDialog(results)
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).text() == "лекция 1.mp4"
    assert "ошибка" in dialog.table.item(1, 1).text()
