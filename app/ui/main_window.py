"""Главное окно LocalASR."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QByteArray, QUrl, QTimer
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import paths
from ..asr.registry import registry
from ..asr.transcriber import FileResult, JobOptions
from ..constants import (
    APP_NAME,
    APP_VERSION,
    AUDIO_EXTENSIONS,
    LANGUAGES,
    MEDIA_EXTENSIONS,
    OUTPUT_FORMATS,
    VIDEO_EXTENSIONS,
)
from ..core.ffmpeg import ffmpeg_version, find_ffmpeg
from ..core.models_catalog import ModelCatalog, ModelEntry
from ..logutils import attach_qt_sink, detach_qt_sink, get_logger
from ..settings import Settings
from . import compat
from .dialogs import (
    CustomModelDialog,
    FirstRunDialog,
    HardwareDialog,
    ModelInfoDialog,
    ProgressDialog,
    SettingsDialog,
)
from .models_panel import ModelListPanel
from .widgets import FileListWidget, make_label, open_in_file_manager
from .workers import CatalogRefreshWorker, ModelDownloadWorker, PackageInstaller, TranscribeWorker

log = get_logger("ui")

MEDIA_FILTER = (
    "Медиафайлы ({});;Все файлы (*)".format(
        " ".join(f"*{ext}" for ext in sorted(MEDIA_EXTENSIONS))
    )
)


# --------------------------------------------------------------------------- #
#  Иконка (встроенный SVG — не зависит от внешних файлов)
# --------------------------------------------------------------------------- #

_ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">
<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#1d2740"/><stop offset="100%" stop-color="#0e1320"/></linearGradient>
<linearGradient id="w" x1="0" y1="0" x2="1" y2="0">
<stop offset="0%" stop-color="#5b8cff"/><stop offset="55%" stop-color="#7fd0ff"/>
<stop offset="100%" stop-color="#3ecf8e"/></linearGradient></defs>
<rect x="4" y="4" width="120" height="120" rx="28" fill="url(#bg)" stroke="#2f3c5c" stroke-width="3"/>
<g stroke="url(#w)" stroke-width="8" stroke-linecap="round" fill="none">
<path d="M28 64v6"/><path d="M42 52v24"/><path d="M56 38v52"/><path d="M70 48v32"/>
<path d="M84 58v12"/><path d="M98 62v4"/></g><circle cx="98" cy="34" r="9" fill="#3ecf8e"/></svg>"""


def app_icon() -> QIcon:
    """Иконка из встроенного SVG (не зависит от внешних файлов и сети)."""
    try:
        image = QImage.fromData(QByteArray(_ICON_SVG.encode("utf-8")))
        if not image.isNull():
            return QIcon(QPixmap.fromImage(image))
    except Exception as exc:  # noqa: BLE001
        log.debug("Не удалось собрать иконку из SVG: %s", exc)
    return QIcon()


# --------------------------------------------------------------------------- #
#  Главное окно
# --------------------------------------------------------------------------- #


class MainWindow(QMainWindow):
    def __init__(self, settings: Optional[Settings] = None, catalog: Optional[ModelCatalog] = None) -> None:
        super().__init__()
        self.settings = settings or Settings.load()
        self.catalog = catalog or ModelCatalog()
        self.hardware: dict = {}
        self.selected_model: Optional[ModelEntry] = None

        self.worker: Optional[TranscribeWorker] = None
        self.download_worker: Optional[ModelDownloadWorker] = None
        self.download_dialog: Optional[ProgressDialog] = None
        self.package_worker: Optional[PackageInstaller] = None
        self.package_dialog: Optional[ProgressDialog] = None
        self.catalog_worker: Optional[CatalogRefreshWorker] = None
        self.log_handler = None

        self.results: dict[int, FileResult] = {}
        self.job_started_at: float = 0.0
        self._pending_job: bool = False

        self.setWindowTitle(f"{APP_NAME} — локальное распознавание речи")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(1180, 760)
        self.setAcceptDrops(True)

        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        self._restore_geometry()

        self.log_handler = attach_qt_sink(self._on_log_record)
        log.info("GUI запущен (%s %s, Python %s)", APP_NAME, APP_VERSION, sys.version.split()[0])

        QTimer.singleShot(60, self._startup_checks)

    # ================================================================== #
    #  Построение интерфейса
    # ================================================================== #
    def _build_ui(self) -> None:
        splitter = QSplitter(compat.HORIZONTAL)
        splitter.setHandleWidth(10)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([720, 520])
        self.splitter = splitter

        self.setCentralWidget(splitter)
        self._build_log_dock()
        self._update_work_button()

    # ---------------- левая колонка ------------------------------------ #
    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(compat.frame_no_frame())
        scroll.setHorizontalScrollBarPolicy(compat.SCROLLBAR_ALWAYS_OFF)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 12, 16)
        layout.setSpacing(14)

        # --- заголовок
        header = QHBoxLayout()
        title = make_label(APP_NAME, "H1")
        header.addWidget(title)
        header.addWidget(make_label("· локальное распознавание речи без облака", "Muted"))
        header.addStretch(1)
        version = make_label(f"v{APP_VERSION}", "Muted")
        header.addWidget(version)
        layout.addLayout(header)

        # --- баннер-рекомендация (появится, если моделей нет)
        self.banner = QFrame()
        self.banner.setObjectName("Banner")
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(14, 12, 12, 12)
        banner_layout.setSpacing(12)
        self.banner_text = make_label("", "Dim", word_wrap=True)
        banner_layout.addWidget(self.banner_text, stretch=1)
        self.banner_button = QPushButton("Открыть каталог")
        self.banner_button.setObjectName("Primary")
        self.banner_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.banner_button.clicked.connect(self._focus_models)
        self._banner_target: Optional[ModelEntry] = None
        banner_layout.addWidget(self.banner_button)
        self.banner.setVisible(False)
        layout.addWidget(self.banner)

        # --- 1. ФАЙЛЫ
        files_box = QFrame()
        files_box.setObjectName("SidePanel")
        files_layout = QVBoxLayout(files_box)
        files_layout.setContentsMargins(14, 14, 14, 14)
        files_layout.setSpacing(10)

        files_header = QHBoxLayout()
        files_header.addWidget(make_label("1 · Файлы", "H2"))
        files_header.addStretch(1)
        self.files_count_label = make_label("0 файлов", "Muted")
        files_header.addWidget(self.files_count_label)
        files_layout.addLayout(files_header)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)

        self.add_files_button = QPushButton("📂 Выбрать файлы…")
        self.add_files_button.setObjectName("Primary")
        self.add_files_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.add_files_button.setShortcut(QKeySequence("Ctrl+O"))
        self.add_files_button.setToolTip("Можно выбрать сразу несколько файлов (Ctrl/Shift в проводнике)")
        self.add_files_button.clicked.connect(self.on_add_files)
        buttons.addWidget(self.add_files_button)

        self.add_folder_button = QPushButton("🗀 Папку…")
        self.add_folder_button.setObjectName("Ghost")
        self.add_folder_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.add_folder_button.setToolTip("Добавить все аудио/видео из папки (рекурсивно)")
        self.add_folder_button.clicked.connect(self.on_add_folder)
        buttons.addWidget(self.add_folder_button)

        self.clear_files_button = QPushButton("Очистить")
        self.clear_files_button.setObjectName("Ghost")
        self.clear_files_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.clear_files_button.clicked.connect(lambda: self.file_list.clear_all())
        buttons.addWidget(self.clear_files_button)

        self.remove_selected_button = QPushButton("Убрать выбранные")
        self.remove_selected_button.setObjectName("Ghost")
        self.remove_selected_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.remove_selected_button.setShortcut(QKeySequence("Delete"))
        self.remove_selected_button.clicked.connect(lambda: self.file_list.remove_selected())
        buttons.addWidget(self.remove_selected_button)
        buttons.addStretch(1)
        files_layout.addLayout(buttons)

        self.file_list = FileListWidget()
        self.file_list.setMinimumHeight(180)
        self.file_list.filesChanged.connect(self._update_files_label)
        files_layout.addWidget(self.file_list, stretch=1)

        self.drop_hint = make_label(
            "Перетащи сюда аудио или видео (mp4, mkv, mp3, wav, m4a, ogg…) — "
            "видео автоматически превратится в звук.",
            "Muted",
            word_wrap=True,
        )
        files_layout.addWidget(self.drop_hint)
        layout.addWidget(files_box, stretch=3)

        # --- 2. ПАРАМЕТРЫ
        options_box = QFrame()
        options_box.setObjectName("SidePanel")
        options_layout = QVBoxLayout(options_box)
        options_layout.setContentsMargins(14, 14, 14, 14)
        options_layout.setSpacing(10)
        options_layout.addWidget(make_label("2 · Как распознавать и куда сохранять", "H2"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        grid.addWidget(make_label("Язык речи", "Dim"), 0, 0)
        self.language_combo = QComboBox()
        for code, title in LANGUAGES:
            self.language_combo.addItem(title, code)
        language_index = self.language_combo.findData(self.settings.language)
        self.language_combo.setCurrentIndex(language_index if language_index >= 0 else 1)
        self.language_combo.currentIndexChanged.connect(self._on_settings_changed)
        grid.addWidget(self.language_combo, 0, 1)

        grid.addWidget(make_label("Формат результата", "Dim"), 0, 2)
        self.format_combo = QComboBox()
        for key, title in OUTPUT_FORMATS.items():
            self.format_combo.addItem(title, key)
        format_index = self.format_combo.findData(self.settings.output_format)
        self.format_combo.setCurrentIndex(format_index if format_index >= 0 else 0)
        self.format_combo.currentIndexChanged.connect(self._on_settings_changed)
        grid.addWidget(self.format_combo, 0, 3)

        grid.addWidget(make_label("Куда сохранять", "Dim"), 1, 0)
        save_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setText(self.settings.output_dir or str(paths.outputs_dir()))
        self.output_edit.setPlaceholderText(str(paths.outputs_dir()))
        self.output_edit.editingFinished.connect(self._on_settings_changed)
        save_row.addWidget(self.output_edit, stretch=1)
        browse_output = QPushButton("Обзор…")
        browse_output.setObjectName("Ghost")
        browse_output.setCursor(compat.CURSOR_POINTING_HAND)
        browse_output.clicked.connect(self.on_browse_output)
        save_row.addWidget(browse_output)
        reset_output = QToolButton()
        reset_output.setText("↺")
        reset_output.setToolTip("Вернуть папку по умолчанию: outputs/ внутри проекта")
        reset_output.setCursor(compat.CURSOR_POINTING_HAND)
        reset_output.clicked.connect(self._reset_output_dir)
        save_row.addWidget(reset_output)
        grid.addLayout(save_row, 1, 1, 1, 3)

        options_layout.addLayout(grid)

        self.output_hint = make_label("", "Muted", word_wrap=True)
        options_layout.addWidget(self.output_hint)

        toggles = QHBoxLayout()
        toggles.setSpacing(16)
        self.vad_check = QCheckBox("VAD (резать тишину)")
        self.vad_check.setChecked(self.settings.use_vad)
        self.vad_check.setToolTip("Silero VAD: убирает тишину и шум — быстрее и меньше галлюцинаций")
        self.vad_check.toggled.connect(self._on_settings_changed)
        toggles.addWidget(self.vad_check)

        self.clean_check = QCheckBox("Анти-галлюцинации")
        self.clean_check.setChecked(self.settings.suppress_hallucinations)
        self.clean_check.setToolTip(
            "Вычищает «[музыка]», «[аплодисменты]», «Спасибо за просмотр», "
            "субтитрные вставки и зацикленные повторы фраз."
        )
        self.clean_check.toggled.connect(self._on_settings_changed)
        toggles.addWidget(self.clean_check)

        self.paragraph_check = QCheckBox("Абзацы")
        self.paragraph_check.setChecked(self.settings.paragraphs)
        self.paragraph_check.toggled.connect(self._on_settings_changed)
        toggles.addWidget(self.paragraph_check)

        self.timestamps_check = QCheckBox("Таймкоды в .txt")
        self.timestamps_check.setChecked(self.settings.add_timestamps_to_txt)
        self.timestamps_check.toggled.connect(self._on_settings_changed)
        toggles.addWidget(self.timestamps_check)
        toggles.addStretch(1)

        self.advanced_toggle = QPushButton("▸ Расширенные настройки")
        self.advanced_toggle.setObjectName("Link")
        self.advanced_toggle.setCursor(compat.CURSOR_POINTING_HAND)
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        toggles.addWidget(self.advanced_toggle)
        options_layout.addLayout(toggles)

        self.advanced_frame = QFrame()
        advanced_layout = QVBoxLayout(self.advanced_frame)
        advanced_layout.setContentsMargins(0, 6, 0, 0)
        advanced_layout.setSpacing(8)
        self.advanced_frame.setVisible(False)

        hot_row = QHBoxLayout()
        hot_row.addWidget(make_label("Слова-подсказки (термины лекции)", "Dim"))
        self.hotwords_edit = QLineEdit()
        self.hotwords_edit.setPlaceholderText("трансформер, градиентный спуск, Kubernetes — через запятую")
        self.hotwords_edit.setText(self.settings.hotwords)
        self.hotwords_edit.editingFinished.connect(self._on_settings_changed)
        hot_row.addWidget(self.hotwords_edit, stretch=1)
        advanced_layout.addLayout(hot_row)

        prompt_row = QHBoxLayout()
        prompt_row.addWidget(make_label("Подсказка стилю", "Dim"))
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Например: Лекция по машинному обучению, русский язык, научный стиль.")
        self.prompt_edit.setText(self.settings.initial_prompt)
        self.prompt_edit.editingFinished.connect(self._on_settings_changed)
        prompt_row.addWidget(self.prompt_edit, stretch=1)
        advanced_layout.addLayout(prompt_row)

        extra_row = QHBoxLayout()
        extra_row.addWidget(make_label("Формат", "Dim"))
        self.beam_combo = QComboBox()
        for value, title in ((1, "beam 1 (быстро)"), (3, "beam 3"), (5, "beam 5 (баланс)"), (8, "beam 8 (точность)")):
            self.beam_combo.addItem(title, value)
        beam_index = self.beam_combo.findData(int(self.settings.beam_size or 5))
        self.beam_combo.setCurrentIndex(beam_index if beam_index >= 0 else 2)
        self.beam_combo.currentIndexChanged.connect(self._on_settings_changed)
        extra_row.addWidget(self.beam_combo)

        self.batched_check = QCheckBox("Batched (до 4× быстрее на GPU)")
        self.batched_check.setChecked(bool(self.settings.batched))
        self.batched_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.batched_check)

        self.wordts_check = QCheckBox("Таймкоды слов")
        self.wordts_check.setChecked(bool(self.settings.word_timestamps))
        self.wordts_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.wordts_check)

        self.translate_check = QCheckBox("Перевод в EN")
        self.translate_check.setChecked(bool(self.settings.translate_to_english))
        self.translate_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.translate_check)

        self.open_folder_check = QCheckBox("Открыть папку после завершения")
        self.open_folder_check.setChecked(bool(self.settings.open_output_folder_when_done))
        self.open_folder_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.open_folder_check)
        extra_row.addStretch(1)

        settings_button = QPushButton("Все настройки…")
        settings_button.setObjectName("Ghost")
        settings_button.setCursor(compat.CURSOR_POINTING_HAND)
        settings_button.clicked.connect(self.on_open_settings)
        extra_row.addWidget(settings_button)
        advanced_layout.addLayout(extra_row)

        options_layout.addWidget(self.advanced_frame)
        layout.addWidget(options_box, stretch=2)

        # --- 3. ПРОГРЕСС + КНОПКА
        work_box = QFrame()
        work_box.setObjectName("SidePanel")
        work_layout = QVBoxLayout(work_box)
        work_layout.setContentsMargins(14, 14, 14, 14)
        work_layout.setSpacing(10)

        work_layout.addWidget(make_label("3 · Работа", "H2"))

        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("%v%")
        work_layout.addWidget(self.overall_progress)

        self.overall_label = make_label("Готов к работе", "Dim", word_wrap=True)
        work_layout.addWidget(self.overall_label)

        self.file_progress = QProgressBar()
        self.file_progress.setObjectName("Subtle")
        self.file_progress.setRange(0, 1000)
        self.file_progress.setValue(0)
        self.file_progress.setFixedHeight(6)
        work_layout.addWidget(self.file_progress)

        self.file_label = make_label("", "Muted", word_wrap=True)
        work_layout.addWidget(self.file_label)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Здесь появится распознанный текст по мере работы…")
        self.preview.setMaximumHeight(96)
        work_layout.addWidget(self.preview)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        self.work_button = QPushButton("РАБОТАТЬ")
        self.work_button.setObjectName("Primary")
        self.work_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.work_button.setMinimumHeight(48)
        shadow = QGraphicsDropShadowEffect(self.work_button)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(compat.COLOR_TRANSPARENT)
        self.work_button.setGraphicsEffect(shadow)
        self.work_button.clicked.connect(self.on_work)
        action_row.addWidget(self.work_button, stretch=3)

        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.setObjectName("Danger")
        self.cancel_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.cancel_button.setMinimumHeight(48)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.on_cancel)
        action_row.addWidget(self.cancel_button, stretch=1)
        work_layout.addLayout(action_row)

        results_row = QHBoxLayout()
        self.results_button = QPushButton("Показать результаты")
        self.results_button.setObjectName("Ghost")
        self.results_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.results_button.setEnabled(False)
        self.results_button.clicked.connect(self.on_show_results)
        results_row.addWidget(self.results_button)
        self.open_output_button = QPushButton("Открыть папку результатов")
        self.open_output_button.setObjectName("Ghost")
        self.open_output_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.open_output_button.clicked.connect(lambda: open_in_file_manager(self.current_output_dir()))
        results_row.addWidget(self.open_output_button)
        results_row.addStretch(1)
        work_layout.addLayout(results_row)

        layout.addWidget(work_box, stretch=2)

        scroll.setWidget(page)
        return scroll

    # ---------------- правая колонка ----------------------------------- #
    def _build_right_panel(self) -> QWidget:
        self.models_panel = ModelListPanel(self.catalog, self.hardware)
        self.models_panel.modelSelected.connect(self.on_model_selected)
        self.models_panel.downloadRequested.connect(self.on_download_model)
        self.models_panel.deleteRequested.connect(self.on_delete_model)
        self.models_panel.infoRequested.connect(self.on_model_info)
        self.models_panel.customRequested.connect(self.on_add_custom_model)
        self.models_panel.refreshRequested.connect(self.on_refresh_catalog)
        return self.models_panel

    # ---------------- лог-панель --------------------------------------- #
    def _build_log_dock(self) -> None:
        self.log_dock = QDockWidget("Журнал", self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setAllowedAreas(compat.BOTTOM_DOCK_AREA | compat.TOP_DOCK_AREA)
        self.log_dock.setFeatures(compat.DOCK_CLOSABLE | compat.DOCK_MOVABLE)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("Журнал событий…")
        self.log_dock.setWidget(self.log_view)
        self.addDockWidget(compat.BOTTOM_DOCK_AREA, self.log_dock)
        self.log_dock.setVisible(bool(self.settings.show_log_panel))

    # ---------------- меню --------------------------------------------- #
    def _build_menu(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("Файлы")
        self._add_action(file_menu, "Добавить файлы…", self.on_add_files, "Ctrl+O")
        self._add_action(file_menu, "Добавить папку…", self.on_add_folder, "Ctrl+Shift+O")
        file_menu.addSeparator()
        self._add_action(file_menu, "Открыть папку результатов", lambda: open_in_file_manager(self.current_output_dir()))
        self._add_action(file_menu, "Показать результаты", self.on_show_results)
        file_menu.addSeparator()
        self._add_action(file_menu, "Выход", self.close, "Ctrl+Q")

        models_menu = bar.addMenu("Модели")
        self._add_action(models_menu, "Добавить свою модель (ссылка/путь)…", self.on_add_custom_model)
        self._add_action(models_menu, "Обновить каталог из сети", self.on_refresh_catalog)
        self._add_action(models_menu, "Открыть папку models/", lambda: open_in_file_manager(self.catalog.models_root))
        models_menu.addSeparator()
        self._add_action(models_menu, "Выгрузить модель из памяти", self._unload_model)

        service_menu = bar.addMenu("Сервис")
        self._add_action(service_menu, "Настройки…", self.on_open_settings, "Ctrl+,")
        self._add_action(service_menu, "Железо и окружение…", self.on_show_hardware)
        service_menu.addSeparator()
        self._add_action(service_menu, "Установить движок whisper.cpp", lambda: self.install_packages_interactive(
            ["pywhispercpp>=1.4.0"], "whisper.cpp"))
        self._add_action(service_menu, "Установить движок Vosk", lambda: self.install_packages_interactive(
            ["vosk>=0.3.45"], "Vosk"))
        self._add_action(service_menu, "Установить CUDA-библиотеки (GPU)", lambda: self.install_packages_interactive(
            ["-r", str(paths.ROOT_DIR / "requirements" / "gpu.txt")], "CUDA"))
        service_menu.addSeparator()
        self._add_action(service_menu, "Проверить и починить окружение (.venv)", self.on_repair_env)
        self._add_action(service_menu, "Открыть лог запуска", self.on_open_bootstrap_log)
        self._add_action(service_menu, "Открыть папку логов", lambda: open_in_file_manager(paths.logs_dir()))

        help_menu = bar.addMenu("Справка")
        self._add_action(help_menu, "Как выбрать модель", self.on_help_models)
        self._add_action(help_menu, "README (GitHub)", lambda: QDesktopServices.openUrl(
            QUrl("https://github.com/local-asr/LocalASR")))
        self._add_action(help_menu, "Сообщить о проблеме", lambda: QDesktopServices.openUrl(
            QUrl("https://github.com/local-asr/LocalASR/issues/new")))
        help_menu.addSeparator()
        self._add_action(help_menu, f"О программе ({APP_VERSION})", self.on_about)

    def _add_action(self, menu: QMenu, text: str, slot, shortcut: str = "") -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    # ---------------- статус-бар ---------------------------------------- #
    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.status_hardware = QLabel("определяю железо…")
        self.status_model = QLabel("модель не выбрана")
        self.status_ffmpeg = QLabel("ffmpeg: …")
        for widget in (self.status_hardware, self.status_model, self.status_ffmpeg):
            widget.setObjectName("Muted")
            bar.addWidget(widget)
            separator = QLabel("  ·  ")
            separator.setObjectName("Muted")
            bar.addWidget(separator)

    # ================================================================== #
    #  Старт
    # ================================================================== #
    def _startup_checks(self) -> None:
        try:
            self.hardware = registry.hardware()
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось собрать информацию о железе: %s", exc)
            self.hardware = {}

        self.models_panel.set_hardware(self.hardware)
        self.models_panel.rebuild()

        gpu = self.hardware.get("vram_mb") or 0
        self.status_hardware.setText(
            f"CPU {self.hardware.get('recommended_threads', '?')} потоков · "
            f"RAM {(self.hardware.get('ram_total_mb') or 0) / 1024:.0f} ГБ · "
            + (f"GPU {gpu / 1024:.1f} ГБ" + (" (CUDA ✓)" if self.hardware.get("cuda_ok") else " (CUDA ✗)")
               if gpu else "без GPU")
        )
        self.status_hardware.setToolTip(self.hardware.get("summary", ""))
        ffmpeg = find_ffmpeg()
        self.status_ffmpeg.setText(f"ffmpeg: {'найден' if ffmpeg else 'НЕ НАЙДЕН'}")
        self.status_ffmpeg.setToolTip(ffmpeg_version() if ffmpeg else "Декодирование пойдёт через PyAV")
        self.status_ffmpeg.setStyleSheet(
            "color: #98a2b3;" if ffmpeg else "color: #f2b544; font-weight: 600;"
        )

        for note in self.hardware.get("notes", []) or []:
            log.warning(note)

        self._restore_selected_model()
        self._update_banner()
        self._update_output_hint()

        if not self.catalog.installed():
            QTimer.singleShot(250, self._show_first_run)

    def _restore_selected_model(self) -> None:
        wanted = self.settings.selected_model_id or self.settings.last_used_model_id
        entry = self.catalog.get(wanted) if wanted else None
        if entry is not None and entry.installed:
            self._select_model(entry, save=False)
        else:
            installed = self.catalog.installed()
            if installed:
                self._select_model(installed[0], save=False)
            else:
                self.selected_model = None
                self.status_model.setText("модель не выбрана")

    def _show_first_run(self) -> None:
        if self.settings.first_run_done:
            return
        recommendation = self.catalog.recommendation(self.hardware)
        top: List[ModelEntry] = []
        if recommendation is not None:
            top.append(recommendation)
        for entry in self.catalog.sorted_for_display(self.catalog.not_installed()):
            if entry in top:
                continue
            if entry.supports_russian or entry.recommended:
                top.append(entry)
            if len(top) >= 4:
                break
        dialog = FirstRunDialog(
            hardware_summary=self.hardware.get("summary", "Информация о железе недоступна"),
            recommendations=top,
            parent=self,
        )
        dialog.downloadRequested.connect(self.on_download_model)
        dialog.exec()
        self.settings.first_run_done = True
        self.settings.save()
        self._update_banner()

    # ================================================================== #
    #  Файлы
    # ================================================================== #
    def on_add_files(self) -> None:
        start_dir = self.settings.last_open_dir or str(Path.home())
        picked, _ = QFileDialog.getOpenFileNames(self, "Выбери аудио или видео", start_dir, MEDIA_FILTER)
        if not picked:
            return
        self.settings.last_open_dir = str(Path(picked[0]).parent)
        added = self.file_list.add_paths(picked)
        self._log(f"Добавлено файлов: {added} (всего {len(self.file_list.files())})")

    def on_add_folder(self) -> None:
        start_dir = self.settings.last_open_dir or str(Path.home())
        picked = QFileDialog.getExistingDirectory(self, "Папка с записями", start_dir)
        if not picked:
            return
        self.settings.last_open_dir = picked
        added = self.file_list.add_paths([picked])
        self._log(f"Из папки добавлено файлов: {added}")
        if added == 0:
            QMessageBox.information(
                self, "Пусто",
                "В этой папке не нашлось аудио/видео файлов.\n"
                f"Поддерживаются: {', '.join(sorted(list(VIDEO_EXTENSIONS)[:8]))}… и "
                f"{', '.join(sorted(list(AUDIO_EXTENSIONS)[:8]))}…",
            )

    def on_browse_output(self) -> None:
        start_dir = self.output_edit.text().strip() or str(paths.outputs_dir())
        picked = QFileDialog.getExistingDirectory(self, "Куда сохранять результаты", start_dir)
        if picked:
            self.output_edit.setText(picked)
            self._on_settings_changed()

    def _reset_output_dir(self) -> None:
        self.output_edit.setText(str(paths.outputs_dir()))
        self._on_settings_changed()

    def current_output_dir(self) -> Path:
        text = self.output_edit.text().strip()
        return Path(text) if text else paths.outputs_dir()

    def _update_files_label(self) -> None:
        count = self.file_list.count()
        total = self.file_list.total_size()
        word = "файл" if count == 1 else "файла" if 1 < count % 10 < 5 and not 10 < count % 100 < 20 else "файлов"
        self.files_count_label.setText(
            f"{count} {word}" + (f" · {paths.human_size(total)}" if total else "")
        )
        self.drop_hint.setVisible(count == 0)
        self._update_work_button()

    # ================================================================== #
    #  Модели
    # ================================================================== #
    def on_model_selected(self, entry: ModelEntry) -> None:
        if not entry.installed:
            answer = QMessageBox.question(
                self, "Модель не установлена",
                f"«{entry.name}» ещё не скачана.\nСкачать сейчас ({entry.size_text})?",
                compat.message_box_yes() | compat.message_box_no(),
                compat.message_box_yes(),
            )
            if answer == compat.message_box_yes():
                self.on_download_model(entry)
            return
        self._select_model(entry)

    def _select_model(self, entry: ModelEntry, save: bool = True) -> None:
        self.selected_model = entry
        self.models_panel.set_selected(entry.id)
        self.status_model.setText(f"модель: {entry.name}")
        self.status_model.setToolTip(
            f"{entry.engine_label} · {entry.size_text} · {entry.memory_text}\n{entry.description}"
        )
        if save:
            self.settings.selected_model_id = entry.id
            self.settings.last_used_model_id = entry.id
            self.settings.save()
            self._log(f"Выбрана модель: {entry.name}")
        self._update_banner()
        self._update_work_button()

    def on_download_model(self, entry: ModelEntry) -> None:
        if self.download_worker is not None and self.download_worker.isRunning():
            QMessageBox.information(self, "Идёт загрузка", "Дождись окончания текущего скачивания.")
            return

        self.catalog.refresh_installed()
        entry = self.catalog.get(entry.id) or entry
        if entry.installed:
            self.on_model_selected(entry)
            return

        if not self._confirm_disk_space(entry):
            return

        dialog = ProgressDialog(
            f"Скачивание: {entry.name}",
            f"{entry.size_text} · {entry.engine_label}\nИсточник: {entry.source}",
            parent=self,
        )
        worker = ModelDownloadWorker(entry, root=self.catalog.root, parent=self)
        card = self.models_panel.card(entry.id)
        if card is not None:
            card.set_downloading(True)

        worker.progress.connect(dialog.set_fraction)
        worker.progress.connect(lambda fraction, status: card.update_progress(fraction, status) if card else None)
        worker.progress.connect(dialog.set_status)
        worker.outputLine.connect(dialog.append_log)
        dialog.cancelRequested.connect(worker.cancel)
        worker.finishedWith.connect(lambda ok, message, result: self._on_download_finished(ok, message, result, dialog, card))

        self.download_dialog = dialog
        self.download_worker = worker
        worker.start()
        dialog.exec()

    def _on_download_finished(self, ok: bool, message: str, entry, dialog: ProgressDialog, card) -> None:
        if card is not None:
            card.set_downloading(False)
        dialog.finish("Готово" if ok else "Остановлено")
        self.catalog.refresh_installed()
        self.models_panel.rebuild()
        self._restore_selected_model()

        if ok and entry is not None:
            self._log(message)
            fresh = self.catalog.get(entry.id) or entry
            self._select_model(fresh)
            QMessageBox.information(self, "Модель установлена", message + "\n\nМодель выбрана — можно работать.")
        else:
            self._log(f"Скачивание не завершилось: {message}")
            if message and not message.startswith("Скачивание отменено"):
                QMessageBox.warning(self, "Не удалось скачать модель", message)
        dialog.accept()
        self.download_worker = None
        self.download_dialog = None

    def _confirm_disk_space(self, entry: ModelEntry) -> bool:
        needed = entry.size_bytes or 0
        free = paths.free_space(self.catalog.models_root)
        if needed and free is not None and free < needed * 2:
            answer = QMessageBox.question(
                self, "Мало места на диске",
                f"Модель занимает {paths.human_size(needed)}, а свободно {paths.human_size(free)}.\n"
                "Нужно примерно вдвое больше (временно хранится .part-файл).\n\nПродолжить?",
                compat.message_box_yes() | compat.message_box_no(),
                compat.message_box_no(),
            )
            return answer == QMessageBox.Yes
        return True

    def on_delete_model(self, entry: ModelEntry) -> None:
        answer = QMessageBox.question(
            self, "Удалить модель",
            f"Удалить «{entry.name}» с диска?\n\nПапка: {entry.install_dir}\n"
            f"Освободится: {paths.human_size(entry.installed_size_bytes or entry.size_bytes)}",
            compat.message_box_yes() | compat.message_box_no(),
            QMessageBox.No,
        )
        if answer != compat.message_box_yes():
            return
        if self.selected_model is not None and self.selected_model.id == entry.id:
            registry.unload()
            self.selected_model = None
        from ..core.downloader import uninstall_model

        if uninstall_model(entry, root=self.catalog.root):
            self._log(f"Модель удалена: {entry.name}")
        else:
            QMessageBox.warning(
                self, "Не удалось удалить",
                "Папка модели лежит вне models/ или занята другим процессом.\n"
                f"Удали вручную: {entry.install_dir}",
            )
        self.catalog.refresh_installed()
        self.models_panel.rebuild()
        self._restore_selected_model()
        self._update_banner()

    def on_model_info(self, entry: ModelEntry) -> None:
        ModelInfoDialog(entry, self).exec()

    def on_add_custom_model(self) -> None:
        dialog = CustomModelDialog(self, models_root=self.catalog.models_root)
        if dialog.exec() != compat.dialog_accepted() or dialog.entry is None:
            return
        entry = dialog.entry
        existing = self.catalog.get(entry.id)
        if existing is not None and not existing.custom:
            entry.id = f"{entry.id}-custom"
        self.catalog.add_custom_entry(entry)
        self.models_panel.rebuild()
        self.models_panel.set_hardware(self.hardware)
        self._log(f"Добавлена своя модель: {entry.name} ({entry.engine_label})")
        self.on_download_model(entry)

    def on_refresh_catalog(self) -> None:
        if self.catalog_worker is not None and self.catalog_worker.isRunning():
            return
        self.models_panel.refresh_button.setEnabled(False)
        self.models_panel.refresh_button.setText("⟳ Проверяю…")
        worker = CatalogRefreshWorker(self)
        worker.finishedWith.connect(self._on_catalog_refreshed)
        self.catalog_worker = worker
        worker.start()

    def _on_catalog_refreshed(self, ok: bool, message: str) -> None:
        self.models_panel.refresh_button.setEnabled(True)
        self.models_panel.refresh_button.setText("⟳ Обновить каталог")
        if ok:
            self.catalog = ModelCatalog()
            self.models_panel.catalog = self.catalog
            self.models_panel.rebuild()
            self._restore_selected_model()
        self._log(message)

    def _unload_model(self) -> None:
        registry.unload()
        self._log("Модель выгружена из оперативной памяти")

    def _focus_models(self) -> None:
        self.splitter.setSizes([420, 820])
        self.models_panel.search.setFocus()

    def _update_banner(self) -> None:
        installed = self.catalog.installed()
        if installed and self.selected_model is not None:
            self.banner.setVisible(False)
            return
        recommendation = self.catalog.recommendation(self.hardware)
        if recommendation is None:
            self.banner.setVisible(False)
            return
        self.banner_text.setText(
            f"<b>Моделей пока нет.</b> Рекомендуем начать с «{recommendation.name}»: "
            f"{recommendation.size_text} на диске, {recommendation.memory_text}. "
            f"{recommendation.recommended_for}"
        )
        self.banner_button.setText("Скачать рекомендованную")
        self._banner_target = recommendation
        try:
            self.banner_button.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.banner_button.clicked.connect(self._on_banner_clicked)
        self.banner.setVisible(True)

    def _on_banner_clicked(self) -> None:
        target = self._banner_target
        if target is not None:
            self.on_download_model(target)
        else:
            self._focus_models()

    # ================================================================== #
    #  Работа
    # ================================================================== #
    def _update_work_button(self) -> None:
        files_ready = self.file_list.count() > 0
        model_ready = self.selected_model is not None and self.selected_model.installed
        busy = self.worker is not None and self.worker.isRunning()
        self.work_button.setEnabled(files_ready and model_ready and not busy)
        if busy:
            self.work_button.setText("РАБОТАЮ…")
        elif not files_ready:
            self.work_button.setText("Добавь файлы")
        elif not model_ready:
            self.work_button.setText("Выбери модель")
        else:
            count = self.file_list.count()
            self.work_button.setText(f"РАБОТАТЬ · {count} файл(ов)")

    def on_work(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        files = self.file_list.files()
        if not files:
            QMessageBox.information(self, "Нет файлов", "Сначала добавь аудио или видео.")
            return
        entry = self.selected_model
        if entry is None or not entry.installed:
            QMessageBox.information(
                self, "Модель не выбрана",
                "Скачай и выбери модель в правой панели — без неё распознавать нечем.\n\n"
                "Если не знаешь, что выбрать: нажми «★ рекомендуем» в баннере наверху.",
            )
            self._focus_models()
            return

        from ..asr.registry import engine_available, missing_engine_packages

        if not engine_available(entry.engine):
            packages = missing_engine_packages(entry.engine)
            answer = QMessageBox.question(
                self, "Нужен движок",
                f"Модель «{entry.name}» работает на движке {entry.engine_label}, "
                f"которого ещё нет.\n\nУстановить в .venv: {', '.join(packages)}?",
                compat.message_box_yes() | compat.message_box_no(),
                compat.message_box_yes(),
            )
            if answer != compat.message_box_yes():
                return
            self._pending_job = True
            self.install_packages_interactive(packages, f"Движок {entry.engine}")
            return

        output_dir = self.current_output_dir()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            probe_file = output_dir / ".localasr-write-test"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink(missing_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Папка недоступна", f"Не удалось писать в {output_dir}:\n{exc}")
            return

        self._collect_settings()
        self.settings.last_used_model_id = entry.id
        self.settings.selected_model_id = entry.id
        self.settings.save()

        options = JobOptions.from_settings(entry, self.settings, output_dir)
        if self.timestamps_check.isChecked() and options.formats == ["txt"]:
            options.formats = ["txt-time"]

        self.results = {}
        self.job_started_at = time.time()
        self.preview.clear()
        self.overall_progress.setRange(0, len(files))
        self.overall_progress.setValue(0)
        self.overall_label.setText(f"Обрабатываю {len(files)} файл(ов) моделью «{entry.name}»")
        self.file_progress.setRange(0, 1000)
        self.file_progress.setValue(0)
        self._update_work_button()
        self.cancel_button.setEnabled(True)
        self.results_button.setEnabled(False)

        worker = TranscribeWorker(files, options, self.settings, self)
        worker.overallProgress.connect(self._on_overall_progress)
        worker.fileProgress.connect(self._on_file_progress)
        worker.segmentReady.connect(self._on_segment_ready)
        worker.fileFinished.connect(self._on_file_finished)
        worker.logLine.connect(self._log)
        worker.finishedWith.connect(self._on_job_finished)
        worker.failed.connect(self._on_job_failed)
        worker.engineMissing.connect(self._on_engine_missing)
        self.worker = worker
        worker.start()
        self._log(
            f"Старт: {len(files)} файл(ов), модель {entry.name} ({entry.engine_label}), "
            f"язык={self.settings.language or 'авто'}, формат={self.settings.output_format}"
        )

    # ---------------- сигналы воркера ---------------------------------- #
    def _on_overall_progress(self, done: int, total: int) -> None:
        self.overall_progress.setRange(0, max(1, total))
        self.overall_progress.setValue(min(done, total))
        self.overall_label.setText(f"Файл {min(done + 1, total)} из {total}")

    def _on_file_progress(self, index: int, fraction: float, stage: str) -> None:
        if fraction < 0:
            self.file_progress.setRange(0, 0)
        else:
            self.file_progress.setRange(0, 1000)
            self.file_progress.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        try:
            name = Path(self.file_list.files()[index]).name
        except IndexError:
            name = f"файл {index + 1}"
        self.file_label.setText(f"{name} · {stage}")

    def _on_segment_ready(self, index: int, text: str, start: float, end: float) -> None:
        if not text:
            return
        self.preview.appendPlainText(f"[{int(start // 60):02d}:{int(start % 60):02d}] {text.strip()}")
        scrollbar = self.preview.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_file_finished(self, index: int, result: FileResult) -> None:
        self.results[index] = result
        self.results_button.setEnabled(bool(self.results))

    def _on_job_finished(self, ok: int, fail: int, seconds: float) -> None:
        total = len(self.file_list.files())
        produced = [path for result in self.results.values() for path in result.outputs]
        words = sum(result.words for result in self.results.values())
        audio = sum(result.audio_duration for result in self.results.values())
        speed = (audio / seconds) if seconds > 0 else 0.0

        self.overall_label.setText(
            f"Готово: {ok} из {total}"
            + (f" · {fail} с ошибкой" if fail else "")
            + f" · {words} слов · {speed:.1f}× реального времени · {seconds:.0f} с"
        )
        self.file_label.setText(", ".join(path.name for path in produced[:3]) + (" …" if len(produced) > 3 else ""))
        self._log(self.overall_label.text())

        if fail:
            failed = [result for result in self.results.values() if not result.ok]
            details = "\n".join(f"• {Path(result.source).name}: {result.error}" for result in failed[:12])
            QMessageBox.warning(
                self, "Часть файлов не обработана",
                f"{fail} из {total} файлов завершились ошибкой:\n\n{details}",
            )
        elif ok:
            self.results_button.setEnabled(True)
            if self.settings.open_output_folder_when_done and produced:
                open_in_file_manager(produced[0])
            self._toast(f"Распознано файлов: {ok}")

    def _on_job_failed(self, message: str) -> None:
        self._set_idle()
        self.overall_label.setText("Ошибка")
        self.file_label.setText(message.splitlines()[0] if message else "")
        self._log("ОШИБКА: " + message.replace("\n", " | "))
        QMessageBox.critical(self, "Ошибка при распознавании", message)

    def _on_engine_missing(self, engine: str, packages) -> None:
        self._set_idle()
        answer = QMessageBox.question(
            self, "Нужен движок",
            f"Для этой модели нужен движок «{engine}».\nУстановить {', '.join(packages)} в .venv?",
            compat.message_box_yes() | compat.message_box_no(),
            QMessageBox.Yes,
        )
        if answer == compat.message_box_yes():
            self._pending_job = True
            self.install_packages_interactive(list(packages or []), f"Движок {engine}")

    def _set_idle(self) -> None:
        """Возвращает интерфейс в состояние «ожидание»."""
        self.cancel_button.setEnabled(False)
        self.file_progress.setRange(0, 1000)
        self.file_progress.setValue(0)
        self._update_work_button()

    def on_cancel(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        self.worker.cancel()
        self.cancel_button.setEnabled(False)
        self.file_label.setText("Останавливаю… (дожидаемся текущего сегмента)")
        self._log("Запрошена отмена")

    # ================================================================== #
    #  Установка пакетов (движки, CUDA, ремонт окружения)
    # ================================================================== #
    def install_packages_interactive(self, packages: List[str], title: str = "") -> None:
        if not packages:
            return
        if self.package_worker is not None and self.package_worker.isRunning():
            QMessageBox.information(self, "Уже идёт установка", "Дождись завершения предыдущей установки.")
            return

        dialog = ProgressDialog(
            title or "Установка компонентов",
            "Устанавливаем строго в .venv проекта — системный Python не трогаем.\n"
            "Пакеты: " + ", ".join(packages),
            parent=self,
        )
        worker = PackageInstaller(packages, title, self)
        worker.outputLine.connect(dialog.append_log)
        worker.progress.connect(lambda fraction, status: dialog.set_fraction(fraction))
        worker.progress.connect(dialog.set_status)
        dialog.cancelRequested.connect(worker.cancel)
        worker.finishedWith.connect(lambda ok, message: self._on_packages_installed(ok, message, dialog))

        self.package_dialog = dialog
        self.package_worker = worker
        worker.start()
        dialog.exec()

    def _on_packages_installed(self, ok: bool, message: str, dialog: ProgressDialog) -> None:
        dialog.finish("Готово" if ok else "Остановлено")
        self._log(message)
        registry.hardware(refresh=True)
        self.hardware = registry.hardware()
        self.models_panel.set_hardware(self.hardware)
        if not ok:
            QMessageBox.warning(self, "Установка не завершилась", message)
        elif getattr(self, "_pending_job", False):
            self._pending_job = False
            QTimer.singleShot(250, self.on_work)
        dialog.accept()
        self.package_worker = None
        self.package_dialog = None

    def on_repair_env(self) -> None:
        answer = QMessageBox.question(
            self, "Проверка окружения",
            "Перепроверить все базовые зависимости в .venv и доустановить недостающее?\n\n"
            "Это безопасно: системный Python не затрагивается.",
            compat.message_box_yes() | compat.message_box_no(),
            QMessageBox.Yes,
        )
        if answer != compat.message_box_yes():
            return
        requirements = paths.ROOT_DIR / "requirements" / "base.txt"
        if requirements.is_file():
            self.install_packages_interactive(["-r", str(requirements)], "Проверка окружения")
        else:
            QMessageBox.warning(self, "Не найдено", f"Нет файла {requirements}")

    def on_open_bootstrap_log(self) -> None:
        log_file = paths.logs_dir() / "bootstrap.log"
        error_log = paths.ROOT_DIR / "logs" / "bootstrap_error.log"
        target = log_file if log_file.is_file() else error_log
        if not target.is_file():
            QMessageBox.information(self, "Логи", "Лог установки ещё не создан.")
            return
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            QMessageBox.warning(self, "Логи", f"Не удалось прочитать: {exc}")
            return
        dialog = ProgressDialog(f"Лог: {target.name}", str(target), self)
        dialog.set_fraction(1.0)
        dialog.set_status(f"{target}")
        dialog.append_log(text[-60000:])
        dialog.finish("Только чтение")
        dialog.cancel_button.setText("Закрыть")
        dialog.exec()

    # ================================================================== #
    #  Прочее
    # ================================================================== #
    def on_open_settings(self) -> None:
        self._collect_settings()
        dialog = SettingsDialog(self.settings, self.hardware, self)
        if dialog.exec() == compat.dialog_accepted():
            self._apply_settings_to_ui()
            self._log("Настройки сохранены")

    def on_show_hardware(self) -> None:
        hardware = registry.hardware(refresh=True)
        self.hardware = hardware
        self.models_panel.set_hardware(hardware)
        HardwareDialog(
            summary=hardware.get("summary", "—"),
            ffmpeg_text=ffmpeg_version(),
            notes=list(hardware.get("notes") or []),
            parent=self,
        ).exec()

    def on_show_results(self) -> None:
        ResultsDialog(self.results, self).exec()

    def on_help_models(self) -> None:
        QMessageBox.information(
            self, "Как выбрать модель",
            "Коротко:\n\n"
            "• Русские лекции, есть 8+ ГБ RAM или любая GPU →\n"
            "   «Whisper Large v3 Turbo · Russian (int8, CT2)» — наш основной выбор.\n"
            "• Мало памяти (4–6 ГБ), хочется точности →\n"
            "   «Whisper Large v3 Turbo q5_0 (whisper.cpp)» — 574 МБ.\n"
            "• Язык заранее неизвестен или запись многоязычная →\n"
            "   «Whisper Large v3 Turbo (multilingual)».\n"
            "• Нужен максимум качества и время не важно →\n"
            "   «Whisper Large v3» (3.1 ГБ).\n"
            "• Совсем слабое железо / надо просто быстро →\n"
            "   Vosk Russian (45 МБ) — но без пунктуации и с ошибками.\n\n"
            "В карточках указаны размер файла, потребление памяти, языки,\n"
            "оценки точности и скорости — сравнивай и выбирай.",
        )

    def on_about(self) -> None:
        QMessageBox.about(
            self, f"О {APP_NAME}",
            f"<h3>{APP_NAME} {APP_VERSION}</h3>"
            "<p>Локальное распознавание речи: аудио и видео → текст, субтитры, JSON.</p>"
            "<p>Движки: faster-whisper (CTranslate2), whisper.cpp (GGML), Vosk (Kaldi).<br>"
            "GUI: PySide6. Всё устанавливается в <code>.venv</code> внутри папки проекта.</p>"
            f"<p>Python: {sys.version.split()[0]}<br>Интерпретатор: {sys.executable}<br>"
            f"Проект: {paths.ROOT_DIR}</p>",
        )

    # ---------------- служебное ---------------------------------------- #
    def _collect_settings(self) -> None:
        self.settings.language = self.language_combo.currentData() or ""
        self.settings.output_format = self.format_combo.currentData() or "txt"
        self.settings.output_dir = self.output_edit.text().strip()
        self.settings.use_vad = self.vad_check.isChecked()
        self.settings.suppress_hallucinations = self.clean_check.isChecked()
        self.settings.paragraphs = self.paragraph_check.isChecked()
        self.settings.add_timestamps_to_txt = self.timestamps_check.isChecked()
        self.settings.word_timestamps = self.wordts_check.isChecked()
        self.settings.translate_to_english = self.translate_check.isChecked()
        self.settings.beam_size = int(self.beam_combo.currentData() or 5)
        self.settings.batched = self.batched_check.isChecked()
        self.settings.hotwords = self.hotwords_edit.text().strip()
        self.settings.initial_prompt = self.prompt_edit.text().strip()
        self.settings.open_output_folder_when_done = self.open_folder_check.isChecked()
        self.settings.save()

    def _apply_settings_to_ui(self) -> None:
        index = self.language_combo.findData(self.settings.language)
        self.language_combo.setCurrentIndex(index if index >= 0 else 1)
        index = self.format_combo.findData(self.settings.output_format)
        self.format_combo.setCurrentIndex(index if index >= 0 else 0)
        self.output_edit.setText(self.settings.output_dir or str(paths.outputs_dir()))
        self.vad_check.setChecked(self.settings.use_vad)
        self.clean_check.setChecked(self.settings.suppress_hallucinations)
        self.paragraph_check.setChecked(self.settings.paragraphs)
        self.timestamps_check.setChecked(self.settings.add_timestamps_to_txt)
        self.wordts_check.setChecked(self.settings.word_timestamps)
        self.translate_check.setChecked(self.settings.translate_to_english)
        index = self.beam_combo.findData(int(self.settings.beam_size or 5))
        self.beam_combo.setCurrentIndex(index if index >= 0 else 2)
        self.batched_check.setChecked(bool(self.settings.batched))
        self.hotwords_edit.setText(self.settings.hotwords)
        self.prompt_edit.setText(self.settings.initial_prompt)
        self.open_folder_check.setChecked(bool(self.settings.open_output_folder_when_done))
        self._update_output_hint()

    def _on_settings_changed(self, *_args) -> None:
        self._collect_settings()
        self._update_output_hint()

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_frame.setVisible(checked)
        self.advanced_toggle.setText("▾ Расширенные настройки" if checked else "▸ Расширенные настройки")

    def _update_output_hint(self) -> None:
        output_dir = self.current_output_dir()
        fmt = self.format_combo.currentData() or "txt"
        sample = paths.sanitize_stem("лекция 1.mp4")
        suffix = paths.output_suffix(fmt)
        self.output_hint.setText(
            f"Результат: {paths.short_path(output_dir, 46)} / «{sample}{suffix}» "
            f"(имя файла = имени оригинала)"
        )

    def _log(self, message: str) -> None:
        text = message.strip()
        if not text:
            return
        self.log_view.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")
        log.info(text)

    def _on_log_record(self, level: str, message: str) -> None:
        try:
            self.log_view.appendPlainText(message)
        except RuntimeError:
            detach_qt_sink(self.log_handler)

    def _toast(self, message: str) -> None:
        self.statusBar().showMessage(message, 6000)

    # ---------------- drag&drop окна ----------------------------------- #
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
        if urls:
            added = self.file_list.add_paths(urls)
            self._log(f"Перетаскиванием добавлено: {added}")
            event.acceptProposedAction()

    # ---------------- геометрия/закрытие -------------------------------- #
    def _restore_geometry(self) -> None:
        try:
            if self.settings.window_geometry:
                self.restoreGeometry(QByteArray.fromBase64(self.settings.window_geometry.encode("ascii")))
            if self.settings.window_state:
                self.restoreState(QByteArray.fromBase64(self.settings.window_state.encode("ascii")))
            if self.settings.splitter_sizes:
                self.splitter.setSizes([int(v) for v in self.settings.splitter_sizes])
            self.log_dock.setVisible(bool(self.settings.show_log_panel))
        except Exception as exc:  # noqa: BLE001
            log.debug("Не удалось восстановить геометрию окна: %s", exc)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            answer = QMessageBox.question(
                self, "Идёт обработка",
                "Распознавание ещё не закончилось. Остановить и выйти?",
                compat.message_box_yes() | compat.message_box_no(),
                compat.message_box_no(),
            )
            if answer != compat.message_box_yes():
                event.ignore()
                return
            self.worker.cancel()
            self.worker.wait(5000)

        for worker in (self.download_worker, self.package_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(3000)

        self._collect_settings()
        try:
            self.settings.window_geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
            self.settings.window_state = bytes(self.saveState().toBase64()).decode("ascii")
            self.settings.splitter_sizes = [int(v) for v in self.splitter.sizes()]
            self.settings.show_log_panel = self.log_dock.isVisible()
            self.settings.save()
        except Exception as exc:  # noqa: BLE001
            log.debug("Не удалось сохранить геометрию: %s", exc)

        detach_qt_sink(self.log_handler)
        registry.unload()
        event.accept()


# --------------------------------------------------------------------------- #
#  Таблица результатов
# --------------------------------------------------------------------------- #


class ResultsDialog(QDialog):
    def __init__(self, results: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("Результаты")
        self.setModal(True)
        self.resize(920, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(10)
        root.addWidget(make_label("Что получилось", "H2"))

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Файл", "Статус", "Результат", "Слов", "Скорость", "Время"])
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, compat.header_stretch_mode())
        header.setSectionResizeMode(2, compat.header_stretch_mode())
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(compat.selection_behaviour_rows())
        self.table.setEditTriggers(compat.no_edit_triggers())
        self.table.setAlternatingRowColors(True)

        for index in sorted(results):
            result: FileResult = results[index]
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(Path(result.source).name))
            status = QTableWidgetItem("✓ готово" if result.ok else "✗ ошибка")
            status.setForeground(compat.COLOR_GREEN if result.ok else compat.COLOR_RED)
            self.table.setItem(row, 1, status)
            outputs = QTableWidgetItem(", ".join(p.name for p in result.outputs) or (result.error or "—"))
            outputs.setToolTip(result.error or "\n".join(str(p) for p in result.outputs))
            self.table.setItem(row, 2, outputs)
            self.table.setItem(row, 3, QTableWidgetItem(str(result.words)))
            self.table.setItem(row, 4, QTableWidgetItem(f"{result.speed_x:.1f}×"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{result.elapsed_seconds:.0f} с"))
            if result.ok and result.outputs:
                self.table.item(row, 2).setData(compat.ROLE_USER, str(result.outputs[0]))

        root.addWidget(self.table, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        open_button = QPushButton("Открыть файл")
        open_button.setObjectName("Primary")
        open_button.clicked.connect(self._open_current)
        buttons.addWidget(open_button)
        folder_button = QPushButton("Открыть папку")
        folder_button.setObjectName("Ghost")
        folder_button.clicked.connect(self._open_folder)
        buttons.addWidget(folder_button)
        close_button = QPushButton("Закрыть")
        close_button.setObjectName("Ghost")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        root.addLayout(buttons)

        self.table.itemDoubleClicked.connect(lambda *_: self._open_current())

    def _current_path(self) -> Optional[Path]:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 2)
        data = item.data(compat.ROLE_USER) if item else None
        return Path(data) if data else None

    def _open_current(self) -> None:
        path = self._current_path()
        if path is None:
            QMessageBox.information(self, "Результаты", "Выбери строку с готовым файлом.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_folder(self) -> None:
        path = self._current_path()
        open_in_file_manager(path or paths.outputs_dir())
