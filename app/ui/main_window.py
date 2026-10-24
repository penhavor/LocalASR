"""Главное окно LocalASR."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QByteArray, QUrl, QTimer, Signal
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QImage, QKeySequence, QPainter, QPixmap
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
    REPO_URL,
    TORCH_CUDA_INDEX_URL,
    TORCH_CUDA_MIN_DRIVER_MAJOR,
    TORCH_ENGINE_NAMES,
    VIDEO_EXTENSIONS,
)
from ..core.ffmpeg import ffmpeg_version, find_ffmpeg
from ..core.models_catalog import ModelCatalog, ModelEntry
from ..logutils import attach_qt_sink, detach_qt_sink, get_logger
from ..settings import Settings
from . import compat
from .dialogs import (
    CustomModelDialog,
    DiagnosticsDialog,
    HardwareDialog,
    ModelInfoDialog,
    ProgressDialog,
    SettingsDialog,
)
from .models_panel import ModelListPanel
from .widgets import FileListWidget, FlowLayout, ToolTipMenu, make_label, open_in_file_manager
from .workers import (
    CatalogRefreshWorker,
    FileRestoreWorker,
    ModelDownloadWorker,
    PackageInstaller,
    TorchCudaProbeWorker,
    TranscribeWorker,
)

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
    """Иконка из встроенного SVG (не зависит от внешних файлов и сети).

    Рендерим сразу в несколько размеров, чтобы и в заголовке окна, и в панели
    задач Windows иконка была чёткой, а не мыльным квадратом.
    """
    icon = QIcon()
    data = QByteArray(_ICON_SVG.encode("utf-8"))
    try:  # векторный рендер в каждый размер — максимально чётко
        from PySide6.QtSvg import QSvgRenderer

        renderer = QSvgRenderer(data)
        if renderer.isValid():
            for size in (16, 20, 24, 32, 48, 64, 96, 128, 256):
                pixmap = QPixmap(size, size)
                pixmap.fill(compat.COLOR_TRANSPARENT)
                painter = QPainter(pixmap)
                renderer.render(painter)
                painter.end()
                icon.addPixmap(pixmap)
            if not icon.isNull():
                return icon
    except Exception as exc:  # noqa: BLE001
        log.debug("QSvgRenderer недоступен, рисуем растром: %s", exc)
    try:
        image = QImage.fromData(data)
        if not image.isNull():
            for size in (16, 24, 32, 48, 64, 128, 256):
                icon.addPixmap(
                    QPixmap.fromImage(image).scaled(
                        size, size, compat.KEEP_ASPECT_SMOOTH, compat.SMOOTH_TRANSFORM
                    )
                )
    except Exception as exc:  # noqa: BLE001
        log.debug("Не удалось собрать иконку из SVG: %s", exc)
    return icon


# --------------------------------------------------------------------------- #
#  Главное окно
# --------------------------------------------------------------------------- #


class MainWindow(QMainWindow):
    #: записи лога из фоновых потоков приходят в GUI-поток очередью (безопасно)
    logRecord = Signal(str, str)

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
        self.restore_worker: Optional[FileRestoreWorker] = None
        self.log_handler = None

        self.results: dict[int, FileResult] = {}
        self.job_started_at: float = 0.0
        self._pending_job: bool = False
        self._working: bool = False
        self._job_total_files: int = 1
        self._job_done_files: int = 0
        self._last_job_engine: str = ""
        self._last_install_torch_cuda: bool = False
        self.torch_probe: Optional[TorchCudaProbeWorker] = None
        # состояние PyTorch в .venv: probed/installed/cuda/detail
        self.gpu_state: dict = {"probed": False, "installed": False, "cuda": False, "detail": ""}

        self.setWindowTitle(f"{APP_NAME} — локальное распознавание речи")
        self.setWindowIcon(app_icon())
        # Компактный минимум: интерфейс построен так, что в узком окне всё
        # переносится по строкам, а остальное можно доскроллить.
        self.setMinimumSize(940, 620)
        self.setAcceptDrops(True)

        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        self._restore_geometry()

        # ВАЖНО: лог-записи рождаются в фоновых потоках, а виджеты Qt можно
        # трогать только из GUI-потока — поэтому мост идёт через сигнал
        # (автоматически доставляется очередью в GUI-поток).
        self.logRecord.connect(self._on_log_record)
        self.log_handler = attach_qt_sink(self.logRecord.emit)
        log.info("GUI запущен (%s %s, Python %s)", APP_NAME, APP_VERSION, sys.version.split()[0])

        # Медленный таймер синхронизирует «рантаймовые» состояния кнопок/пунктов
        # меню (например, «Выгрузить модель» активен, только когда модель в памяти).
        self._state_timer = QTimer(self)
        self._state_timer.setInterval(2000)
        self._state_timer.timeout.connect(self._sync_runtime_actions)
        self._state_timer.start()
        self._sync_runtime_actions()

        # Дебаунс сохранения геометрии окна: любое движение/ресайз сохраняется
        # не мгновенно, а через паузу — настройки переживают даже аварийный выход.
        self._geo_timer = QTimer(self)
        self._geo_timer.setSingleShot(True)
        self._geo_timer.setInterval(800)
        self._geo_timer.timeout.connect(self._save_geometry_now)

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
        splitter.splitterMoved.connect(lambda *_: self._schedule_geometry_save())
        self.splitter = splitter

        self.setCentralWidget(splitter)
        self._build_log_dock()
        self._update_work_button()

    # ---------------- левая колонка ------------------------------------ #
    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(compat.frame_no_frame())
        # в очень узком окне лучше дать горизонтальную прокрутку, чем обрезать кнопки
        scroll.setHorizontalScrollBarPolicy(compat.SCROLLBAR_AS_NEEDED)

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

        buttons = FlowLayout(spacing=8)

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
        self.add_folder_button.setToolTip("Добавить все аудио/видео из папки (рекурсивно, включая подпапки)")
        self.add_folder_button.clicked.connect(self.on_add_folder)
        buttons.addWidget(self.add_folder_button)

        self.clear_files_button = QPushButton("Очистить")
        self.clear_files_button.setObjectName("Ghost")
        self.clear_files_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.clear_files_button.setToolTip("Убрать из списка ВСЕ файлы (сами файлы на диске не удаляются)")
        self.clear_files_button.clicked.connect(lambda: self.file_list.clear_all())
        buttons.addWidget(self.clear_files_button)

        self.remove_selected_button = QPushButton("Убрать выбранные")
        self.remove_selected_button.setObjectName("Ghost")
        self.remove_selected_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.remove_selected_button.setToolTip("Убрать выделенные строки (Del). Выделение — мышью с Ctrl/Shift, как в проводнике")
        self.remove_selected_button.setShortcut(QKeySequence("Delete"))
        self.remove_selected_button.clicked.connect(lambda: self.file_list.remove_selected())
        buttons.addWidget(self.remove_selected_button)
        files_layout.addLayout(buttons)

        self.file_list = FileListWidget()
        self.file_list.setMinimumHeight(180)
        self.file_list.setToolTip(
            "Очередь файлов на распознавание.\n"
            "Выделение — мышью с Ctrl/Shift, удаление выделенного — клавишей Del\n"
            "или кнопкой «Убрать выбранные». Сами файлы на диске не изменяются."
        )
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

        language_label = make_label("Язык речи", "Dim")
        language_label.setToolTip(
            "Язык, на котором говорят в записи.\n"
            "«Определить автоматически» — модель сама поймёт язык (чуть медленнее и на\n"
            "коротких/шумных записях менее надёжно, чем явное указание)."
        )
        grid.addWidget(language_label, 0, 0)
        self.language_combo = QComboBox()
        for code, title in LANGUAGES:
            self.language_combo.addItem(title, code)
        self.language_combo.setToolTip(language_label.toolTip())
        language_index = self.language_combo.findData(self.settings.language)
        self.language_combo.setCurrentIndex(language_index if language_index >= 0 else 1)
        self.language_combo.currentIndexChanged.connect(self._on_settings_changed)
        grid.addWidget(self.language_combo, 0, 1, 1, 3)

        save_label = make_label("Куда сохранять", "Dim")
        save_label.setToolTip(
            "Папка, куда лягут готовые .txt-файлы (имя файла = имени оригинала).\n"
            "По умолчанию — outputs/ внутри папки проекта."
        )
        grid.addWidget(save_label, 1, 0)
        save_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setText(self.settings.output_dir or str(paths.outputs_dir()))
        self.output_edit.setPlaceholderText(str(paths.outputs_dir()))
        self.output_edit.setToolTip(save_label.toolTip())
        self.output_edit.editingFinished.connect(self._on_settings_changed)
        save_row.addWidget(self.output_edit, stretch=1)
        browse_output = QPushButton("Обзор…")
        browse_output.setObjectName("Ghost")
        browse_output.setCursor(compat.CURSOR_POINTING_HAND)
        browse_output.setToolTip("Выбрать папку для результатов в проводнике")
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

        # галочки переносятся по строкам (FlowLayout), а не обрезаются краем
        toggles = FlowLayout(spacing=14)
        self.vad_check = QCheckBox("VAD (резать тишину)")
        self.vad_check.setChecked(self.settings.use_vad)
        self.vad_check.setToolTip(
            "Silero VAD: вырезает паузы, шум и музыку до распознавания.\n"
            "Плюсы: быстрее и меньше галлюцинаций («Спасибо за просмотр» на тишине).\n"
            "Выключи только если в записи важна сама пауза (например, речь с долгими остановками)."
        )
        toggles.addWidget(self.vad_check)
        self.vad_check.toggled.connect(self._on_settings_changed)

        self.clean_check = QCheckBox("Анти-галлюцинации")
        self.clean_check.setChecked(self.settings.suppress_hallucinations)
        self.clean_check.setToolTip(
            "Вычищает «[музыка]», «[аплодисменты]», «Спасибо за просмотр», "
            "субтитрные вставки и зацикленные повторы фраз.\n"
            "Держи включённым: на лекциях с тишиной и музыкой без этого в тексте появляется мусор."
        )
        self.clean_check.toggled.connect(self._on_settings_changed)
        toggles.addWidget(self.clean_check)

        self.paragraph_check = QCheckBox("Абзацы")
        self.paragraph_check.setChecked(self.settings.paragraphs)
        self.paragraph_check.setToolTip(
            "Склеивает короткие сегменты распознавания в абзацы по смысловым паузам.\n"
            "Выключи, если нужен «телеграфный» формат: каждая фраза с новой строки."
        )
        self.paragraph_check.toggled.connect(self._on_settings_changed)
        toggles.addWidget(self.paragraph_check)

        self.timestamps_check = QCheckBox("Таймкоды в .txt")
        self.timestamps_check.setChecked(self.settings.add_timestamps_to_txt)
        self.timestamps_check.setToolTip(
            "Добавляет в .txt метки времени вида [00:12:34] перед фразами — удобно искать\n"
            "место в записи. Результат всё равно остаётся обычным .txt-файлом."
        )
        self.timestamps_check.toggled.connect(self._on_settings_changed)
        toggles.addWidget(self.timestamps_check)

        self.advanced_toggle = QPushButton("▸ Расширенные настройки")
        self.advanced_toggle.setObjectName("Link")
        self.advanced_toggle.setCursor(compat.CURSOR_POINTING_HAND)
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(bool(self.settings.advanced_open))
        self.advanced_toggle.setToolTip(
            "Тонкие настройки: слова-подсказки, подсказка стилю, beam, batched-режим,\n"
            "таймкоды слов, перевод, автооткрытие папки."
        )
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        toggles.addWidget(self.advanced_toggle)
        options_layout.addLayout(toggles)

        self.advanced_frame = QFrame()
        advanced_layout = QVBoxLayout(self.advanced_frame)
        advanced_layout.setContentsMargins(0, 6, 0, 0)
        advanced_layout.setSpacing(8)
        self.advanced_frame.setVisible(bool(self.settings.advanced_open))
        if self.settings.advanced_open:
            self.advanced_toggle.setText("▾ Расширенные настройки")

        hot_row = QHBoxLayout()
        hot_label = make_label("Слова-подсказки (термины лекции)", "Dim")
        hot_label.setToolTip(
            "Слова, которые модель будет «держать в голове» при распознавании.\n"
            "Помогает с редкими терминами и фамилиями: без подсказки Whisper часто пишет их на слух.\n"
            "Поддерживается движком faster-whisper; остальные движки подсказку игнорируют."
        )
        hot_row.addWidget(hot_label)
        self.hotwords_edit = QLineEdit()
        self.hotwords_edit.setPlaceholderText("трансформер, градиентный спуск, Kubernetes — через запятую")
        self.hotwords_edit.setText(self.settings.hotwords)
        self.hotwords_edit.setToolTip(hot_label.toolTip())
        self.hotwords_edit.editingFinished.connect(self._on_settings_changed)
        hot_row.addWidget(self.hotwords_edit, stretch=1)
        advanced_layout.addLayout(hot_row)

        prompt_row = QHBoxLayout()
        prompt_label = make_label("Подсказка стилю", "Dim")
        prompt_label.setToolTip(
            "initial_prompt: короткий текст-пример стиля и тематики. Модель продолжает его,\n"
            "поэтому держит стиль (научный, разговорный) и верно пишет термины из подсказки.\n"
            "Не дублируй сюда весь словарь — для слов есть поле «Слова-подсказки»."
        )
        prompt_row.addWidget(prompt_label)
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Например: Лекция по машинному обучению, русский язык, научный стиль.")
        self.prompt_edit.setText(self.settings.initial_prompt)
        self.prompt_edit.setToolTip(prompt_label.toolTip())
        self.prompt_edit.editingFinished.connect(self._on_settings_changed)
        prompt_row.addWidget(self.prompt_edit, stretch=1)
        advanced_layout.addLayout(prompt_row)

        extra_row = FlowLayout(spacing=12)
        self.beam_combo = QComboBox()
        for value, title in ((1, "beam 1 (быстро)"), (3, "beam 3"), (5, "beam 5 (баланс)"), (8, "beam 8 (точность)")):
            self.beam_combo.addItem(title, value)
        beam_index = self.beam_combo.findData(int(self.settings.beam_size or 5))
        self.beam_combo.setCurrentIndex(beam_index if beam_index >= 0 else 2)
        self.beam_combo.currentIndexChanged.connect(self._on_settings_changed)
        self.beam_combo.setToolTip(
            "Beam size — сколько вариантов продолжения фразы держит декодер.\n"
            "1 — быстрее всего; 5 — оптимально для речи; 8 — чуть точнее, но заметно медленнее.\n"
            "Для обычных лекций оставляй 5."
        )
        extra_row.addWidget(self.beam_combo)

        self.batched_check = QCheckBox("Batched (до 4× быстрее на GPU)")
        self.batched_check.setChecked(bool(self.settings.batched))
        self.batched_check.setToolTip(
            "Пакетный декодинг (BatchedInferencePipeline) у faster-whisper: несколько кусков\n"
            "аудио считаются одним батчем. Выигрыш до 4× на GPU; на CPU почти не помогает.\n"
            "Требует больше памяти — при нехватке RAM выключи."
        )
        self.batched_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.batched_check)

        self.wordts_check = QCheckBox("Таймкоды слов")
        self.wordts_check.setChecked(bool(self.settings.word_timestamps))
        self.wordts_check.setToolTip(
            "Считать таймкоды каждого слова (нужно только для точных субтитров srt/vtt).\n"
            "Замедляет распознавание; для обычного .txt-результата выключи."
        )
        self.wordts_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.wordts_check)

        self.translate_check = QCheckBox("Перевод в EN")
        self.translate_check.setChecked(bool(self.settings.translate_to_english))
        self.translate_check.setToolTip(
            "task=translate: модель сразу выдаёт английский перевод услышанного.\n"
            "Для конспекта русской лекции выключи; полезно для иноязычных докладов."
        )
        self.translate_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.translate_check)

        self.open_folder_check = QCheckBox("Открыть папку после завершения")
        self.open_folder_check.setChecked(bool(self.settings.open_output_folder_when_done))
        self.open_folder_check.setToolTip(
            "Когда очередь догорит, открыть папку с результатами в проводнике.\n"
            "Удобно при разовых прогонах; мешает, если запускаешь папками на ночь."
        )
        self.open_folder_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.open_folder_check)

        settings_button = QPushButton("Все настройки…")
        settings_button.setObjectName("Ghost")
        settings_button.setCursor(compat.CURSOR_POINTING_HAND)
        settings_button.setToolTip(
            "Полный диалог настроек: устройство (CPU/GPU), точность вычислений, потоки,\n"
            "параллельные файлы, движки и CUDA, справка о программе."
        )
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

        # --- выбор модели: выпадающий список ВСЕХ скачанных моделей --------- #
        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_label = make_label("Модель", "Dim")
        model_label.setToolTip(
            "Какую из СКАЧАННЫХ моделей использовать для распознавания.\n"
            "Выбрать можно только одну; пока выбрано «Не выбрано», кнопка «РАБОТАТЬ»\n"
            "неактивна. Скачать новые модели — в правой панели окна."
        )
        model_row.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.setToolTip(model_label.toolTip())
        self.model_combo.setCursor(compat.CURSOR_POINTING_HAND)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        model_row.addWidget(self.model_combo, stretch=1)
        work_layout.addLayout(model_row)
        self.rebuild_model_combo()

        # --- GPU-статус/кнопка: предлагает CUDA-сборку PyTorch, когда есть
        # видеокарта NVIDIA, но torch считает на процессоре -----------------
        self.gpu_button = QPushButton("Определяю железо…")
        self.gpu_button.setObjectName("Ghost")
        self.gpu_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.gpu_button.setEnabled(False)
        self.gpu_button.clicked.connect(self._install_torch_cuda)
        work_layout.addWidget(self.gpu_button)

        self.overall_progress = QProgressBar()
        self.overall_progress.setToolTip("Общий прогресс очереди файлов")
        self.overall_progress.setRange(0, 1000)
        self.overall_progress.setValue(0)
        # %p — именно ПРОЦЕНТ (Qt сам пересчитывает из диапазона 0..1000);
        # %v показал бы «сырое» значение — отсюда были «1000%» в финале
        self.overall_progress.setFormat("%p%")
        work_layout.addWidget(self.overall_progress)

        self.overall_label = make_label("Готов к работе", "Dim", word_wrap=True)
        work_layout.addWidget(self.overall_label)

        self.file_progress = QProgressBar()
        self.file_progress.setToolTip("Прогресс текущего файла")
        self.file_progress.setObjectName("Subtle")
        self.file_progress.setRange(0, 1000)
        self.file_progress.setValue(0)
        self.file_progress.setFixedHeight(6)
        self.file_progress.setFormat("")  # текст на 6-пиксельной полосе не помещается
        work_layout.addWidget(self.file_progress)

        self.file_label = make_label("", "Muted", word_wrap=True)
        work_layout.addWidget(self.file_label)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setToolTip(
            "Живая лента распознанного текста текущего файла\n"
            "(таймкод начала фразы + сама фраза). Итоговый результат — в файле."
        )
        self.preview.setPlaceholderText("Здесь появится распознанный текст по мере работы…")
        self.preview.setMaximumHeight(96)
        self.preview.setMaximumBlockCount(1500)  # память не растёт бесконечно
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
        action_row.addWidget(self.work_button, stretch=1)

        # Кнопка-переключатель: на время работы «РАБОТАТЬ» заменяется «ОТМЕНИТЬ»
        self.cancel_button = QPushButton("ОТМЕНИТЬ")
        self.cancel_button.setObjectName("Danger")
        self.cancel_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.cancel_button.setMinimumHeight(48)
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.on_cancel)
        action_row.addWidget(self.cancel_button, stretch=1)
        work_layout.addLayout(action_row)

        results_row = FlowLayout(spacing=8)
        self.results_button = QPushButton("Показать результаты")
        self.results_button.setObjectName("Ghost")
        self.results_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.results_button.setToolTip(
            "Таблица обработанных файлов: статус, имена результатов, число слов,\n"
            "скорость и время. Появляется после первого завершённого запуска."
        )
        self.results_button.setEnabled(False)
        self.results_button.clicked.connect(self.on_show_results)
        results_row.addWidget(self.results_button)
        self.open_output_button = QPushButton("Открыть папку результатов")
        self.open_output_button.setObjectName("Ghost")
        self.open_output_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.open_output_button.setToolTip(
            "Открыть в проводнике папку, куда сохраняются готовые тексты\n"
            "(настраивается в поле «Куда сохранять»)"
        )
        self.open_output_button.clicked.connect(lambda: open_in_file_manager(self.current_output_dir()))
        results_row.addWidget(self.open_output_button)
        work_layout.addLayout(results_row)

        layout.addWidget(work_box, stretch=2)

        scroll.setWidget(page)
        return scroll

    # ---------------- правая колонка ----------------------------------- #
    def _build_right_panel(self) -> QWidget:
        self.models_panel = ModelListPanel(self.catalog, self.hardware, settings=self.settings)
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
        # подсказка журналу не нужна: и так понятно, что это лог событий,
        # а всплывающее окно при наведении только отвлекает при прокрутке
        self.log_view.setToolTip("")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("Журнал событий…")
        self.log_dock.setWidget(self.log_view)
        self.log_dock.setToolTip("")
        self.addDockWidget(compat.BOTTOM_DOCK_AREA, self.log_dock)
        self.log_dock.setVisible(bool(self.settings.show_log_panel))
        self.log_dock.visibilityChanged.connect(lambda *_: self._schedule_geometry_save())

    # ---------------- меню --------------------------------------------- #
    def _build_menu(self) -> None:
        bar = self.menuBar()

        file_menu = ToolTipMenu("Файлы", self)
        bar.addMenu(file_menu)
        file_menu.menuAction().setToolTip(
            "Работа с файлами: добавить аудио/видео (или целую папку),\n"
            "открыть результаты, показать таблицу обработанных файлов, выйти."
        )
        self._add_action(file_menu, "Добавить файлы…", self.on_add_files, "Ctrl+O",
                         tip="Выбрать аудио/видео файлы для распознавания (можно несколько сразу)")
        self._add_action(file_menu, "Добавить папку…", self.on_add_folder, "Ctrl+Shift+O",
                         tip="Добавить все аудио/видео из папки, включая подпапки")
        file_menu.addSeparator()
        self._add_action(file_menu, "Открыть папку результатов",
                         lambda: open_in_file_manager(self.current_output_dir()),
                         tip="Открыть в проводнике папку, куда сохраняются готовые тексты")
        self._add_action(file_menu, "Показать результаты", self.on_show_results,
                         tip="Таблица обработанных файлов: статусы, имена результатов, скорость")
        file_menu.addSeparator()
        self._add_action(file_menu, "Выход", self.close, "Ctrl+Q",
                         tip="Закрыть программу (идущая обработка будет остановлена после подтверждения)")

        models_menu = ToolTipMenu("Модели", self)
        bar.addMenu(models_menu)
        models_menu.menuAction().setToolTip(
            "Модели распознавания: добавить свою (ссылка/путь), обновить каталог,\n"
            "открыть папку models/, выгрузить загруженную модель из памяти."
        )
        self._add_action(models_menu, "Добавить свою модель (ссылка/путь)…", self.on_add_custom_model,
                         tip="Модель не из каталога: репозиторий Hugging Face, прямая ссылка или папка на диске")
        self._add_action(models_menu, "Обновить каталог из сети", self.on_refresh_catalog,
                         tip="Проверить онлайн-каталог: не появились ли новые модели")
        self._add_action(models_menu, "Открыть папку models/",
                         lambda: open_in_file_manager(self.catalog.models_root),
                         tip="Папка, куда скачиваются модели (создаётся автоматически)")
        models_menu.addSeparator()
        self.unload_action = self._add_action(
            models_menu, "Выгрузить модель из памяти", self._unload_model,
            tip="Освободить оперативную память (и видеопамять), занятую загруженной моделью.\n"
                "Доступно, только когда модель действительно загружена; при следующем запуске\n"
                "она загрузится заново (несколько секунд).",
        )
        self.unload_action.setEnabled(False)

        service_menu = ToolTipMenu("Сервис", self)
        bar.addMenu(service_menu)
        service_menu.menuAction().setToolTip(
            "Сервисные инструменты: настройки программы, сведения о железе,\n"
            "диагностика и восстановление, журналы (лог запуска, папка логов)."
        )
        self._add_action(service_menu, "Настройки…", self.on_open_settings, "Ctrl+,",
                         tip="Качество текста, устройство (CPU/GPU), журнал (лог-файлы), сброс настроек по умолчанию")
        self._add_action(service_menu, "Железо и окружение…", self.on_show_hardware,
                         tip="Что видит программа: процессор, память, видеокарта, ffmpeg")
        service_menu.addSeparator()
        self._add_action(service_menu, "Диагностика и восстановление…", self.on_diagnostics,
                         tip="Проверить файлы программы, папки и библиотеки в .venv;\n"
                             "восстановить удалённое, починить окружение, убрать лишние пакеты")
        self._add_action(service_menu, "Открыть лог запуска", self.on_open_bootstrap_log,
                         tip="Журнал подготовки окружения (main.py). При успешном запуске\n"
                             "удаляется автоматически — остаётся только в случае ошибок")
        self._add_action(service_menu, "Открыть папку логов",
                         lambda: open_in_file_manager(paths.logs_dir()),
                         tip="Журналы работы программы: лог установки и подробные логи запусков\n"
                             "(файл-лог включается в Настройки → «Журнал и диагностика»)")

        help_menu = ToolTipMenu("Справка", self)
        bar.addMenu(help_menu)
        help_menu.menuAction().setToolTip(
            "Справка: документация проекта, сообщение о проблеме, сведения о программе."
        )
        self._add_action(help_menu, "README (GitHub)", lambda: QDesktopServices.openUrl(QUrl(REPO_URL)),
                         tip="Открыть страницу проекта с документацией в браузере")
        self._add_action(help_menu, "Сообщить о проблеме",
                         lambda: QDesktopServices.openUrl(QUrl(f"{REPO_URL}/issues/new")),
                         tip="Открыть форму создания issue на GitHub (нужен аккаунт)")
        help_menu.addSeparator()
        self._add_action(help_menu, f"О программе ({APP_VERSION})", self.on_about,
                         tip="Версия, движки, интерпретатор Python и путь к папке проекта")

    def _add_action(self, menu: QMenu, text: str, slot, shortcut: str = "", tip: str = "") -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setToolTip(tip)
            action.setStatusTip(tip.splitlines()[0])   # показывается в статус-баре при наведении
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

        log.debug("Железо при старте: %s", self.hardware)
        self.models_panel.set_hardware(self.hardware)
        self.models_panel.rebuild()
        self._refresh_gpu_button()
        self._start_torch_probe()

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
        self._update_output_hint()
        QTimer.singleShot(400, self._check_integrity)

    def _restore_selected_model(self) -> None:
        self.rebuild_model_combo()
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
                self.model_combo.setCurrentIndex(0)

    # ================================================================== #
    #  Файлы
    # ================================================================== #
    def on_add_files(self) -> None:
        start_dir = self.settings.last_open_dir or str(Path.home())
        picked, _ = QFileDialog.getOpenFileNames(self, "Выбери аудио или видео", start_dir, MEDIA_FILTER)
        if not picked:
            log.debug("Добавление файлов: диалог закрыт без выбора")
            return
        log.debug("Добавление файлов: выбрано %d (%s)", len(picked), "; ".join(picked[:6]))
        self.settings.last_open_dir = str(Path(picked[0]).parent)
        self.settings.save()
        added = self.file_list.add_paths(picked)
        self._log(f"Добавлено файлов: {added} (всего {len(self.file_list.files())})")

    def on_add_folder(self) -> None:
        start_dir = self.settings.last_open_dir or str(Path.home())
        picked = QFileDialog.getExistingDirectory(self, "Папка с записями", start_dir)
        if not picked:
            return
        self.settings.last_open_dir = picked
        self.settings.save()
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
        word = paths.plural_ru(count, "файл", "файла", "файлов")
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

    def rebuild_model_combo(self) -> None:
        """Пересобирает выпадающий список СКАЧАННЫХ моделей (пункт 0 — «Не выбрано»)."""
        combo = getattr(self, "model_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("— Не выбрано —", "")
        for entry in self.catalog.sorted_for_display(self.catalog.installed()):
            combo.addItem(f"{entry.name} · {entry.size_text}", entry.id)
        current_id = self.selected_model.id if self.selected_model is not None else ""
        index = combo.findData(current_id)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _on_model_combo_changed(self, _index: int) -> None:
        model_id = self.model_combo.currentData() or ""
        if not model_id:
            # «Нет» — снимаем выбор, кнопка «РАБОТАТЬ» гаснет
            if self.selected_model is not None:
                self.selected_model = None
                self.models_panel.set_selected("")
                self.status_model.setText("модель не выбрана")
                self._update_work_button()
            return
        entry = self.catalog.get(model_id)
        if entry is not None and entry.installed:
            self._select_model(entry)

    def _select_model(self, entry: ModelEntry, save: bool = True) -> None:
        log.debug(
            "Выбор модели: id=%s name=%s engine=%s installed=%s dir=%s save=%s",
            entry.id, entry.name, entry.engine, entry.installed, entry.install_dir, save,
        )
        self.selected_model = entry
        self.models_panel.set_selected(entry.id)
        index = self.model_combo.findData(entry.id)
        if index >= 0 and self.model_combo.currentIndex() != index:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(index)
            self.model_combo.blockSignals(False)
        self.status_model.setText(f"модель: {entry.name}")
        self.status_model.setToolTip(
            f"{entry.engine_label} · {entry.size_text} · {entry.memory_text}\n{entry.description}"
        )
        if save:
            self.settings.selected_model_id = entry.id
            self.settings.last_used_model_id = entry.id
            self.settings.save()
            self._log(f"Выбрана модель: {entry.name}")
            self._toast(f"Модель для работы: {entry.name}")
        self._update_work_button()

    def on_download_model(self, entry: ModelEntry) -> None:
        log.debug("Запрошена установка модели: %s (source=%s)", entry.id, entry.source)
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
        # ВАЖНО: сигнал несёт (доля, текст); слоту статуса нужен именно текст,
        # иначе QLabel.setText(float) ронял приложение на каждом тике прогресса
        worker.progress.connect(lambda fraction, status: dialog.set_status(status))
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
        self._restore_selected_model()   # внутри пересобирается и комбо моделей

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
            return answer == compat.message_box_yes()
        return True

    def on_delete_model(self, entry: ModelEntry) -> None:
        if self._working:
            QMessageBox.information(
                self, "Идёт распознавание",
                "Нельзя удалять модель во время работы — сначала нажми «ОТМЕНИТЬ» "
                "и дождись остановки.",
            )
            return
        answer = QMessageBox.question(
            self, "Удалить модель",
            f"Удалить «{entry.name}» с диска?\n\nПапка: {entry.install_dir}\n"
            f"Освободится: {paths.human_size(entry.installed_size_bytes or entry.size_bytes)}",
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_no(),
        )
        if answer != compat.message_box_yes():
            return
        if self.selected_model is not None and self.selected_model.id == entry.id:
            registry.unload()
            self.selected_model = None
        from ..core.downloader import uninstall_model

        if uninstall_model(entry, root=self.catalog.root):
            log.debug("Модель удалена с диска: %s (%s)", entry.id, entry.install_dir)
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
        if self._working:
            QMessageBox.information(
                self, "Идёт распознавание",
                "Модель сейчас используется — выгрузить можно после завершения работы.",
            )
            return
        log.debug("Выгрузка модели из памяти по запросу пользователя")
        registry.unload()
        self._log("Модель выгружена из оперативной памяти")
        self._sync_runtime_actions()

    def _sync_runtime_actions(self) -> None:
        """Держит «состояние зависимые» пункты меню честными.

        «Выгрузить модель из памяти» имеет смысл только когда модель реально
        загружена в память (или идёт работа — тогда выгружать нельзя).
        """
        action = getattr(self, "unload_action", None)
        if action is None:
            return
        loaded = registry.current() is not None
        action.setEnabled(loaded and not self._working)

    def _focus_models(self) -> None:
        self.splitter.setSizes([420, 820])
        self.models_panel.search.setFocus()

    # ================================================================== #
    #  Работа
    # ================================================================== #
    def _update_work_button(self) -> None:
        files_ready = self.file_list.count() > 0
        model_ready = self.selected_model is not None and self.selected_model.installed
        if self._working:
            # на время работы «РАБОТАТЬ» скрыта, видна «ОТМЕНИТЬ»
            self.work_button.setEnabled(False)
            return
        self.work_button.setEnabled(files_ready and model_ready)
        if not files_ready:
            self.work_button.setText("Добавь файлы")
        elif not model_ready:
            self.work_button.setText("Выбери модель")
        else:
            count = self.file_list.count()
            word = paths.plural_ru(count, "файл", "файла", "файлов")
            self.work_button.setText(f"РАБОТАТЬ · {count} {word}")

    def _set_working(self, working: bool) -> None:
        """Переключает интерфейс между состояниями «ожидание» и «работа».

        «РАБОТАТЬ» на время работы ЗАМЕНЯЕТСЯ кнопкой «ОТМЕНИТЬ» (и наоборот):
        одна и та же позиция, интуитивное поведение.
        """
        self._working = working
        self.work_button.setVisible(not working)
        self.cancel_button.setVisible(working)
        if working:
            self.cancel_button.setEnabled(True)
            self.cancel_button.setText("ОТМЕНИТЬ")
        self._update_work_button()
        self._sync_runtime_actions()

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
                "Скачай и выбери модель в правой панели — без неё распознавать нечем.\n"
                "В карточках моделей указаны размер, потребление памяти, языки,\n"
                "оценки точности и скорости — выбери под своё железо.",
            )
            self._focus_models()
            return

        from ..asr.base import engine_install_size
        from ..asr.registry import engine_available, missing_engine_packages

        if not engine_available(entry.engine):
            packages = missing_engine_packages(entry.engine)
            size_text = engine_install_size(entry.engine)
            answer = QMessageBox.question(
                self, "Нужны дополнительные компоненты",
                f"Модель «{entry.name}» работает на движке «{entry.engine_label}».\n"
                "Его библиотеки установятся в папку программы (.venv) — системный\n"
                "Python не затрагивается, это разовая операция.\n\n"
                + (f"Объём скачивания: {size_text}\n" if size_text else "")
                + f"Пакеты: {', '.join(packages)}\n\nУстановить сейчас?",
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
        self._last_job_engine = entry.engine
        self.settings.save()

        options = JobOptions.from_settings(entry, self.settings, output_dir)
        if self.timestamps_check.isChecked() and options.formats == ["txt"]:
            options.formats = ["txt-time"]
        log.debug(
            "on_work: файлы=%s | модель=%s (engine=%s, dir=%s) | output=%s | язык=%s | "
            "beam=%s vad=%s clean=%s paragraphs=%s timestamps=%s words=%s batched=%s batch=%s | "
            "device=%s compute=%s threads=%s gpu_index=%s parallel=%s overwrite=%s",
            [str(f) for f in files], entry.id, entry.engine, entry.install_dir,
            options.output_dir, options.language or "авто",
            options.beam_size, options.use_vad, options.suppress_hallucinations,
            options.paragraphs, self.settings.add_timestamps_to_txt,
            options.word_timestamps, options.batched, options.batch_size,
            options.device, options.compute_type, options.cpu_threads, options.gpu_index,
            self.settings.parallel_files, options.overwrite,
        )

        self.results = {}
        self.job_started_at = time.time()
        self.preview.clear()
        self._job_total_files = max(1, len(files))
        self._job_done_files = 0
        self.overall_progress.setRange(0, 1000)
        self.overall_progress.setValue(0)
        self.overall_label.setText(f"Обрабатываю {len(files)} файл(ов) моделью «{entry.name}»")
        self.file_progress.setRange(0, 1000)
        self.file_progress.setValue(0)
        self._set_working(True)   # ДО start(): без гонки «кнопка не погасла»
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
        self._job_total_files = max(1, total)
        self._job_done_files = max(0, min(done, total))
        self.overall_progress.setRange(0, 1000)
        self.overall_progress.setValue(int(self._job_done_files / self._job_total_files * 1000))
        if total > 1:
            self.overall_label.setText(f"Файл {min(done + 1, total)} из {total}")

    def _on_file_progress(self, index: int, fraction: float, stage: str) -> None:
        if fraction < 0:
            self.file_progress.setRange(0, 0)
        else:
            fraction = max(0.0, min(1.0, fraction))
            self.file_progress.setRange(0, 1000)
            self.file_progress.setValue(int(fraction * 1000))
            # общий бар: доля готовых файлов + прогресс текущего — чтобы на
            # ОДНОМ файле проценты шли плавно, а не стояли на нуле всю лекцию
            overall = (self._job_done_files + fraction) / max(1, self._job_total_files)
            self.overall_progress.setValue(max(0, min(1000, int(overall * 1000))))
        try:
            name = Path(self.file_list.files()[index]).name
        except IndexError:
            name = f"файл {index + 1}"
        self.file_label.setText(f"{name} · {stage}")

    def _on_segment_ready(self, index: int, text: str, start: float, end: float) -> None:
        if not text:
            return
        start = max(0.0, float(start))
        if start >= 3600:
            stamp = f"{int(start // 3600)}:{int(start % 3600 // 60):02d}:{int(start % 60):02d}"
        else:
            stamp = f"{int(start // 60):02d}:{int(start % 60):02d}"
        self.preview.appendPlainText(f"[{stamp}] {text.strip()}")
        scrollbar = self.preview.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_file_finished(self, index: int, result: FileResult) -> None:
        self.results[index] = result
        self.results_button.setEnabled(bool(self.results))

    def _on_job_finished(self, ok: int, fail: int, seconds: float) -> None:
        log.debug("Задача завершена: ok=%d fail=%d seconds=%.1f", ok, fail, seconds)
        cancelled = self.worker is not None and self.worker.cancelled
        engine_missing = bool(getattr(self.worker, "engine_missing", False))
        total = self._job_total_files
        produced = [path for result in self.results.values() for path in result.outputs]
        words = sum(result.words for result in self.results.values())
        audio = sum(result.audio_duration for result in self.results.values())
        speed = (audio / seconds) if seconds > 0 else 0.0

        if engine_missing:
            self.overall_label.setText(
                "Распознавание отложено: движку не хватает компонентов — "
                "установи их в появившемся окне, задача продолжится сама."
            )
            self.file_label.setText("")
            self._log(self.overall_label.text())
            self._finish_worker()
            return

        self.overall_label.setText(
            ("Отменено" if cancelled else "Готово") + f": {ok} из {total}"
            + (f" · {fail} с ошибкой" if fail else "")
            + f" · {words} слов · {speed:.1f}× реального времени · {seconds:.0f} с"
        )
        self.file_label.setText(", ".join(path.name for path in produced[:3]) + (" …" if len(produced) > 3 else ""))
        self._log(self.overall_label.text())

        self._finish_worker()

        if fail and not cancelled:
            failed = [result for result in self.results.values() if not result.ok]
            details = "\n".join(f"• {Path(result.source).name}: {result.error}" for result in failed[:12])
            QMessageBox.warning(
                self, "Часть файлов не обработана",
                f"{fail} из {total} файлов завершились ошибкой:\n\n{details}",
            )
        elif ok:
            self.results_button.setEnabled(True)
            if self.settings.open_output_folder_when_done and produced and not cancelled:
                open_in_file_manager(produced[0])
            self._toast(f"Распознано файлов: {ok}")
            self._maybe_offer_gpu_boost()

    def _finish_worker(self) -> None:
        """Возвращает интерфейс в ожидание и убирает завершённый воркер."""
        self._set_working(False)
        self.file_progress.setRange(0, 1000)
        self.file_progress.setValue(0)
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()

    def _on_job_failed(self, message: str) -> None:
        self._finish_worker()
        self.overall_label.setText("Ошибка")
        self.file_label.setText(message.splitlines()[0] if message else "")
        self._log("ОШИБКА: " + message.replace("\n", " | "))
        QMessageBox.critical(self, "Ошибка при распознавании", message)

    def _on_engine_missing(self, engine: str, packages) -> None:
        self._finish_worker()
        from ..asr.base import engine_install_size

        size_text = engine_install_size(engine)
        answer = QMessageBox.question(
            self, "Нужны дополнительные компоненты",
            f"Для этой модели нужен движок «{engine}».\n"
            + (f"Объём скачивания: {size_text}\n" if size_text else "")
            + f"Установить {', '.join(packages)} в .venv проекта?",
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_yes(),
        )
        if answer == compat.message_box_yes():
            self._pending_job = True
            self.install_packages_interactive(list(packages or []), f"Движок {engine}")

    def on_cancel(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            self._finish_worker()
            return
        self.worker.cancel()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Останавливаю…")
        self.file_label.setText("Останавливаю… (дожидаемся текущего сегмента)")
        self._log("Запрошена отмена")

    # ================================================================== #
    #  GPU-ускорение torch-моделей (кнопка-статус + разовое предложение)
    # ================================================================== #
    def _start_torch_probe(self) -> None:
        """Асинхронно узнаёт, стоит ли torch и видит ли он CUDA (subprocess)."""
        if not self.hardware.get("has_nvidia"):
            return    # нет карты NVIDIA — зонд не нужен, кнопка уже серая
        if self.torch_probe is not None and self.torch_probe.isRunning():
            return
        probe = TorchCudaProbeWorker(self)
        probe.probed.connect(self._on_torch_probed)
        self.torch_probe = probe
        probe.start()

    def _on_torch_probed(self, installed: bool, cuda: bool, detail: str) -> None:
        log.debug("Зонд PyTorch: installed=%s cuda=%s detail=%r", installed, cuda, detail)
        self.gpu_state = {"probed": True, "installed": installed, "cuda": cuda, "detail": detail}
        probe, self.torch_probe = self.torch_probe, None
        if probe is not None:
            probe.deleteLater()
        self._refresh_gpu_button()
        if installed:
            log.info(
                "PyTorch в .venv: %s, CUDA %s (%s)",
                "установлен" if installed else "нет",
                "доступна" if cuda else "НЕ доступна",
                detail or "—",
            )

    def _refresh_gpu_button(self) -> None:
        """Текст/доступность кнопки GPU = честное текущее состояние ускорения."""
        button = getattr(self, "gpu_button", None)
        if button is None:
            return
        hardware = self.hardware or {}
        gpu_name = str(hardware.get("gpu_name") or "NVIDIA")
        vram = int(hardware.get("vram_mb") or 0)
        state = self.gpu_state

        if not hardware:
            button.setText("Определяю железо…")
            button.setEnabled(False)
            return

        if not hardware.get("has_nvidia"):
            button.setText("GPU-ускорение недоступно: нет видеокарты NVIDIA")
            button.setToolTip(
                "Все движки будут работать на процессоре.\n"
                "Для максимально быстрого CPU-распознавания выбирай модели с int8/q5_0\n"
                "(faster-whisper / whisper.cpp) — их характеристики видны в каталоге справа."
            )
            button.setEnabled(False)
            return

        if not state.get("probed"):
            button.setText(f"Проверяю PyTorch для {gpu_name}…")
            button.setEnabled(False)
            return

        if state.get("cuda"):
            button.setText(f"✓ GPU используется: PyTorch CUDA ({state.get('detail') or gpu_name})")
            button.setToolTip(
                f"{gpu_name}" + (f" · {vram / 1024:.1f} ГБ VRAM" if vram else "") + "\n"
                "CUDA-сборка PyTorch на месте: GigaAM / Qwen3-ASR / HF-модели\n"
                "считают энкодер на видеокарте, а декодирование — параллельно на CPU.\n"
                "Ставить больше нечего — кнопка неактивна.\n\n"
                "(Для Whisper-моделей faster-whisper CUDA настраивается отдельно:\n"
                "Сервис → Настройки → вкладка «Движки».)"
            )
            button.setEnabled(False)
            return

        # NVIDIA есть, а torch CPU-сборка (или ещё не установлен) — можно ускорить
        if state.get("installed"):
            text = f"⚡ Ускорить на GPU: PyTorch сейчас без CUDA ({state.get('detail') or 'cpu'})"
        else:
            text = "⚡ Ускорить на GPU: поставить PyTorch с CUDA (~2.5–3 ГБ)"
        button.setText(text)
        button.setToolTip(
            f"Видеокарта {gpu_name}" + (f" ({vram / 1024:.1f} ГБ VRAM) " if vram else " ") +
            "простаивает: в .venv стоит CPU-сборка PyTorch.\n"
            "Клик установит CUDA-сборку (torch+torchaudio, разово ~2.5–3 ГБ) —\n"
            "torch-модели (GigaAM, Qwen3-ASR, любые HF) ускорятся ориентировочно в 1.5–2 раза:\n"
            "энкодер перейдёт на видеокарту, декодирование останется на процессоре.\n\n"
            "Установка идёт строго в .venv проекта; нужен интернет."
        )
        button.setEnabled(True)

    def _install_torch_cuda(self) -> None:
        """Ставит CUDA-сборку torch в .venv (с удалением CPU-сборки)."""
        if self.package_worker is not None and self.package_worker.isRunning():
            QMessageBox.information(self, "Уже идёт установка", "Дождись завершения предыдущей установки.")
            return
        hardware = self.hardware or {}
        gpu_name = str(hardware.get("gpu_name") or "NVIDIA")
        vram = int(hardware.get("vram_mb") or 0)
        driver = str(hardware.get("gpu_driver") or "")

        driver_note = ""
        try:
            major = int(driver.split(".")[0])
            if 0 < major < TORCH_CUDA_MIN_DRIVER_MAJOR:
                driver_note = (
                    f"\n\n⚠ Драйвер NVIDIA {driver} старее {TORCH_CUDA_MIN_DRIVER_MAJOR} — "
                    "CUDA-сборка может не заработать. Сначала обнови драйвер (nvidia.com)."
                )
        except (ValueError, IndexError):
            pass

        answer = QMessageBox.question(
            self, "Ускорение на видеокарте",
            f"Видеокарта: {gpu_name}" + (f" · {vram / 1024:.1f} ГБ VRAM" if vram else "") +
            (f" · драйвер {driver}" if driver else "") + "\n\n"
            "Сейчас torch-модели (GigaAM, Qwen3-ASR, HF) считаются на процессоре.\n"
            "Установить CUDA-сборку PyTorch в .venv проекта?\n\n"
            "• скачается ~2.5–3 ГБ (разово, при следующих запусках — ничего);\n"
            "• энкодер перейдёт на GPU: ориентировочно в 1.5–2 раза быстрее;\n"
            "• системный Python не затрагивается." + driver_note + "\n\n"
            "Установить сейчас?",
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_yes(),
        )
        if answer != compat.message_box_yes():
            return

        self._last_install_torch_cuda = True
        self.install_packages_interactive(
            ["--index-url", TORCH_CUDA_INDEX_URL, "torch>=2.6", "torchaudio>=2.6"],
            "PyTorch CUDA (GPU-ускорение)",
            pre_uninstall=["torch", "torchaudio"],
        )

    def _maybe_offer_gpu_boost(self) -> None:
        """Разовое предложение «сейчас / позже / не предлагать» после CPU-прогона
        torch-модели на машине с видеокартой NVIDIA."""
        try:
            if self._last_job_engine not in TORCH_ENGINE_NAMES:
                return   # задача шла не на torch-движке — предложение не к месту
            if self.settings.gpu_boost_offered:
                return
            if not self.hardware.get("has_nvidia"):
                return
            state = self.gpu_state
            if not state.get("probed") or state.get("cuda"):
                return
            # Linux: pip-torch по умолчанию уже идёт с CUDA; CPU-сборка там —
            # редкость, и предложение только запутает. macOS: CUDA не существует.
            if os.name != "nt":
                return

            self.settings.gpu_boost_offered = True
            self.settings.save()

            box = QMessageBox(self)
            box.setWindowTitle("Доступно GPU-ускорение")
            box.setIcon(QMessageBox.Icon.Information)
            gpu_name = str(self.hardware.get("gpu_name") or "NVIDIA")
            box.setText(
                f"Задача выполнена на процессоре, хотя у тебя есть видеокарта {gpu_name}:\n"
                "PyTorch в .venv установлен без CUDA.\n\n"
                "Установка CUDA-сборки (~2.5–3 ГБ, разово) ускорит GigaAM /\n"
                "Qwen3-ASR / HF-модели ориентировочно в 1.5–2 раза.\n\n"
                "Поставить сейчас? (потом можно в любой момент — кнопкой\n"
                "«⚡ Ускорить на GPU» в блоке «3 · Работа»)"
            )
            now = box.addButton("Поставить сейчас", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Позже", QMessageBox.ButtonRole.RejectRole)
            box.addButton("Не предлагать", QMessageBox.ButtonRole.DestructiveRole)
            box.exec()
            if box.clickedButton() is now:
                self._install_torch_cuda()
        except Exception as exc:  # noqa: BLE001 - предложение не должно ронять финал задачи
            log.debug("Не удалось предложить GPU-ускорение: %s", exc)

    # ================================================================== #
    #  Установка пакетов (движки, CUDA, ремонт окружения)
    # ================================================================== #
    @staticmethod
    def _with_cuda_index(packages: List[str], hardware: Optional[dict]) -> List[str]:
        """Добавляет CUDA-индекс PyTorch, если ставится torch на Windows+NVIDIA.

        На Linux pip и так ставит torch с CUDA, на macOS CUDA нет — поэтому
        только Windows. Явный --index-url в списке не перебиваем.

        ВАЖНО: --index-url ПОЛНОСТЬЮ заменяет PyPI. Для списка только из
        torch-пакетов это то, что нужно (берём CUDA-сборку). Но движки
        ставятся смешанным списком (torch + transformers + hydra-core + …),
        и на индексе PyTorch остальных пакетов просто нет — установка падала
        с «No matching distribution». Поэтому для смешанных списков индекс
        PyTorch добавляется как ДОПОЛНИТЕЛЬНЫЙ (--extra-index-url): pip/uv
        сами возьмут torch/torchaudio в CUDA-сборке (версия с суффиксом
        +cu126 старше чистой по PEP 440), а остальное — с PyPI.
        """
        packages = list(packages)
        if not (os.name == "nt" and (hardware or {}).get("has_nvidia")):
            return packages
        if "--index-url" in packages or "--extra-index-url" in packages:
            return packages
        if not any(str(pkg).lower().startswith("torch") for pkg in packages):
            return packages
        has_non_torch = any(
            not str(pkg).lower().startswith("torch") for pkg in packages
        )
        flag = "--extra-index-url" if has_non_torch else "--index-url"
        return [flag, TORCH_CUDA_INDEX_URL, *packages]

    def install_packages_interactive(
        self, packages: List[str], title: str = "", pre_uninstall: Optional[List[str]] = None,
    ) -> None:
        if not packages:
            return
        if self.package_worker is not None and self.package_worker.isRunning():
            QMessageBox.information(self, "Уже идёт установка", "Дождись завершения предыдущей установки.")
            return

        # Windows: pip по умолчанию ставит CPU-сборку torch. Если есть карта
        # NVIDIA и в списке есть torch — сразу берём CUDA-индекс PyTorch,
        # чтобы движки «из коробки» использовали видеокарту.
        before = list(packages)
        packages = self._with_cuda_index(packages, self.hardware)
        if packages != before:
            self._log("Есть GPU NVIDIA — torch ставится сразу в CUDA-сборке")

        size_note = ""
        if any(pkg.lower().startswith("torch") for pkg in packages):
            size_note = (
                "\nОбъём большой из-за PyTorch: ~120–200 МБ (CPU-сборка) — это разовая "
                "загрузка, при следующих запусках ничего не качается."
            )
        dialog = ProgressDialog(
            title or "Установка компонентов",
            "Устанавливаем строго в .venv проекта — системный Python не трогаем.\n"
            "Пакеты: " + ", ".join(packages) + size_note,
            parent=self,
        )
        worker = PackageInstaller(packages, title, self, pre_uninstall=pre_uninstall)
        worker.outputLine.connect(dialog.append_log)
        worker.progress.connect(lambda fraction, status: dialog.set_fraction(fraction))
        worker.progress.connect(lambda fraction, status: dialog.set_status(status))
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

        was_torch_cuda = self._last_install_torch_cuda
        self._last_install_torch_cuda = False

        if not ok:
            QMessageBox.warning(self, "Установка не завершилась", message + self._install_hint(message))
        elif was_torch_cuda:
            # перепроверяем состояние torch (зондом в subprocess) и подсказываем
            # про перезапуск: если torch уже был импортирован в этой сессии,
            # процесс продолжит видеть старую (CPU) сборку до перезапуска
            self.gpu_state = {"probed": False, "installed": False, "cuda": False, "detail": ""}
            self._refresh_gpu_button()
            self._start_torch_probe()
            restart_note = ""
            if "torch" in sys.modules:
                restart_note = (
                    "\n\nВажно: в этой сессии torch уже загружен — перезапусти программу,\n"
                    "чтобы новая CUDA-сборка подхватилась."
                )
            QMessageBox.information(
                self, "PyTorch CUDA установлен",
                "CUDA-сборка PyTorch установлена в .venv.\n"
                "Состояние кнопки «⚡ Ускорить на GPU» обновится после проверки." + restart_note,
            )
        elif getattr(self, "_pending_job", False):
            self._pending_job = False
            QTimer.singleShot(250, self.on_work)
        dialog.accept()
        self.package_worker = None
        self.package_dialog = None

    @staticmethod
    def _install_hint(message: str) -> str:
        """Человеческая подсказка к типичным причинам падения pip."""
        low = (message or "").lower()
        if any(marker in low for marker in (
            "failed to build", "build wheel", "cmake", "winerror 2",
            "microsoft visual c++", "no compiler",
        )):
            return (
                "\n\nПохоже, pip не нашёл готовое колесо одного из пакетов под твой Python "
                "и попытался собрать его из исходников (для этого нужны компилятор и CMake, "
                "которых обычно нет).\n"
                "Что делать:\n"
                "  1) обнови LocalASR до последней версии — пины зависимостей ослаблены, "
                "ставятся готовые колёса;\n"
                "  2) если не помогло — поставь Python 3.11–3.13 и запусти программу им "
                "(под экзотически новый Python колёса выходят не сразу)."
            )
        if "could not find a version" in low or "no matching distribution" in low:
            return (
                "\n\nОдин из пакетов не находится в PyPI под твою версию Python. "
                "Попробуй Python 3.11–3.13 или проверь, что есть доступ в интернет без прокси."
            )
        return ""

    def on_diagnostics(self) -> None:
        """Сервис → Диагностика и восстановление (файлы, папки, .venv)."""
        DiagnosticsDialog(self).exec()
        # после ремонта окружения железо/движки могли измениться
        registry.hardware(refresh=True)
        self.hardware = registry.hardware()
        self.models_panel.set_hardware(self.hardware)
        self._refresh_gpu_button()

    def _check_integrity(self) -> None:
        """Тихая проверка файлов программы при старте.

        Ничего не ломает: если пользователь удалил файлы проекта — сообщаем
        и предлагаем восстановить их с GitHub (одной кнопкой).
        """
        try:
            from ..core import integrity

            missing = integrity.missing_project_files()
        except Exception as exc:  # noqa: BLE001
            log.debug("Проверка целостности не выполнена: %s", exc)
            return
        if not missing:
            return
        log.warning("Отсутствуют файлы программы: %s", ", ".join(missing[:10]))
        listing = "\n".join("  • " + name for name in missing[:12])
        if len(missing) > 12:
            listing += f"\n  … и ещё {len(missing) - 12}"
        answer = QMessageBox.question(
            self, "Отсутствуют файлы программы",
            f"В папке программы не хватает {len(missing)} файл(ов):\n{listing}\n\n"
            "Программа продолжит работу, но может быть неполноценной.\n"
            "Скачать недостающие файлы заново из репозитория проекта?\n"
            "(нужен интернет; вес — считаные килобайты, данные не затрагиваются)",
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_yes(),
        )
        if answer == compat.message_box_yes():
            self._restore_files_interactive(missing)

    def _restore_files_interactive(self, missing: List[str]) -> None:
        if self.restore_worker is not None and self.restore_worker.isRunning():
            return
        dialog = ProgressDialog(
            "Восстановление файлов программы",
            f"Скачиваем {len(missing)} файл(ов) из репозитория проекта…",
            parent=self,
        )
        worker = FileRestoreWorker(missing, self)
        worker.progress.connect(dialog.set_fraction)
        worker.progress.connect(lambda fraction, status: dialog.set_status(status))
        worker.outputLine.connect(dialog.append_log)
        dialog.cancelRequested.connect(worker.cancel)
        worker.finishedWith.connect(lambda ok, message: self._on_restore_finished(ok, message, dialog))
        self.restore_worker = worker
        worker.start()
        dialog.exec()

    def _on_restore_finished(self, ok: bool, message: str, dialog: ProgressDialog) -> None:
        dialog.finish("Готово" if ok else "Остановлено")
        dialog.accept()
        worker, self.restore_worker = self.restore_worker, None
        if worker is not None:
            worker.deleteLater()
        self._log(message)
        if ok:
            self._toast("Файлы программы восстановлены")
        else:
            QMessageBox.warning(self, "Восстановление", message)

    def on_open_bootstrap_log(self) -> None:
        log_file = paths.logs_dir() / "bootstrap.log"
        error_log = paths.ROOT_DIR / "logs" / "bootstrap_error.log"
        target = log_file if log_file.is_file() else error_log
        if not target.is_file():
            QMessageBox.information(
                self, "Логи",
                "Лога запуска нет — и это хорошая новость.\n\n"
                "bootstrap.log ведётся только во время подготовки окружения и\n"
                "автоматически удаляется при успешном запуске программы.\n"
                "Он остаётся на диске, только если запуск завершился ошибкой.",
            )
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
        dialog = SettingsDialog(
            self.settings, self.hardware, self,
            on_reset_defaults=self._reset_settings_to_defaults,
        )
        if dialog.exec() == compat.dialog_accepted():
            self._apply_settings_to_ui()
            log.debug("Настройки применены из диалога: %s", self.settings.to_dict())
            self._log("Настройки сохранены")

    def _reset_settings_to_defaults(self) -> None:
        """Вызывается из диалога настроек после сброса: применяем дефолты к UI.

        К этому моменту `self.settings` уже сброшен в стандартные значения и
        сохранён на диск. Осталось привести интерфейс в соответствие и вернуть
        окну/фильтрам исходный вид.
        """
        self._apply_settings_to_ui()
        # геометрия окна — к стандартной
        self.settings.window_geometry = ""
        self.settings.window_state = ""
        self.settings.splitter_sizes = []
        try:
            self.splitter.setSizes([720, 520])
            self.log_dock.setVisible(bool(self.settings.show_log_panel))
            self.showNormal()
            self.resize(1280, 800)
        except Exception as exc:  # noqa: BLE001
            log.debug("Не удалось сбросить геометрию: %s", exc)
        self.settings.save()
        # фильтры каталога — к стандартным
        try:
            self.models_panel.reset_filters_to_defaults()
        except Exception as exc:  # noqa: BLE001
            log.debug("Не удалось сбросить фильтры: %s", exc)
        # файловый лог — к стандартному состоянию (по умолчанию выключен)
        try:
            from .. import logutils

            logutils.set_file_logging(bool(self.settings.file_logging))
        except Exception as exc:  # noqa: BLE001
            log.debug("Не удалось применить состояние логирования: %s", exc)
        self._update_output_hint()
        self._log("Настройки сброшены к стандартным")
        self._toast("Настройки сброшены к стандартным")

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

    def on_about(self) -> None:
        QMessageBox.about(
            self, f"О {APP_NAME}",
            f"<h3>{APP_NAME} {APP_VERSION}</h3>"
            "<p>Локальное распознавание речи: аудио и видео → текст, субтитры, JSON.</p>"
            "<p>Движки: faster-whisper (CTranslate2), whisper.cpp (GGML), Vosk (Kaldi), "
            "Qwen3-ASR и GigaAM (torch), а также универсальный hf-transformers "
            "для любых ASR-моделей Hugging Face.<br>"
            "GUI: PySide6. Всё устанавливается в <code>.venv</code> внутри папки проекта.</p>"
            f"<p>Python: {sys.version.split()[0]}<br>Интерпретатор: {sys.executable}<br>"
            f"Проект: {paths.ROOT_DIR}</p>",
        )

    # ---------------- служебное ---------------------------------------- #
    def _collect_settings(self) -> None:
        self.settings.language = self.language_combo.currentData() or ""
        # результат всегда в txt: отдельной настройки формата больше нет
        self.settings.output_format = "txt"
        # ВАЖНО: значение по умолчанию («outputs/ внутри проекта») НЕ
        # сохраняется абсолютным путём — иначе после переноса папки проекта
        # результаты продолжали бы уезжать по СТАРОМУ пути (баг «создал
        # LocalASR/outputs на рабочем столе»). Пустая строка = «по умолчанию».
        output_text = self.output_edit.text().strip()
        default_text = str(paths.outputs_dir())
        if not output_text or os.path.normcase(output_text) == os.path.normcase(default_text):
            self.settings.output_dir = ""
        else:
            self.settings.output_dir = output_text
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
        self.output_edit.setText(self.settings.output_dir or str(paths.outputs_dir()))
        self.vad_check.setChecked(self.settings.use_vad)
        self.clean_check.setChecked(self.settings.suppress_hallucinations)
        self.paragraph_check.setChecked(self.settings.paragraphs)
        self.timestamps_check.setChecked(self.settings.add_timestamps_to_txt)
        self.wordts_check.setChecked(self.settings.word_timestamps)
        self.translate_check.setChecked(self.settings.translate_to_english)
        beam = int(self.settings.beam_size or 5)
        index = self.beam_combo.findData(beam)
        if index < 0:
            # значение из «Все настройки…» (например 7) не теряем — добавляем пункт
            self.beam_combo.addItem(f"beam {beam}", beam)
            index = self.beam_combo.count() - 1
        self.beam_combo.setCurrentIndex(index)
        self.batched_check.setChecked(bool(self.settings.batched))
        self.hotwords_edit.setText(self.settings.hotwords)
        self.prompt_edit.setText(self.settings.initial_prompt)
        self.open_folder_check.setChecked(bool(self.settings.open_output_folder_when_done))
        # состояние панели «Расширенные настройки»
        self.advanced_toggle.blockSignals(True)
        self.advanced_toggle.setChecked(bool(self.settings.advanced_open))
        self.advanced_toggle.blockSignals(False)
        self.advanced_frame.setVisible(bool(self.settings.advanced_open))
        self.advanced_toggle.setText(
            "▾ Расширенные настройки" if self.settings.advanced_open else "▸ Расширенные настройки"
        )
        self._update_output_hint()

    def _on_settings_changed(self, *_args) -> None:
        self._collect_settings()
        self._update_output_hint()

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_frame.setVisible(checked)
        self.advanced_toggle.setText("▾ Расширенные настройки" if checked else "▸ Расширенные настройки")
        self.settings.advanced_open = bool(checked)
        self.settings.save()

    def _update_output_hint(self) -> None:
        output_dir = self.current_output_dir()
        fmt = "txt-time" if self.timestamps_check.isChecked() else "txt"
        sample = paths.sanitize_stem("лекция 1.mp4")
        suffix = paths.output_suffix(fmt)
        sample_name = f"{sample} {paths.datetime_stamp()}{suffix}"
        note = (
            "(имя оригинала + дата и время — результаты разных запусков\n"
            "никогда не перетираются)"
        )
        try:
            outside = paths.ROOT_DIR.resolve() not in output_dir.resolve().parents
        except OSError:
            outside = False
        if outside:
            note = "⚠ папка ВНЕ проекта — проверь, что выбрал её намеренно.\n" + note
        self.output_hint.setText(
            f"Результат: {paths.short_path(output_dir, 46)} / «{sample_name}» {note}"
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
        log.debug("Drag&drop: %d объект(ов): %s", len(urls), "; ".join(urls[:6]))
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

    def _save_geometry_now(self) -> None:
        """Пишет текущую геометрию/состояние окна в настройки (атомарно)."""
        try:
            self.settings.window_geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
            self.settings.window_state = bytes(self.saveState().toBase64()).decode("ascii")
            self.settings.splitter_sizes = [int(v) for v in self.splitter.sizes()]
            self.settings.show_log_panel = self.log_dock.isVisible()
            self.settings.save()
        except Exception as exc:  # noqa: BLE001
            log.debug("Не удалось сохранить геометрию: %s", exc)

    def _schedule_geometry_save(self) -> None:
        """Дебаунс: окно можно двигать непрерывно — сохраним один раз через паузу.

        Благодаря этому любое изменение (размер, позиция, свернутое/развёрнутое
        состояние, панели) переживает даже аварийное завершение программы.
        """
        try:
            self._geo_timer.start()
        except RuntimeError:
            pass    # окно уже удалено — сохранять некуда

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._schedule_geometry_save()

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        self._schedule_geometry_save()

    def changeEvent(self, event) -> None:  # noqa: N802
        # смена состояния окна (свернуто/развёрнуто/обычное)
        super().changeEvent(event)
        try:
            from PySide6.QtCore import QEvent

            if event.type() == QEvent.Type.WindowStateChange:
                self._schedule_geometry_save()
        except Exception:  # noqa: BLE001
            pass

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

        # Все фоновые потоки останавливаются ГАРАНТИРОВАННО (cancel → wait →
        # terminate). Иначе Qt на выходе пишет «QThread: Destroyed while thread
        # is still running», а сетевые ретраи/pip могут пережить окно.
        from .workers import stop_worker

        log.debug("closeEvent: останавливаю фоновые потоки")
        stop_worker(self.worker, 6000, "TranscribeWorker")
        for name, worker in (
            ("download", self.download_worker),
            ("package", self.package_worker),
            ("restore", self.restore_worker),
            ("catalog", self.catalog_worker),
            ("torch_probe", self.torch_probe),
        ):
            stop_worker(worker, 4000, name)

        self._collect_settings()
        self._save_geometry_now()

        detach_qt_sink(self.log_handler)
        registry.unload()
        log.debug("closeEvent: завершение корректно")
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
