"""LocalASR main window."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QByteArray, QUrl, QTimer, Signal
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
from ..i18n import _, tn
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
    UpdateApplyWorker,
    UpdateCheckWorker,
)

log = get_logger("ui")


def media_filter() -> str:
    return _("Media files ({});;All files (*)").format(
        " ".join(f"*{ext}" for ext in sorted(MEDIA_EXTENSIONS))
    )


# --------------------------------------------------------------------------- #
#  Icon (embedded SVG — no external files)
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
    """Icon from the embedded SVG; rendered at several sizes so the window
    title and the Windows taskbar stay crisp."""
    icon = QIcon()
    data = QByteArray(_ICON_SVG.encode("utf-8"))
    try:  # vector render per size — maximum sharpness
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
        log.debug("QSvgRenderer unavailable, drawing raster: %s", exc)
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
        log.debug("Failed to build the icon from SVG: %s", exc)
    return icon


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #


class MainWindow(QMainWindow):
    #: log records from background threads reach the GUI thread via a queued signal
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
        self.update_worker: Optional[UpdateCheckWorker] = None
        self.update_apply_worker: Optional[UpdateApplyWorker] = None
        self.update_dialog: Optional[ProgressDialog] = None
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
        # PyTorch state in the environment: probed/installed/cuda/detail
        self.gpu_state: dict = {"probed": False, "installed": False, "cuda": False, "detail": ""}

        self.setWindowTitle(_("{app} — local speech recognition").format(app=APP_NAME))
        self.setWindowIcon(app_icon())
        # Compact minimum: in a narrow window everything wraps and the rest scrolls.
        self.setMinimumSize(940, 620)
        self.setAcceptDrops(True)

        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        self._restore_geometry()

        # IMPORTANT: log records are born in background threads, and Qt widgets
        # may only be touched from the GUI thread — hence the signal bridge
        # (delivered to the GUI thread through a queue automatically).
        self.logRecord.connect(self._on_log_record)
        self.log_handler = attach_qt_sink(self.logRecord.emit)
        log.info(_("GUI started (%s %s, Python %s)"), APP_NAME, APP_VERSION, sys.version.split()[0])

        # A slow timer keeps runtime-dependent buttons/menu items honest
        # (e.g. "Unload model" is active only while a model is in memory).
        self._state_timer = QTimer(self)
        self._state_timer.setInterval(2000)
        self._state_timer.timeout.connect(self._sync_runtime_actions)
        self._state_timer.start()
        self._sync_runtime_actions()

        # Debounced window geometry saving: moves/resizes are persisted after
        # a pause — settings survive even a crash.
        self._geo_timer = QTimer(self)
        self._geo_timer.setSingleShot(True)
        self._geo_timer.setInterval(800)
        self._geo_timer.timeout.connect(self._save_geometry_now)

        QTimer.singleShot(60, self._startup_checks)

    # ================================================================== #
    #  UI construction
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

    # ---------------- left column -------------------------------------- #
    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(compat.frame_no_frame())
        # in a very narrow window horizontal scrolling beats clipped buttons
        scroll.setHorizontalScrollBarPolicy(compat.SCROLLBAR_AS_NEEDED)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 12, 16)
        layout.setSpacing(14)

        # --- header
        header = QHBoxLayout()
        title = make_label(APP_NAME, "H1")
        header.addWidget(title)
        header.addWidget(make_label(_("· local speech recognition without the cloud"), "Muted"))
        header.addStretch(1)
        version = make_label(f"v{APP_VERSION}", "Muted")
        header.addWidget(version)
        layout.addLayout(header)

        # --- 1. FILES
        files_box = QFrame()
        files_box.setObjectName("SidePanel")
        files_layout = QVBoxLayout(files_box)
        files_layout.setContentsMargins(14, 14, 14, 14)
        files_layout.setSpacing(10)

        files_header = QHBoxLayout()
        files_header.addWidget(make_label(_("1 · Files"), "H2"))
        files_header.addStretch(1)
        self.files_count_label = make_label(_("0 files"), "Muted")
        files_header.addWidget(self.files_count_label)
        files_layout.addLayout(files_header)

        buttons = FlowLayout(spacing=8)

        self.add_files_button = QPushButton(_("📂 Choose files…"))
        self.add_files_button.setObjectName("Primary")
        self.add_files_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.add_files_button.setShortcut(QKeySequence("Ctrl+O"))
        self.add_files_button.setToolTip(_("You can select several files at once (Ctrl/Shift in the file manager)"))
        self.add_files_button.clicked.connect(self.on_add_files)
        buttons.addWidget(self.add_files_button)

        self.add_folder_button = QPushButton(_("🗀 Folder…"))
        self.add_folder_button.setObjectName("Ghost")
        self.add_folder_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.add_folder_button.setToolTip(_("Add all audio/video from a folder (recursively, including subfolders)"))
        self.add_folder_button.clicked.connect(self.on_add_folder)
        buttons.addWidget(self.add_folder_button)

        self.clear_files_button = QPushButton(_("Clear"))
        self.clear_files_button.setObjectName("Ghost")
        self.clear_files_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.clear_files_button.setToolTip(_("Remove ALL files from the list (files on disk are not deleted)"))
        self.clear_files_button.clicked.connect(lambda: self.file_list.clear_all())
        buttons.addWidget(self.clear_files_button)

        self.remove_selected_button = QPushButton(_("Remove selected"))
        self.remove_selected_button.setObjectName("Ghost")
        self.remove_selected_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.remove_selected_button.setToolTip(_("Remove the highlighted rows (Del). Select with the mouse + Ctrl/Shift, like in a file manager"))
        self.remove_selected_button.setShortcut(QKeySequence("Delete"))
        self.remove_selected_button.clicked.connect(lambda: self.file_list.remove_selected())
        buttons.addWidget(self.remove_selected_button)
        files_layout.addLayout(buttons)

        self.file_list = FileListWidget()
        self.file_list.setMinimumHeight(180)
        self.file_list.setToolTip(_(
            "Queue of files to recognize.\n"
            "Select with the mouse + Ctrl/Shift, delete the selection with Del\n"
            "or the \"Remove selected\" button. Files on disk are not modified."
        ))
        self.file_list.filesChanged.connect(self._update_files_label)
        files_layout.addWidget(self.file_list, stretch=1)

        self.drop_hint = make_label(
            _("Drag audio or video here (mp4, mkv, mp3, wav, m4a, ogg…) — "
              "video is converted to sound automatically."),
            "Muted",
            word_wrap=True,
        )
        files_layout.addWidget(self.drop_hint)
        layout.addWidget(files_box, stretch=3)

        # --- 2. OPTIONS
        options_box = QFrame()
        options_box.setObjectName("SidePanel")
        options_layout = QVBoxLayout(options_box)
        options_layout.setContentsMargins(14, 14, 14, 14)
        options_layout.setSpacing(10)
        options_layout.addWidget(make_label(_("2 · How to recognize and where to save"), "H2"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        language_label = make_label(_("Speech language"), "Dim")
        language_label.setToolTip(_(
            "The language spoken in the recording.\n"
            "\"Detect automatically\" — the model figures out the language itself (slightly\n"
            "slower and less reliable on short/noisy recordings than an explicit choice)."
        ))
        grid.addWidget(language_label, 0, 0)
        self.language_combo = QComboBox()
        for code, title in LANGUAGES:
            self.language_combo.addItem(_(title), code)
        self.language_combo.setToolTip(language_label.toolTip())
        language_index = self.language_combo.findData(self.settings.language)
        self.language_combo.setCurrentIndex(language_index if language_index >= 0 else 1)
        self.language_combo.currentIndexChanged.connect(self._on_settings_changed)
        grid.addWidget(self.language_combo, 0, 1, 1, 3)

        save_label = make_label(_("Where to save"), "Dim")
        save_label.setToolTip(_(
            "Folder for the result files (file name = original name).\n"
            "Default: outputs/ inside the project folder."
        ))
        grid.addWidget(save_label, 1, 0)
        save_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setText(self.settings.output_dir or str(paths.outputs_dir()))
        self.output_edit.setPlaceholderText(str(paths.outputs_dir()))
        self.output_edit.setToolTip(save_label.toolTip())
        self.output_edit.editingFinished.connect(self._on_settings_changed)
        save_row.addWidget(self.output_edit, stretch=1)
        browse_output = QPushButton(_("Browse…"))
        browse_output.setObjectName("Ghost")
        browse_output.setCursor(compat.CURSOR_POINTING_HAND)
        browse_output.setToolTip(_("Choose the results folder in the file manager"))
        browse_output.clicked.connect(self.on_browse_output)
        save_row.addWidget(browse_output)
        reset_output = QToolButton()
        reset_output.setText("↺")
        reset_output.setToolTip(_("Restore the default folder: outputs/ inside the project"))
        reset_output.setCursor(compat.CURSOR_POINTING_HAND)
        reset_output.clicked.connect(self._reset_output_dir)
        save_row.addWidget(reset_output)
        grid.addLayout(save_row, 1, 1, 1, 3)

        format_label = make_label(_("Result format"), "Dim")
        format_label.setToolTip(_(
            "Extension and structure of the result file:\n"
            "  TXT — plain text (notes, summaries);\n"
            "  SRT / VTT — subtitles for players and the web;\n"
            "  JSON — segments with timecodes (for further processing);\n"
            "  Markdown — text with an information header."
        ))
        grid.addWidget(format_label, 2, 0)
        self.format_combo = QComboBox()
        for value, title in (
            ("txt", _("TXT — plain text")),
            ("srt", _("SRT — subtitles")),
            ("vtt", _("VTT — subtitles for the web")),
            ("json", _("JSON — segments with timecodes")),
            ("md", _("Markdown — text with a header")),
        ):
            self.format_combo.addItem(title, value)
        self.format_combo.setToolTip(format_label.toolTip())
        self.format_combo.setCursor(compat.CURSOR_POINTING_HAND)
        format_index = self.format_combo.findData(self._ui_format(self.settings.output_format))
        self.format_combo.setCurrentIndex(format_index if format_index >= 0 else 0)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        grid.addWidget(self.format_combo, 2, 1, 1, 3)

        options_layout.addLayout(grid)

        self.output_hint = make_label("", "Muted", word_wrap=True)
        options_layout.addWidget(self.output_hint)

        # checkboxes wrap by rows (FlowLayout) instead of being clipped
        toggles = FlowLayout(spacing=14)
        self.vad_check = QCheckBox(_("VAD (cut silence)"))
        self.vad_check.setChecked(self.settings.use_vad)
        self.vad_check.setToolTip(_(
            "Silero VAD: cuts pauses, noise and music before recognition.\n"
            "Pros: faster and fewer hallucinations (\"Thanks for watching\" on silence).\n"
            "Turn off only if the pauses themselves matter (e.g. speech with long stops)."
        ))
        toggles.addWidget(self.vad_check)
        self.vad_check.toggled.connect(self._on_settings_changed)

        self.clean_check = QCheckBox(_("Anti-hallucinations"))
        self.clean_check.setChecked(self.settings.suppress_hallucinations)
        self.clean_check.setToolTip(_(
            "Removes \"[music]\", \"[applause]\", \"Thanks for watching\", "
            "subtitle inserts and looped phrase repeats.\n"
            "Keep it on: lectures with silence and music produce junk text without it."
        ))
        self.clean_check.toggled.connect(self._on_settings_changed)
        toggles.addWidget(self.clean_check)

        self.paragraph_check = QCheckBox(_("Paragraphs"))
        self.paragraph_check.setChecked(self.settings.paragraphs)
        self.paragraph_check.setToolTip(_(
            "Merges short recognition segments into paragraphs at meaningful pauses.\n"
            "Turn off for a \"telegraph\" format: every phrase on a new line."
        ))
        self.paragraph_check.toggled.connect(self._on_settings_changed)
        toggles.addWidget(self.paragraph_check)

        self.timestamps_check = QCheckBox(_("Timecodes in .txt"))
        self.timestamps_check.setChecked(
            bool(self.settings.add_timestamps_to_txt) or self.settings.output_format == "txt-time"
        )
        self.timestamps_check.setToolTip(_(
            "Adds [00:12:34]-style timestamps before phrases in the .txt — handy for\n"
            "finding a place in the recording. The result is still a plain .txt file.\n"
            "Applies only to the TXT format (disabled for the other formats)."
        ))
        self.timestamps_check.setEnabled(self._ui_format(self.settings.output_format) == "txt")
        self.timestamps_check.toggled.connect(self._on_settings_changed)
        toggles.addWidget(self.timestamps_check)

        self.advanced_toggle = QPushButton(_("▸ Advanced options"))
        self.advanced_toggle.setObjectName("Link")
        self.advanced_toggle.setCursor(compat.CURSOR_POINTING_HAND)
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(bool(self.settings.advanced_open))
        self.advanced_toggle.setToolTip(_(
            "Fine tuning: hint words, style prompt, beam, batched mode,\n"
            "word timestamps, translation, auto-opening the folder."
        ))
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        toggles.addWidget(self.advanced_toggle)
        options_layout.addLayout(toggles)

        self.advanced_frame = QFrame()
        advanced_layout = QVBoxLayout(self.advanced_frame)
        advanced_layout.setContentsMargins(0, 6, 0, 0)
        advanced_layout.setSpacing(8)
        self.advanced_frame.setVisible(bool(self.settings.advanced_open))
        if self.settings.advanced_open:
            self.advanced_toggle.setText(_("▾ Advanced options"))

        hot_row = QHBoxLayout()
        hot_label = make_label(_("Hint words (lecture terms)"), "Dim")
        hot_label.setToolTip(_(
            "Words the model keeps \"in mind\" while recognizing.\n"
            "Helps with rare terms and names: without a hint Whisper often spells them phonetically.\n"
            "Supported by the faster-whisper engine; other engines ignore the hint."
        ))
        hot_row.addWidget(hot_label)
        self.hotwords_edit = QLineEdit()
        self.hotwords_edit.setPlaceholderText(_("transformer, gradient descent, Kubernetes — comma-separated"))
        self.hotwords_edit.setText(self.settings.hotwords)
        self.hotwords_edit.setToolTip(hot_label.toolTip())
        self.hotwords_edit.editingFinished.connect(self._on_settings_changed)
        hot_row.addWidget(self.hotwords_edit, stretch=1)
        advanced_layout.addLayout(hot_row)

        prompt_row = QHBoxLayout()
        prompt_label = make_label(_("Style prompt"), "Dim")
        prompt_label.setToolTip(_(
            "initial_prompt: a short example of style and topic. The model continues it,\n"
            "so it keeps the style (scientific, conversational) and spells terms from the prompt.\n"
            "Do not duplicate the whole vocabulary here — use \"Hint words\" for that."
        ))
        prompt_row.addWidget(prompt_label)
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText(_("Example: Machine learning lecture, Russian language, scientific style."))
        self.prompt_edit.setText(self.settings.initial_prompt)
        self.prompt_edit.setToolTip(prompt_label.toolTip())
        self.prompt_edit.editingFinished.connect(self._on_settings_changed)
        prompt_row.addWidget(self.prompt_edit, stretch=1)
        advanced_layout.addLayout(prompt_row)

        extra_row = FlowLayout(spacing=12)
        self.beam_combo = QComboBox()
        for value, title in (
            (1, _("beam 1 (fast)")), (3, "beam 3"), (5, _("beam 5 (balanced)")), (8, _("beam 8 (accurate)")),
        ):
            self.beam_combo.addItem(title, value)
        beam_index = self.beam_combo.findData(int(self.settings.beam_size or 5))
        self.beam_combo.setCurrentIndex(beam_index if beam_index >= 0 else 2)
        self.beam_combo.currentIndexChanged.connect(self._on_settings_changed)
        self.beam_combo.setToolTip(_(
            "Beam size — how many phrase continuations the decoder keeps.\n"
            "1 — fastest; 5 — optimal for speech; 8 — slightly more accurate but noticeably slower.\n"
            "Keep 5 for regular lectures."
        ))
        extra_row.addWidget(self.beam_combo)

        self.batched_check = QCheckBox(_("Batched (up to 4× faster on GPU)"))
        self.batched_check.setChecked(bool(self.settings.batched))
        self.batched_check.setToolTip(_(
            "Batched decoding (BatchedInferencePipeline) in faster-whisper: several audio\n"
            "chunks are computed as one batch. Up to 4× gain on GPU; almost none on CPU.\n"
            "Needs more memory — turn off when RAM is tight."
        ))
        self.batched_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.batched_check)

        self.wordts_check = QCheckBox(_("Word timestamps"))
        self.wordts_check.setChecked(bool(self.settings.word_timestamps))
        self.wordts_check.setToolTip(_(
            "Compute timestamps of every word (only needed for accurate srt/vtt subtitles).\n"
            "Slows recognition down; turn off for a plain .txt result."
        ))
        self.wordts_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.wordts_check)

        self.translate_check = QCheckBox(_("Translate to EN"))
        self.translate_check.setChecked(bool(self.settings.translate_to_english))
        self.translate_check.setToolTip(_(
            "task=translate: the model directly outputs an English translation of what it heard.\n"
            "Turn off for notes of a Russian lecture; useful for foreign-language talks."
        ))
        self.translate_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.translate_check)

        self.open_folder_check = QCheckBox(_("Open the folder when done"))
        self.open_folder_check.setChecked(bool(self.settings.open_output_folder_when_done))
        self.open_folder_check.setToolTip(_(
            "When the queue finishes, open the results folder in the file manager.\n"
            "Handy for one-off runs; annoying for overnight folder batches."
        ))
        self.open_folder_check.toggled.connect(self._on_settings_changed)
        extra_row.addWidget(self.open_folder_check)

        settings_button = QPushButton(_("All settings…"))
        settings_button.setObjectName("Ghost")
        settings_button.setCursor(compat.CURSOR_POINTING_HAND)
        settings_button.setToolTip(_(
            "The full settings dialog: device (CPU/GPU), compute precision, threads,\n"
            "parallel files, engines and CUDA, about the program."
        ))
        settings_button.clicked.connect(self.on_open_settings)
        extra_row.addWidget(settings_button)
        advanced_layout.addLayout(extra_row)

        options_layout.addWidget(self.advanced_frame)
        layout.addWidget(options_box, stretch=2)

        # --- 3. PROGRESS + BUTTON
        work_box = QFrame()
        work_box.setObjectName("SidePanel")
        work_layout = QVBoxLayout(work_box)
        work_layout.setContentsMargins(14, 14, 14, 14)
        work_layout.setSpacing(10)

        work_layout.addWidget(make_label(_("3 · Work"), "H2"))

        # --- model selection: dropdown of ALL downloaded models ------------ #
        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_label = make_label(_("Model"), "Dim")
        model_label.setToolTip(_(
            "Which of the DOWNLOADED models to use for recognition.\n"
            "Only one can be selected; while \"Not selected\" is chosen, the RUN\n"
            "button is inactive. Download new models in the right panel."
        ))
        model_row.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.setToolTip(model_label.toolTip())
        self.model_combo.setCursor(compat.CURSOR_POINTING_HAND)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        model_row.addWidget(self.model_combo, stretch=1)
        work_layout.addLayout(model_row)
        self.rebuild_model_combo()

        # --- GPU status/button: offers the CUDA build of PyTorch when an
        # NVIDIA card exists but torch computes on the CPU -------------------
        self.gpu_button = QPushButton(_("Detecting hardware…"))
        self.gpu_button.setObjectName("Ghost")
        self.gpu_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.gpu_button.setEnabled(False)
        self.gpu_button.clicked.connect(self._install_torch_cuda)
        work_layout.addWidget(self.gpu_button)

        self.overall_progress = QProgressBar()
        self.overall_progress.setToolTip(_("Overall progress of the file queue"))
        self.overall_progress.setRange(0, 1000)
        self.overall_progress.setValue(0)
        # %p is the PERCENT (Qt recomputes it from the 0..1000 range);
        # %v would show the raw value — that used to produce "1000%" at the end
        self.overall_progress.setFormat("%p%")
        work_layout.addWidget(self.overall_progress)

        self.overall_label = make_label(_("Ready"), "Dim", word_wrap=True)
        work_layout.addWidget(self.overall_label)

        self.file_progress = QProgressBar()
        self.file_progress.setToolTip(_("Progress of the current file"))
        self.file_progress.setObjectName("Subtle")
        self.file_progress.setRange(0, 1000)
        self.file_progress.setValue(0)
        self.file_progress.setFixedHeight(6)
        self.file_progress.setFormat("")  # text does not fit on a 6-pixel stripe
        work_layout.addWidget(self.file_progress)

        self.file_label = make_label("", "Muted", word_wrap=True)
        work_layout.addWidget(self.file_label)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setToolTip(_(
            "Live feed of the recognized text of the current file\n"
            "(phrase start timecode + the phrase). The final result is in the file."
        ))
        self.preview.setPlaceholderText(_("Recognized text will appear here as the work progresses…"))
        self.preview.setMaximumHeight(96)
        self.preview.setMaximumBlockCount(1500)  # memory does not grow forever
        work_layout.addWidget(self.preview)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        self.work_button = QPushButton(_("RUN"))
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

        # toggle button: while working, RUN is replaced by CANCEL
        self.cancel_button = QPushButton(_("CANCEL"))
        self.cancel_button.setObjectName("Danger")
        self.cancel_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.cancel_button.setMinimumHeight(48)
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.on_cancel)
        action_row.addWidget(self.cancel_button, stretch=1)
        work_layout.addLayout(action_row)

        results_row = FlowLayout(spacing=8)
        self.results_button = QPushButton(_("Show results"))
        self.results_button.setObjectName("Ghost")
        self.results_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.results_button.setToolTip(_(
            "Table of processed files: status, result names, word count,\n"
            "speed and time. Appears after the first completed run."
        ))
        self.results_button.setEnabled(False)
        self.results_button.clicked.connect(self.on_show_results)
        results_row.addWidget(self.results_button)
        self.open_output_button = QPushButton(_("Open the results folder"))
        self.open_output_button.setObjectName("Ghost")
        self.open_output_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.open_output_button.setToolTip(_(
            "Open the folder where finished texts are saved\n"
            "(configured in the \"Where to save\" field)"
        ))
        self.open_output_button.clicked.connect(lambda: open_in_file_manager(self.current_output_dir()))
        results_row.addWidget(self.open_output_button)
        work_layout.addLayout(results_row)

        layout.addWidget(work_box, stretch=2)

        scroll.setWidget(page)
        return scroll

    # ---------------- right column ------------------------------------- #
    def _build_right_panel(self) -> QWidget:
        self.models_panel = ModelListPanel(self.catalog, self.hardware, settings=self.settings)
        self.models_panel.modelSelected.connect(self.on_model_selected)
        self.models_panel.downloadRequested.connect(self.on_download_model)
        self.models_panel.deleteRequested.connect(self.on_delete_model)
        self.models_panel.infoRequested.connect(self.on_model_info)
        self.models_panel.customRequested.connect(self.on_add_custom_model)
        self.models_panel.refreshRequested.connect(self.on_refresh_catalog)
        return self.models_panel

    # ---------------- log panel ---------------------------------------- #
    def _build_log_dock(self) -> None:
        self.log_dock = QDockWidget(_("Journal"), self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setAllowedAreas(compat.BOTTOM_DOCK_AREA | compat.TOP_DOCK_AREA)
        self.log_dock.setFeatures(compat.DOCK_CLOSABLE | compat.DOCK_MOVABLE)

        self.log_view = QPlainTextEdit()
        # no tooltip for the journal: it is obviously an event log, and a popup
        # on hover only distracts while scrolling
        self.log_view.setToolTip("")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText(_("Event journal…"))
        self.log_dock.setWidget(self.log_view)
        self.log_dock.setToolTip("")
        self.addDockWidget(compat.BOTTOM_DOCK_AREA, self.log_dock)
        self.log_dock.setVisible(bool(self.settings.show_log_panel))
        self.log_dock.visibilityChanged.connect(lambda *_: self._schedule_geometry_save())

    # ---------------- menus --------------------------------------------- #
    def _build_menu(self) -> None:
        bar = self.menuBar()

        file_menu = ToolTipMenu(_("File"), self)
        bar.addMenu(file_menu)
        file_menu.menuAction().setToolTip(_(
            "Working with files: add audio/video (or a whole folder),\n"
            "open results, show the processed files table, exit."
        ))
        self._add_action(file_menu, _("Add files…"), self.on_add_files, "Ctrl+O",
                         tip=_("Select audio/video files to recognize (several at once)"))
        self._add_action(file_menu, _("Add folder…"), self.on_add_folder, "Ctrl+Shift+O",
                         tip=_("Add all audio/video from a folder, including subfolders"))
        file_menu.addSeparator()
        self._add_action(file_menu, _("Open the results folder"),
                         lambda: open_in_file_manager(self.current_output_dir()),
                         tip=_("Open the folder where finished texts are saved"))
        self._add_action(file_menu, _("Show results"), self.on_show_results,
                         tip=_("Table of processed files: statuses, result names, speed"))
        file_menu.addSeparator()
        self._add_action(file_menu, _("Exit"), self.close, "Ctrl+Q",
                         tip=_("Close the program (a running job is stopped after confirmation)"))

        models_menu = ToolTipMenu(_("Models"), self)
        bar.addMenu(models_menu)
        models_menu.menuAction().setToolTip(_(
            "Recognition models: add your own (link/path), refresh the catalog,\n"
            "open the models/ folder, unload the loaded model from memory."
        ))
        self._add_action(models_menu, _("Add your own model (link/path)…"), self.on_add_custom_model,
                         tip=_("A model not in the catalog: a Hugging Face repo, a direct link or a folder on disk"))
        self._add_action(models_menu, _("Refresh the catalog online"), self.on_refresh_catalog,
                         tip=_("Check the online catalog for new models"))
        self._add_action(models_menu, _("Open the models/ folder"),
                         lambda: open_in_file_manager(self.catalog.models_root),
                         tip=_("The folder models are downloaded into (created automatically)"))
        models_menu.addSeparator()
        self.unload_action = self._add_action(
            models_menu, _("Unload the model from memory"), self._unload_model,
            tip=_("Free the RAM (and VRAM) taken by the loaded model.\n"
                "Available only while a model is actually loaded; the next run\n"
                "will load it again (a few seconds)."),
        )
        self.unload_action.setEnabled(False)

        service_menu = ToolTipMenu(_("Tools"), self)
        bar.addMenu(service_menu)
        service_menu.menuAction().setToolTip(_(
            "Service tools: program settings, hardware info,\n"
            "diagnostics and recovery, logs (startup log, logs folder)."
        ))
        self._add_action(service_menu, _("Settings…"), self.on_open_settings, "Ctrl+,",
                         tip=_("Text quality, device (CPU/GPU), journal (log files), reset settings to defaults"))
        self._add_action(service_menu, _("Hardware & environment…"), self.on_show_hardware,
                         tip=_("What the program sees: CPU, memory, GPU, ffmpeg"))
        service_menu.addSeparator()
        self._add_action(service_menu, _("Diagnostics & recovery…"), self.on_diagnostics,
                         tip=_("Check program files, folders and environment libraries;\n"
                             "restore deleted ones, repair the environment, remove extra packages"))
        self._add_action(service_menu, _("Open the startup log"), self.on_open_bootstrap_log,
                         tip=_("Environment preparation log (launcher/main.py). Deleted automatically\n"
                             "on a successful start — stays only in case of errors"))
        self._add_action(service_menu, _("Open the logs folder"),
                         lambda: open_in_file_manager(paths.logs_dir()),
                         tip=_("Program journals: install log and verbose launch logs\n"
                             "(the file log is enabled in Settings → \"Logging & diagnostics\")"))

        help_menu = ToolTipMenu(_("Help"), self)
        bar.addMenu(help_menu)
        help_menu.menuAction().setToolTip(_(
            "Help: project documentation, report an issue, about the program."
        ))
        self._add_action(help_menu, _("README (GitHub)"), lambda: QDesktopServices.openUrl(QUrl(REPO_URL)),
                         tip=_("Open the project page with documentation in the browser"))
        self._add_action(help_menu, _("Check for updates…"), lambda: self.on_check_updates(manual=True),
                         tip=_("Compare the installed version with the latest release on GitHub.\n"
                             "Only the changed program files are downloaded; models, settings\n"
                             "and results are not touched. Runs automatically on startup\n"
                             "(silently — when there is no internet or no new version)."))
        self._add_action(help_menu, _("Report an issue"),
                         lambda: QDesktopServices.openUrl(QUrl(f"{REPO_URL}/issues/new")),
                         tip=_("Open the GitHub new-issue form (an account is needed)"))
        help_menu.addSeparator()
        self._add_action(help_menu, _("About ({version})").format(version=APP_VERSION), self.on_about,
                         tip=_("Version, engines, Python interpreter and the project folder path"))

    def _add_action(self, menu: QMenu, text: str, slot, shortcut: str = "", tip: str = "") -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setToolTip(tip)
            action.setStatusTip(tip.splitlines()[0])   # shown in the status bar on hover
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    # ---------------- status bar ---------------------------------------- #
    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.status_hardware = QLabel(_("detecting hardware…"))
        self.status_model = QLabel(_("no model selected"))
        self.status_ffmpeg = QLabel("ffmpeg: …")
        for widget in (self.status_hardware, self.status_model, self.status_ffmpeg):
            widget.setObjectName("Muted")
            bar.addWidget(widget)
            separator = QLabel("  ·  ")
            separator.setObjectName("Muted")
            bar.addWidget(separator)

    # ================================================================== #
    #  Startup
    # ================================================================== #
    def _startup_checks(self) -> None:
        try:
            self.hardware = registry.hardware()
        except Exception as exc:  # noqa: BLE001
            log.warning(_("Failed to collect hardware info: %s"), exc)
            self.hardware = {}

        log.debug("Hardware at startup: %s", self.hardware)
        self.models_panel.set_hardware(self.hardware)
        self.models_panel.rebuild()
        self._refresh_gpu_button()
        self._start_torch_probe()

        gpu = self.hardware.get("vram_mb") or 0
        self.status_hardware.setText(
            _("CPU {threads} threads · RAM {ram:.0f} GB · ").format(
                threads=self.hardware.get("recommended_threads", "?"),
                ram=(self.hardware.get("ram_total_mb") or 0) / 1024,
            )
            + (_("GPU {vram:.1f} GB").format(vram=gpu / 1024)
               + (" (CUDA ✓)" if self.hardware.get("cuda_ok") else " (CUDA ✗)")
               if gpu else _("no GPU"))
        )
        self.status_hardware.setToolTip(self.hardware.get("summary", ""))
        ffmpeg = find_ffmpeg()
        self.status_ffmpeg.setText(_("ffmpeg: {state}").format(
            state=_("found") if ffmpeg else _("NOT FOUND")))
        self.status_ffmpeg.setToolTip(ffmpeg_version() if ffmpeg else _("Decoding will go through PyAV"))
        self.status_ffmpeg.setStyleSheet(
            "color: #98a2b3;" if ffmpeg else "color: #f2b544; font-weight: 600;"
        )

        for note in self.hardware.get("notes", []) or []:
            log.warning(note)

        self._restore_selected_model()
        self._update_output_hint()
        QTimer.singleShot(400, self._check_integrity)
        # Automatic update check a little later: the window is already visible,
        # and a missing internet connection ends the check silently.
        # Skipped in quiet/test mode (LOCALASR_QUIET) and when explicitly
        # disabled via LOCALASR_NO_UPDATE_CHECK=1.
        if (
            os.environ.get("LOCALASR_QUIET") != "1"
            and os.environ.get("LOCALASR_NO_UPDATE_CHECK") != "1"
        ):
            QTimer.singleShot(2500, lambda: self.on_check_updates(manual=False))

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
                self.status_model.setText(_("no model selected"))
                self.model_combo.setCurrentIndex(0)

    # ================================================================== #
    #  Files
    # ================================================================== #
    def on_add_files(self) -> None:
        start_dir = self.settings.last_open_dir or str(Path.home())
        picked, _sel = QFileDialog.getOpenFileNames(self, _("Choose audio or video"), start_dir, media_filter())
        if not picked:
            log.debug("Add files: dialog closed without a selection")
            return
        log.debug("Add files: selected %d (%s)", len(picked), "; ".join(picked[:6]))
        self.settings.last_open_dir = str(Path(picked[0]).parent)
        self.settings.save()
        added = self.file_list.add_paths(picked)
        self._log(_("Files added: {n} (total {total})").format(n=added, total=len(self.file_list.files())))

    def on_add_folder(self) -> None:
        start_dir = self.settings.last_open_dir or str(Path.home())
        picked = QFileDialog.getExistingDirectory(self, _("Folder with recordings"), start_dir)
        if not picked:
            return
        self.settings.last_open_dir = picked
        self.settings.save()
        added = self.file_list.add_paths([picked])
        self._log(_("Files added from the folder: {n}").format(n=added))
        if added == 0:
            QMessageBox.information(
                self, _("Empty"),
                _("No audio/video files found in this folder.\nSupported: {vid}… and {aud}…").format(
                    vid=", ".join(sorted(list(VIDEO_EXTENSIONS)[:8])),
                    aud=", ".join(sorted(list(AUDIO_EXTENSIONS)[:8])),
                ),
            )

    def on_browse_output(self) -> None:
        start_dir = self.output_edit.text().strip() or str(paths.outputs_dir())
        picked = QFileDialog.getExistingDirectory(self, _("Where to save the results"), start_dir)
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
        text = tn("{} file", "{} files", count).format(count)
        self.files_count_label.setText(
            text + (f" · {paths.human_size(total)}" if total else "")
        )
        self.drop_hint.setVisible(count == 0)
        self._update_work_button()

    # ================================================================== #
    #  Models
    # ================================================================== #
    def on_model_selected(self, entry: ModelEntry) -> None:
        if not entry.installed:
            answer = QMessageBox.question(
                self, _("Model not installed"),
                _("\"{name}\" has not been downloaded yet.\nDownload it now ({size})?").format(
                    name=entry.display_name, size=entry.size_text),
                compat.message_box_yes() | compat.message_box_no(),
                compat.message_box_yes(),
            )
            if answer == compat.message_box_yes():
                self.on_download_model(entry)
            return
        self._select_model(entry)

    def rebuild_model_combo(self) -> None:
        """Rebuild the dropdown of DOWNLOADED models (item 0 — "Not selected")."""
        combo = getattr(self, "model_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(_("— Not selected —"), "")
        for entry in self.catalog.sorted_for_display(self.catalog.installed()):
            combo.addItem(f"{entry.display_name} · {entry.size_text}", entry.id)
        current_id = self.selected_model.id if self.selected_model is not None else ""
        index = combo.findData(current_id)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _on_model_combo_changed(self, _index: int) -> None:
        model_id = self.model_combo.currentData() or ""
        if not model_id:
            # "Not selected" — clear the selection, the RUN button goes dark
            if self.selected_model is not None:
                self.selected_model = None
                self.models_panel.set_selected("")
                self.status_model.setText(_("no model selected"))
                self._update_work_button()
            return
        entry = self.catalog.get(model_id)
        if entry is not None and entry.installed:
            self._select_model(entry)

    def _select_model(self, entry: ModelEntry, save: bool = True) -> None:
        log.debug(
            "Model selection: id=%s name=%s engine=%s installed=%s dir=%s save=%s",
            entry.id, entry.name, entry.engine, entry.installed, entry.install_dir, save,
        )
        self.selected_model = entry
        self.models_panel.set_selected(entry.id)
        index = self.model_combo.findData(entry.id)
        if index >= 0 and self.model_combo.currentIndex() != index:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(index)
            self.model_combo.blockSignals(False)
        self.status_model.setText(_("model: {name}").format(name=entry.display_name))
        self.status_model.setToolTip(
            f"{_(entry.engine_label)} · {entry.size_text} · {entry.memory_text}\n{_(entry.description)}"
        )
        if save:
            self.settings.selected_model_id = entry.id
            self.settings.last_used_model_id = entry.id
            self.settings.save()
            self._log(_("Model selected: {name}").format(name=entry.display_name))
            self._toast(_("Model for the job: {name}").format(name=entry.display_name))
        self._update_work_button()

    def on_download_model(self, entry: ModelEntry) -> None:
        log.debug("Model install requested: %s (source=%s)", entry.id, entry.source)
        if self.download_worker is not None and self.download_worker.isRunning():
            QMessageBox.information(self, _("Download in progress"), _("Wait for the current download to finish."))
            return

        self.catalog.refresh_installed()
        entry = self.catalog.get(entry.id) or entry
        if entry.installed:
            self.on_model_selected(entry)
            return

        if not self._confirm_disk_space(entry):
            return

        dialog = ProgressDialog(
            _("Downloading: {name}").format(name=entry.display_name),
            _("{size} · {engine}\nSource: {source}").format(
                size=entry.size_text, engine=_(entry.engine_label), source=entry.source),
            parent=self,
        )
        worker = ModelDownloadWorker(entry, root=self.catalog.root, parent=self)
        card = self.models_panel.card(entry.id)
        if card is not None:
            card.set_downloading(True)

        worker.progress.connect(dialog.set_fraction)
        worker.progress.connect(lambda fraction, status: card.update_progress(fraction, status) if card else None)
        # IMPORTANT: the signal carries (fraction, text); the status slot needs
        # the text, otherwise QLabel.setText(float) crashed the app on every tick
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
        dialog.finish(_("Done") if ok else _("Stopped"))
        self.catalog.refresh_installed()
        self.models_panel.rebuild()
        self._restore_selected_model()   # rebuilds the model combo too

        if ok and entry is not None:
            self._log(message)
            fresh = self.catalog.get(entry.id) or entry
            self._select_model(fresh)
            QMessageBox.information(self, _("Model installed"), message + _("\n\nThe model is selected — ready to work."))
        else:
            self._log(_("Download did not finish: {msg}").format(msg=message))
            if message and not dialog.cancelled:
                QMessageBox.warning(self, _("Failed to download the model"), message)
        dialog.accept()
        self.download_worker = None
        self.download_dialog = None

    def _confirm_disk_space(self, entry: ModelEntry) -> bool:
        needed = entry.size_bytes or 0
        free = paths.free_space(self.catalog.models_root)
        if needed and free is not None and free < needed * 2:
            answer = QMessageBox.question(
                self, _("Low disk space"),
                _("The model takes {need}, but only {free} is available.\n"
                  "About twice as much is needed (a .part file is kept temporarily).\n\nContinue?").format(
                    need=paths.human_size(needed), free=paths.human_size(free)),
                compat.message_box_yes() | compat.message_box_no(),
                compat.message_box_no(),
            )
            return answer == compat.message_box_yes()
        return True

    def on_delete_model(self, entry: ModelEntry) -> None:
        if self._working:
            QMessageBox.information(
                self, _("Recognition in progress"),
                _("A model cannot be deleted while working — press CANCEL first "
                  "and wait for it to stop."),
            )
            return
        answer = QMessageBox.question(
            self, _("Delete the model"),
            _("Delete \"{name}\" from disk?\n\nFolder: {dir}\nFrees: {size}").format(
                name=entry.display_name, dir=entry.install_dir,
                size=paths.human_size(entry.installed_size_bytes or entry.size_bytes)),
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
            log.debug("Model deleted from disk: %s (%s)", entry.id, entry.install_dir)
            self._log(_("Model deleted: {name}").format(name=entry.display_name))
        else:
            QMessageBox.warning(
                self, _("Delete failed"),
                _("The model folder is outside models/ or busy with another process.\n"
                  "Delete it manually: {dir}").format(dir=entry.install_dir),
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
        self._log(_("Custom model added: {name} ({engine})").format(name=entry.display_name, engine=_(entry.engine_label)))
        self.on_download_model(entry)

    def on_refresh_catalog(self) -> None:
        if self.catalog_worker is not None and self.catalog_worker.isRunning():
            return
        self.models_panel.refresh_button.setEnabled(False)
        self.models_panel.refresh_button.setText(_("⟳ Checking…"))
        worker = CatalogRefreshWorker(self)
        worker.finishedWith.connect(self._on_catalog_refreshed)
        self.catalog_worker = worker
        worker.start()

    def _on_catalog_refreshed(self, ok: bool, message: str) -> None:
        self.models_panel.refresh_button.setEnabled(True)
        self.models_panel.refresh_button.setText(_("⟳ Refresh catalog"))
        if ok:
            self.catalog = ModelCatalog()
            self.models_panel.catalog = self.catalog
            self.models_panel.rebuild()
            self._restore_selected_model()
        self._log(message)

    # ================================================================== #
    #  Self-update (Help → Check for updates… + a silent check on startup)
    # ================================================================== #
    def on_check_updates(self, manual: bool = True, force: bool = False) -> None:
        """Start the version check in the background (GitHub API).

        manual=True  — user pressed the menu item: EVERY outcome is reported
                       (up to date / no internet / error).
        manual=False — startup auto-check: silent unless an update is found.
        force=True   — repair mode: audit the installation against its own
                       release tree even when no newer version exists.
        """
        if self.update_worker is not None and self.update_worker.isRunning():
            if manual:
                self._toast(_("The update check is already running…"))
            return
        if self.update_apply_worker is not None and self.update_apply_worker.isRunning():
            if manual:
                QMessageBox.information(
                    self, _("Update"), _("The update is already in progress."))
            return
        if manual:
            self._toast(_("Checking for updates…"))
            log.debug("Manual update check started (current version %s, force=%s)",
                      APP_VERSION, force)
        worker = UpdateCheckWorker(manual=manual, force=force, parent=self)
        worker.finishedWith.connect(self._on_update_checked)
        self.update_worker = worker
        worker.start()

    def _on_update_checked(self, info) -> None:
        worker, self.update_worker = self.update_worker, None
        if worker is not None:
            worker.deleteLater()

        if not info.ok:
            # no internet / GitHub unreachable / API error
            if info.manual:
                log.info("Update check: %s", info.error)
                QMessageBox.warning(
                    self, _("Update check"),
                    info.error or _("GitHub is unavailable. Check your internet connection."),
                )
            else:
                # the automatic startup check stays SILENT without internet
                log.debug("Automatic update check skipped: %s", info.error)
            return

        if not info.update_available:
            if info.manual:
                if info.repair:
                    text = _("All program files match version {version} — nothing to restore.").format(
                        version=info.latest_version or info.current_version)
                else:
                    # shows BOTH versions: a local build may be NEWER than the
                    # hub (unpublished release) — the user must see why nothing
                    # is offered instead of a bare "you are up to date"
                    text = _("No updates: the latest version on GitHub is {latest}, you have {version}.").format(
                        latest=info.latest_version or "—", version=info.current_version)
                self._log(text)
                QMessageBox.information(self, _("Update check"), text)
            else:
                log.debug("Automatic update check: up to date (%s)", info.current_version)
            return

        self._offer_update(info)

    def _offer_update(self, info) -> None:
        """Yes/No prompt: a newer version exists — update now?

        Before asking, the dialog explains HOW BIG the update is (major or
        minor, how many files are added/changed/removed, whether dependencies
        will be re-installed) so the user understands what they agree to.
        """
        from ..core import updater

        size_text = paths.human_size(info.total_bytes) if info.total_bytes else "—"
        added_n = len(info.added)
        changed_n = len(info.download) - added_n
        if updater.classify_update(info) == "major":
            kind = _("Major update: replaces most of the program files.")
        else:
            kind = _("Minor update: touches only a part of the program files.")
        details = _("To be added: {add}, changed: {chg}, removed: {del_} ({size}).").format(
            add=added_n, chg=changed_n, del_=len(info.delete), size=size_text)
        deps = _(
            "\n⚠ This update also refreshes the dependencies: at the next start "
            "the program will install new libraries (internet needed)."
        ) if info.deps_changed else ""
        data = _("Your models, settings, results and logs are not touched in any case.")
        notes = (info.release_notes or "").strip()
        if len(notes) > 400:
            notes = notes[:400].rstrip() + "…"
        notes_block = ("\n\n" + _("What's new:") + "\n" + notes) if notes and not info.repair else ""
        self._log(_("A new version is available: {ver} (installed: {cur})").format(
            ver=info.latest_version, cur=info.current_version))
        if info.repair:
            prompt = _(
                "Some program files do not match version {ver} — probably after "
                "an interrupted update.\n\n"
                "{details}{deps}\n"
                "{data}\n"
                "Only the mismatched files will be downloaded.{notes}\n\n"
                "Restore them now?"
            ).format(ver=info.latest_version or info.tag, details=details, deps=deps,
                     data=data, notes=notes_block)
        else:
            prompt = _(
                "A new version of {app} is available: {ver} (installed: {cur}).\n\n"
                "{kind}\n"
                "{details}{deps}\n"
                "{data}\n"
                "Only the changed files will be downloaded.{notes}\n\n"
                "Update now?"
            ).format(app=APP_NAME, ver=info.latest_version, cur=info.current_version,
                     kind=kind, details=details, deps=deps, data=data, notes=notes_block)
        answer = QMessageBox.question(
            self, _("Update available"), prompt,
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_yes(),
        )
        if answer == compat.message_box_yes():
            self._start_update_apply(info)

    def _start_update_apply(self, info) -> None:
        if self.update_apply_worker is not None and self.update_apply_worker.isRunning():
            return
        dialog = ProgressDialog(
            _("Updating {app}").format(app=APP_NAME),
            _("Downloading only the changed files of version {ver}…\n"
              "User data (models, settings, results, logs) is not affected.").format(
                ver=info.latest_version or info.tag),
            parent=self,
        )
        worker = UpdateApplyWorker(info, self)
        worker.progress.connect(dialog.set_fraction)
        worker.progress.connect(lambda fraction, status: dialog.set_status(status))
        worker.outputLine.connect(dialog.append_log)
        dialog.cancelRequested.connect(worker.cancel)
        worker.finishedWith.connect(
            lambda ok, message: self._on_update_applied(ok, message, info, dialog))
        self.update_dialog = dialog
        self.update_apply_worker = worker
        worker.start()
        dialog.exec()

    def _on_update_applied(self, ok: bool, message: str, info, dialog: ProgressDialog) -> None:
        dialog.finish(_("Done") if ok else _("Stopped"))
        dialog.accept()
        worker, self.update_apply_worker = self.update_apply_worker, None
        reason = getattr(worker, "reason", "") if worker is not None else ""
        if worker is not None:
            worker.deleteLater()
        self.update_dialog = None
        self._log(message)

        if not ok:
            QMessageBox.warning(self, _("Update"), message)
            if reason in ("verify", "failed"):
                # something is missing/broken on disk — offer to finish the job
                retry = QMessageBox.question(
                    self, _("Update"),
                    _("Re-run the update check now to download what is missing?"),
                    compat.message_box_yes() | compat.message_box_no(),
                    compat.message_box_yes(),
                )
                if retry == compat.message_box_yes():
                    # force=True: audit against the release tree even though the
                    # version already looks current (partial-update aftermath)
                    QTimer.singleShot(0, lambda: self.on_check_updates(manual=True, force=True))
            return

        # files on disk are new AND verified, but the RUNNING process still
        # holds the old code in memory — a restart activates the new version.
        # The verified-success message is shown INSIDE the restart question so
        # the user always sees one of the two outcomes (success / failure).
        question_text = (
            message + "\n\n" + _("Restart the application now to run the new version?")
        )
        if info.deps_changed:
            question_text += _(
                "\n\nThe update also refreshed the dependency list — the program "
                "will prepare them automatically at startup."
            )
        answer = QMessageBox.question(
            self, _("Update finished"), question_text,
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_yes(),
        )
        if answer == compat.message_box_yes():
            from ..core.relaunch import restart_app

            # a dependency change needs the bootstrap env repair on the way
            entry = "launcher/main.py" if info.deps_changed else None
            if restart_app(entry):
                app = QApplication.instance()
                if app is not None:
                    app.quit()
            else:
                QMessageBox.information(
                    self, _("Update finished"),
                    _("Automatic restart failed. Please close the application "
                      "and start it again to run the new version."),
                )
        else:
            self._toast(_("The new version will take effect after a restart"))

    def _unload_model(self) -> None:
        if self._working:
            QMessageBox.information(
                self, _("Recognition in progress"),
                _("The model is in use — it can be unloaded after the job finishes."),
            )
            return
        log.debug("Unloading the model from memory on user request")
        registry.unload()
        self._log(_("Model unloaded from memory"))
        self._sync_runtime_actions()

    def _sync_runtime_actions(self) -> None:
        """Keep state-dependent menu items honest.

        "Unload the model from memory" only makes sense while a model is
        actually loaded (and not while a job is running).
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
    #  Work
    # ================================================================== #
    def _update_work_button(self) -> None:
        files_ready = self.file_list.count() > 0
        model_ready = self.selected_model is not None and self.selected_model.installed
        if self._working:
            # while working, RUN is hidden and CANCEL is visible
            self.work_button.setEnabled(False)
            return
        self.work_button.setEnabled(files_ready and model_ready)
        if not files_ready:
            self.work_button.setText(_("Add files"))
        elif not model_ready:
            self.work_button.setText(_("Select a model"))
        else:
            count = self.file_list.count()
            self.work_button.setText(_("RUN · {}").format(tn("{} file", "{} files", count).format(count)))

    def _set_working(self, working: bool) -> None:
        """Switch the interface between "waiting" and "working".

        While working, RUN is REPLACED by CANCEL (and vice versa): the same
        spot, intuitive behavior.
        """
        self._working = working
        self.work_button.setVisible(not working)
        self.cancel_button.setVisible(working)
        if working:
            self.cancel_button.setEnabled(True)
            self.cancel_button.setText(_("CANCEL"))
        self._update_work_button()
        self._sync_runtime_actions()

    def on_work(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        files = self.file_list.files()
        if not files:
            QMessageBox.information(self, _("No files"), _("Add audio or video first."))
            return
        entry = self.selected_model
        if entry is None or not entry.installed:
            QMessageBox.information(
                self, _("No model selected"),
                _("Download and select a model in the right panel — there is nothing to recognize with.\n"
                  "Model cards show size, memory usage, languages, accuracy and speed\n"
                  "ratings — pick one that fits your hardware."),
            )
            self._focus_models()
            return

        from ..asr.base import engine_install_size
        from ..asr.registry import engine_available, missing_engine_packages

        if not engine_available(entry.engine):
            packages = missing_engine_packages(entry.engine)
            size_text = _(engine_install_size(entry.engine)) if engine_install_size(entry.engine) else ""
            answer = QMessageBox.question(
                self, _("Extra components needed"),
                _("The model \"{name}\" runs on the \"{engine}\" engine.\n"
                  "Its libraries will be installed into the program folder — the system\n"
                  "Python is not affected; this is a one-time operation.\n\n").format(
                    name=entry.display_name, engine=_(entry.engine_label))
                + (_("Download size: {size}\n").format(size=size_text) if size_text else "")
                + _("Packages: {list}\n\nInstall now?").format(list=", ".join(packages)),
                compat.message_box_yes() | compat.message_box_no(),
                compat.message_box_yes(),
            )
            if answer != compat.message_box_yes():
                return
            self._pending_job = True
            self.install_packages_interactive(packages, _("Engine {engine}").format(engine=entry.engine))
            return

        output_dir = self.current_output_dir()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            probe_file = output_dir / ".localasr-write-test"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink(missing_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, _("Folder unavailable"), _("Failed to write into {dir}:\n{exc}").format(dir=output_dir, exc=exc))
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
            "on_work: files=%s | model=%s (engine=%s, dir=%s) | output=%s | language=%s | "
            "beam=%s vad=%s clean=%s paragraphs=%s timestamps=%s words=%s batched=%s batch=%s | "
            "device=%s compute=%s threads=%s gpu_index=%s parallel=%s overwrite=%s",
            [str(f) for f in files], entry.id, entry.engine, entry.install_dir,
            options.output_dir, options.language or "auto",
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
        self.overall_label.setText(_("Processing {n} file(s) with \"{name}\"").format(n=len(files), name=entry.display_name))
        self.file_progress.setRange(0, 1000)
        self.file_progress.setValue(0)
        self._set_working(True)   # BEFORE start(): avoids the "button stayed lit" race
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
        self._log(_("Start: {n} file(s), model {name} ({engine}), language={lang}, format={fmt}").format(
            n=len(files), name=entry.display_name, engine=_(entry.engine_label),
            lang=self.settings.language or _("auto"), fmt=self.settings.output_format,
        ))

    # ---------------- worker signals ------------------------------------ #
    def _on_overall_progress(self, done: int, total: int) -> None:
        self._job_total_files = max(1, total)
        self._job_done_files = max(0, min(done, total))
        self.overall_progress.setRange(0, 1000)
        self.overall_progress.setValue(int(self._job_done_files / self._job_total_files * 1000))
        if total > 1:
            self.overall_label.setText(_("File {cur} of {total}").format(cur=min(done + 1, total), total=total))

    def _on_file_progress(self, index: int, fraction: float, stage: str) -> None:
        if fraction < 0:
            self.file_progress.setRange(0, 0)
        else:
            fraction = max(0.0, min(1.0, fraction))
            self.file_progress.setRange(0, 1000)
            self.file_progress.setValue(int(fraction * 1000))
            # overall bar: share of finished files + progress of the current one,
            # so a SINGLE file shows smooth percentages instead of standing at zero
            overall = (self._job_done_files + fraction) / max(1, self._job_total_files)
            self.overall_progress.setValue(max(0, min(1000, int(overall * 1000))))
        try:
            name = Path(self.file_list.files()[index]).name
        except IndexError:
            name = _("file {n}").format(n=index + 1)
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
        log.debug("Job finished: ok=%d fail=%d seconds=%.1f", ok, fail, seconds)
        cancelled = self.worker is not None and self.worker.cancelled
        engine_missing = bool(getattr(self.worker, "engine_missing", False))
        total = self._job_total_files
        produced = [path for result in self.results.values() for path in result.outputs]
        words = sum(result.words for result in self.results.values())
        audio = sum(result.audio_duration for result in self.results.values())
        speed = (audio / seconds) if seconds > 0 else 0.0

        if engine_missing:
            self.overall_label.setText(_(
                "Recognition postponed: the engine lacks components — install them in "
                "the window that appeared and the job will continue by itself."
            ))
            self.file_label.setText("")
            self._log(self.overall_label.text())
            self._finish_worker()
            return

        self.overall_label.setText(
            (_("Cancelled") if cancelled else _("Done")) + _(": {ok} of {total}").format(ok=ok, total=total)
            + (_(" · {fail} failed").format(fail=fail) if fail else "")
            + _(" · {words} words · {speed:.1f}× realtime · {t:.0f} s").format(
                words=words, speed=speed, t=seconds)
        )
        self.file_label.setText(", ".join(path.name for path in produced[:3]) + (" …" if len(produced) > 3 else ""))
        self._log(self.overall_label.text())

        self._finish_worker()

        if fail and not cancelled:
            failed = [result for result in self.results.values() if not result.ok]
            details = "\n".join(f"• {Path(result.source).name}: {result.error}" for result in failed[:12])
            QMessageBox.warning(
                self, _("Some files were not processed"),
                _("{fail} of {total} files finished with an error:\n\n{details}").format(
                    fail=fail, total=total, details=details),
            )
        elif ok:
            self.results_button.setEnabled(True)
            if self.settings.open_output_folder_when_done and produced and not cancelled:
                open_in_file_manager(produced[0])
            self._toast(_("Files recognized: {n}").format(n=ok))
            self._maybe_offer_gpu_boost()

    def _finish_worker(self) -> None:
        """Return the interface to the waiting state and drop the finished worker."""
        self._set_working(False)
        self.file_progress.setRange(0, 1000)
        self.file_progress.setValue(0)
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()

    def _on_job_failed(self, message: str) -> None:
        self._finish_worker()
        self.overall_label.setText(_("Error"))
        self.file_label.setText(message.splitlines()[0] if message else "")
        self._log(_("ERROR: ") + message.replace("\n", " | "))
        QMessageBox.critical(self, _("Recognition error"), message)

    def _on_engine_missing(self, engine: str, packages) -> None:
        self._finish_worker()
        from ..asr.base import engine_install_size

        size_text = _(engine_install_size(engine)) if engine_install_size(engine) else ""
        answer = QMessageBox.question(
            self, _("Extra components needed"),
            _("This model needs the \"{engine}\" engine.\n").format(engine=engine)
            + (_("Download size: {size}\n").format(size=size_text) if size_text else "")
            + _("Install {list} into the project environment?").format(list=", ".join(packages)),
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_yes(),
        )
        if answer == compat.message_box_yes():
            self._pending_job = True
            self.install_packages_interactive(list(packages or []), _("Engine {engine}").format(engine=engine))

    def on_cancel(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            self._finish_worker()
            return
        self.worker.cancel()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText(_("Stopping…"))
        self.file_label.setText(_("Stopping… (waiting for the current segment)"))
        self._log(_("Cancellation requested"))

    # ================================================================== #
    #  GPU acceleration for torch models (status button + one-time offer)
    # ================================================================== #
    def _start_torch_probe(self) -> None:
        """Asynchronously check whether torch is installed and sees CUDA (subprocess)."""
        if not self.hardware.get("has_nvidia"):
            return    # no NVIDIA card — no probe needed, the button is already gray
        if self.torch_probe is not None and self.torch_probe.isRunning():
            return
        probe = TorchCudaProbeWorker(self)
        probe.probed.connect(self._on_torch_probed)
        self.torch_probe = probe
        probe.start()

    def _on_torch_probed(self, installed: bool, cuda: bool, detail: str) -> None:
        log.debug("PyTorch probe: installed=%s cuda=%s detail=%r", installed, cuda, detail)
        self.gpu_state = {"probed": True, "installed": installed, "cuda": cuda, "detail": detail}
        probe, self.torch_probe = self.torch_probe, None
        if probe is not None:
            probe.deleteLater()
        self._refresh_gpu_button()
        if installed:
            log.info(
                _("PyTorch in the environment: %s, CUDA %s (%s)"),
                _("installed") if installed else _("absent"),
                _("available") if cuda else _("NOT available"),
                detail or "—",
            )

    def _refresh_gpu_button(self) -> None:
        """GPU button text/availability = the honest current acceleration state."""
        button = getattr(self, "gpu_button", None)
        if button is None:
            return
        hardware = self.hardware or {}
        gpu_name = str(hardware.get("gpu_name") or "NVIDIA")
        vram = int(hardware.get("vram_mb") or 0)
        state = self.gpu_state

        if not hardware:
            button.setText(_("Detecting hardware…"))
            button.setEnabled(False)
            return

        if not hardware.get("has_nvidia"):
            button.setText(_("GPU acceleration unavailable: no NVIDIA graphics card"))
            button.setToolTip(_(
                "All engines will run on the CPU.\n"
                "For the fastest CPU recognition pick int8/q5_0 models\n"
                "(faster-whisper / whisper.cpp) — their specs are in the catalog on the right."
            ))
            button.setEnabled(False)
            return

        if not state.get("probed"):
            button.setText(_("Checking PyTorch for {gpu}…").format(gpu=gpu_name))
            button.setEnabled(False)
            return

        if state.get("cuda"):
            button.setText(_("✓ GPU in use: PyTorch CUDA ({detail})").format(
                detail=state.get("detail") or gpu_name))
            button.setToolTip(_(
                "{gpu}{vram}\n"
                "The CUDA build of PyTorch is in place: GigaAM / Qwen3-ASR / HF models\n"
                "compute the encoder on the graphics card while decoding runs on the CPU.\n"
                "Nothing left to install — the button is inactive.\n\n"
                "(For faster-whisper Whisper models CUDA is configured separately:\n"
                "Tools → Settings → \"Engines\" tab.)"
            ).format(
                gpu=gpu_name,
                vram=_(" · {:.1f} GB VRAM").format(vram / 1024) if vram else "",
            ))
            button.setEnabled(False)
            return

        # NVIDIA exists but torch is a CPU build (or not installed) — can accelerate
        if state.get("installed"):
            text = _("⚡ Accelerate on GPU: PyTorch currently has no CUDA ({detail})").format(
                detail=state.get("detail") or "cpu")
        else:
            text = _("⚡ Accelerate on GPU: install PyTorch with CUDA (~2.5–3 GB)")
        button.setText(text)
        button.setToolTip(_(
            "The {gpu} graphics card{vram}is idle: the environment has a CPU build of PyTorch.\n"
            "Clicking installs the CUDA build (torch+torchaudio, once, ~2.5–3 GB) —\n"
            "torch models (GigaAM, Qwen3-ASR, any HF) speed up roughly 1.5–2×:\n"
            "the encoder moves to the GPU, decoding stays on the CPU.\n\n"
            "The installation goes strictly into the project environment; internet required."
        ).format(gpu=gpu_name,
                 vram=_(" ({:.1f} GB VRAM) ").format(vram / 1024) if vram else " "))
        button.setEnabled(True)

    def _install_torch_cuda(self) -> None:
        """Install the CUDA build of torch into the environment (removing the CPU build)."""
        if self.package_worker is not None and self.package_worker.isRunning():
            QMessageBox.information(self, _("Installation already running"), _("Wait for the previous installation to finish."))
            return
        hardware = self.hardware or {}
        gpu_name = str(hardware.get("gpu_name") or "NVIDIA")
        vram = int(hardware.get("vram_mb") or 0)
        driver = str(hardware.get("gpu_driver") or "")

        driver_note = ""
        try:
            major = int(driver.split(".")[0])
            if 0 < major < TORCH_CUDA_MIN_DRIVER_MAJOR:
                driver_note = _("\n\n⚠ NVIDIA driver {driver} is older than {min} — "
                                "the CUDA build may not work. Update the driver first (nvidia.com)."
                                ).format(driver=driver, min=TORCH_CUDA_MIN_DRIVER_MAJOR)
        except (ValueError, IndexError):
            pass

        answer = QMessageBox.question(
            self, _("GPU acceleration"),
            _("Graphics card: {gpu}").format(gpu=gpu_name)
            + (_(" · {:.1f} GB VRAM").format(vram / 1024) if vram else "")
            + (_(" · driver {driver}").format(driver=driver) if driver else "") + "\n\n"
            + _("Currently torch models (GigaAM, Qwen3-ASR, HF) compute on the CPU.\n"
                "Install the CUDA build of PyTorch into the project environment?\n\n"
                "• ~2.5–3 GB will be downloaded (once; later runs download nothing);\n"
                "• the encoder moves to the GPU: roughly 1.5–2× faster;\n"
                "• the system Python is not affected.")
            + driver_note + "\n\n" + _("Install now?"),
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_yes(),
        )
        if answer != compat.message_box_yes():
            return

        self._last_install_torch_cuda = True
        self.install_packages_interactive(
            ["--index-url", TORCH_CUDA_INDEX_URL, "torch>=2.6", "torchaudio>=2.6"],
            _("PyTorch CUDA (GPU acceleration)"),
            pre_uninstall=["torch", "torchaudio"],
        )

    def _maybe_offer_gpu_boost(self) -> None:
        """One-time "now / later / never" offer after a CPU run of a torch model
        on a machine with an NVIDIA card."""
        try:
            if self._last_job_engine not in TORCH_ENGINE_NAMES:
                return   # the job ran on a non-torch engine — the offer is off-topic
            if self.settings.gpu_boost_offered:
                return
            if not self.hardware.get("has_nvidia"):
                return
            state = self.gpu_state
            if not state.get("probed") or state.get("cuda"):
                return
            # Linux: pip torch already ships with CUDA by default; a CPU build
            # there is rare and the offer would only confuse. macOS: no CUDA.
            if os.name != "nt":
                return

            self.settings.gpu_boost_offered = True
            self.settings.save()

            box = QMessageBox(self)
            box.setWindowTitle(_("GPU acceleration available"))
            box.setIcon(QMessageBox.Icon.Information)
            gpu_name = str(self.hardware.get("gpu_name") or "NVIDIA")
            box.setText(_(
                "The job ran on the CPU even though you have a {gpu} graphics card:\n"
                "PyTorch in the environment is installed without CUDA.\n\n"
                "Installing the CUDA build (~2.5–3 GB, once) speeds up GigaAM /\n"
                "Qwen3-ASR / HF models roughly 1.5–2×.\n\n"
                "Install now? (possible at any time later — via the\n"
                "\"⚡ Accelerate on GPU\" button in block \"3 · Work\")"
            ).format(gpu=gpu_name))
            now = box.addButton(_("Install now"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(_("Later"), QMessageBox.ButtonRole.RejectRole)
            box.addButton(_("Do not offer again"), QMessageBox.ButtonRole.DestructiveRole)
            box.exec()
            if box.clickedButton() is now:
                self._install_torch_cuda()
        except Exception as exc:  # noqa: BLE001 - the offer must not crash the job finale
            log.debug("Failed to offer GPU acceleration: %s", exc)

    # ================================================================== #
    #  Package installation (engines, CUDA, environment repair)
    # ================================================================== #
    @staticmethod
    def _with_cuda_index(packages: List[str], hardware: Optional[dict]) -> List[str]:
        """Add the PyTorch CUDA index when installing torch on Windows+NVIDIA.

        On Linux pip installs CUDA torch anyway; macOS has no CUDA — hence
        Windows only. An explicit --index-url in the list is not overridden.

        IMPORTANT: --index-url FULLY replaces PyPI. For a torch-only list that
        is what we want (take the CUDA build). But engines install a mixed
        list (torch + transformers + hydra-core + …) which the PyTorch index
        does not have — the install would die with "No matching distribution".
        So for mixed lists the PyTorch index is added as an EXTRA one
        (--extra-index-url): pip/uv pick torch/torchaudio in the CUDA build
        (the +cu126 version sorts above the plain one per PEP 440) and the
        rest from PyPI.
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
            QMessageBox.information(self, _("Installation already running"), _("Wait for the previous installation to finish."))
            return

        # Windows: pip installs the CPU build of torch by default. With an
        # NVIDIA card and torch in the list, take the PyTorch CUDA index right
        # away so engines use the GPU out of the box.
        before = list(packages)
        packages = self._with_cuda_index(packages, self.hardware)
        if packages != before:
            self._log(_("NVIDIA GPU present — torch is installed in the CUDA build right away"))

        size_note = ""
        if any(pkg.lower().startswith("torch") for pkg in packages):
            size_note = _(
                "\nThe size is large because of PyTorch: ~120–200 MB (CPU build) — a one-time "
                "download; later runs download nothing."
            )
        dialog = ProgressDialog(
            title or _("Installing components"),
            _("Installing strictly into the project environment — the system Python is not touched.\n")
            + _("Packages: ") + ", ".join(packages) + size_note,
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
        dialog.finish(_("Done") if ok else _("Stopped"))
        self._log(message)
        registry.hardware(refresh=True)
        self.hardware = registry.hardware()
        self.models_panel.set_hardware(self.hardware)

        was_torch_cuda = self._last_install_torch_cuda
        self._last_install_torch_cuda = False

        if not ok:
            QMessageBox.warning(self, _("Installation did not finish"), message + self._install_hint(message))
        elif was_torch_cuda:
            # re-probe the torch state (subprocess) and hint about a restart:
            # if torch was already imported in this session, the process keeps
            # seeing the old (CPU) build until restarted
            self.gpu_state = {"probed": False, "installed": False, "cuda": False, "detail": ""}
            self._refresh_gpu_button()
            self._start_torch_probe()
            restart_note = ""
            if "torch" in sys.modules:
                restart_note = _(
                    "\n\nImportant: torch is already loaded in this session — restart the app\n"
                    "so the new CUDA build is picked up."
                )
            QMessageBox.information(
                self, _("PyTorch CUDA installed"),
                _("The CUDA build of PyTorch is installed into the project environment.\n"
                  "The \"⚡ Accelerate on GPU\" button updates after the check.") + restart_note,
            )
        elif getattr(self, "_pending_job", False):
            self._pending_job = False
            QTimer.singleShot(250, self.on_work)
        dialog.accept()
        self.package_worker = None
        self.package_dialog = None

    @staticmethod
    def _install_hint(message: str) -> str:
        """Human hint for the typical pip failure reasons."""
        low = (message or "").lower()
        if any(marker in low for marker in (
            "failed to build", "build wheel", "cmake", "winerror 2",
            "microsoft visual c++", "no compiler",
        )):
            return _(
                "\n\nIt looks like pip found no prebuilt wheel for one of the packages for your "
                "Python and tried to build it from source (which needs a compiler and CMake, "
                "usually absent).\n"
                "What to do:\n"
                "  1) update LocalASR to the latest version — dependency pins are relaxed, "
                "prebuilt wheels install;\n"
                "  2) if that does not help, install Python 3.11–3.13 and run the app with it "
                "(wheels for exotic new Pythons are not published immediately)."
            )
        if "could not find a version" in low or "no matching distribution" in low:
            return _(
                "\n\nOne of the packages cannot be found in PyPI for your Python version. "
                "Try Python 3.11–3.13 or check that the internet works without a proxy."
            )
        return ""

    def on_diagnostics(self) -> None:
        """Tools → Diagnostics & recovery (files, folders, environment)."""
        DiagnosticsDialog(self).exec()
        # hardware/engines may have changed after a repair
        registry.hardware(refresh=True)
        self.hardware = registry.hardware()
        self.models_panel.set_hardware(self.hardware)
        self._refresh_gpu_button()

    def _check_integrity(self) -> None:
        """Silent program-file check at startup.

        Breaks nothing: if the user deleted project files, report and offer
        to restore them from GitHub (one click).
        """
        try:
            from ..core import integrity

            missing = integrity.missing_project_files()
        except Exception as exc:  # noqa: BLE001
            log.debug("Integrity check failed: %s", exc)
            return
        if not missing:
            return
        log.warning(_("Program files are missing: %s"), ", ".join(missing[:10]))
        listing = "\n".join("  • " + name for name in missing[:12])
        if len(missing) > 12:
            listing += _("\n  … and {n} more").format(n=len(missing) - 12)
        answer = QMessageBox.question(
            self, _("Program files are missing"),
            _("The program folder lacks {n} file(s):\n{listing}\n\n"
              "The app will keep working but may be incomplete.\n"
              "Re-download the missing files from the project repository?\n"
              "(internet required; a few kilobytes, user data is not affected)").format(
                n=len(missing), listing=listing),
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_yes(),
        )
        if answer == compat.message_box_yes():
            self._restore_files_interactive(missing)

    def _restore_files_interactive(self, missing: List[str]) -> None:
        if self.restore_worker is not None and self.restore_worker.isRunning():
            return
        dialog = ProgressDialog(
            _("Restoring program files"),
            _("Downloading {n} file(s) from the project repository…").format(n=len(missing)),
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
        dialog.finish(_("Done") if ok else _("Stopped"))
        dialog.accept()
        worker, self.restore_worker = self.restore_worker, None
        if worker is not None:
            worker.deleteLater()
        self._log(message)
        if ok:
            self._toast(_("Program files restored"))
        else:
            QMessageBox.warning(self, _("Recovery"), message)

    def on_open_bootstrap_log(self) -> None:
        log_file = paths.logs_dir() / "bootstrap.log"
        error_log = paths.ROOT_DIR / "logs" / "bootstrap_error.log"
        target = log_file if log_file.is_file() else error_log
        if not target.is_file():
            QMessageBox.information(
                self, _("Logs"),
                _("There is no startup log — and that is good news.\n\n"
                  "bootstrap.log is written only while the environment is prepared and\n"
                  "is deleted automatically on a successful start.\n"
                  "It stays on disk only if the launch ended with an error."),
            )
            return
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            QMessageBox.warning(self, _("Logs"), _("Failed to read: {exc}").format(exc=exc))
            return
        dialog = ProgressDialog(_("Log: {name}").format(name=target.name), str(target), self)
        dialog.set_fraction(1.0)
        dialog.set_status(f"{target}")
        dialog.append_log(text[-60000:])
        dialog.finish(_("Read only"))
        dialog.cancel_button.setText(_("Close"))
        dialog.exec()

    # ================================================================== #
    #  Misc
    # ================================================================== #
    def on_open_settings(self) -> None:
        self._collect_settings()
        dialog = SettingsDialog(
            self.settings, self.hardware, self,
            on_reset_defaults=self._reset_settings_to_defaults,
        )
        if dialog.exec() == compat.dialog_accepted():
            self._apply_settings_to_ui()
            log.debug("Settings applied from the dialog: %s", self.settings.to_dict())
            self._log(_("Settings saved"))

    def _reset_settings_to_defaults(self) -> None:
        """Called from the settings dialog after a reset: apply defaults to the UI.

        By this moment `self.settings` is already reset and saved. All that
        remains is bringing the interface and the window/filters back to the
        original look.
        """
        self._apply_settings_to_ui()
        # window geometry — back to default
        self.settings.window_geometry = ""
        self.settings.window_state = ""
        self.settings.splitter_sizes = []
        try:
            self.splitter.setSizes([720, 520])
            self.log_dock.setVisible(bool(self.settings.show_log_panel))
            self.showNormal()
            self.resize(1280, 800)
        except Exception as exc:  # noqa: BLE001
            log.debug("Failed to reset geometry: %s", exc)
        self.settings.save()
        # catalog filters — back to defaults
        try:
            self.models_panel.reset_filters_to_defaults()
        except Exception as exc:  # noqa: BLE001
            log.debug("Failed to reset filters: %s", exc)
        # file log — back to the default state (off)
        try:
            from .. import logutils

            logutils.set_file_logging(bool(self.settings.file_logging))
        except Exception as exc:  # noqa: BLE001
            log.debug("Failed to apply the logging state: %s", exc)
        self._update_output_hint()
        self._log(_("Settings reset to defaults"))
        self._toast(_("Settings reset to defaults"))

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
            self, _("About {app}").format(app=APP_NAME),
            _("<h3>{app} {version}</h3>"
              "<p>Local speech recognition: audio and video → text, subtitles, JSON.</p>"
              "<p>Engines: faster-whisper (CTranslate2), whisper.cpp (GGML), Vosk (Kaldi), "
              "Qwen3-ASR and GigaAM (torch), plus the generic hf-transformers "
              "for any Hugging Face ASR model.<br>"
              "GUI: PySide6. Everything is installed inside the project folder.</p>"
              "<p>Python: {py}<br>Interpreter: {exe}<br>"
              "Project: {root}</p>").format(
                app=APP_NAME, version=APP_VERSION,
                py=sys.version.split()[0], exe=sys.executable, root=paths.ROOT_DIR),
        )

    # ---------------- service ------------------------------------------- #
    @staticmethod
    def _ui_format(fmt: str) -> str:
        """settings.output_format -> a value of the format combo.

        "txt-time" is not a separate combo item: it means TXT + the checked
        "Timecodes in .txt" box (kept for settings.json/CLI compatibility).
        """
        fmt = (fmt or "txt").strip().lower()
        if fmt == "txt-time":
            return "txt"
        return fmt if fmt in ("txt", "srt", "vtt", "json", "md") else "txt"

    def _collect_settings(self) -> None:
        self.settings.language = self.language_combo.currentData() or ""
        # result format: the combo in the "2 · How to recognize" block;
        # "Timecodes in .txt" upgrades txt to txt-time at the start of the job
        self.settings.output_format = self._ui_format(self.format_combo.currentData() or "txt")
        # IMPORTANT: the default value ("outputs/ inside the project") is NOT
        # saved as an absolute path — otherwise moving the project folder would
        # keep sending results to the OLD path. Empty string = "default".
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
        fmt = self._ui_format(self.settings.output_format)
        format_index = self.format_combo.findData(fmt)
        self.format_combo.setCurrentIndex(format_index if format_index >= 0 else 0)
        self.vad_check.setChecked(self.settings.use_vad)
        self.clean_check.setChecked(self.settings.suppress_hallucinations)
        self.paragraph_check.setChecked(self.settings.paragraphs)
        # "txt-time" in the settings means TXT + timecodes (CLI compatibility)
        self.timestamps_check.setChecked(
            bool(self.settings.add_timestamps_to_txt) or self.settings.output_format == "txt-time"
        )
        self.timestamps_check.setEnabled(fmt == "txt")
        self.wordts_check.setChecked(self.settings.word_timestamps)
        self.translate_check.setChecked(self.settings.translate_to_english)
        beam = int(self.settings.beam_size or 5)
        index = self.beam_combo.findData(beam)
        if index < 0:
            # do not lose a value set via "All settings…" (e.g. 7) — add an item
            self.beam_combo.addItem(f"beam {beam}", beam)
            index = self.beam_combo.count() - 1
        self.beam_combo.setCurrentIndex(index)
        self.batched_check.setChecked(bool(self.settings.batched))
        self.hotwords_edit.setText(self.settings.hotwords)
        self.prompt_edit.setText(self.settings.initial_prompt)
        self.open_folder_check.setChecked(self.settings.open_output_folder_when_done)
        # state of the "Advanced options" panel
        self.advanced_toggle.blockSignals(True)
        self.advanced_toggle.setChecked(bool(self.settings.advanced_open))
        self.advanced_toggle.blockSignals(False)
        self.advanced_frame.setVisible(bool(self.settings.advanced_open))
        self.advanced_toggle.setText(
            _("▾ Advanced options") if self.settings.advanced_open else _("▸ Advanced options")
        )
        self._update_output_hint()

    def _on_settings_changed(self, *_args) -> None:
        self._collect_settings()
        self._update_output_hint()

    def _on_format_changed(self, *_args) -> None:
        # timecodes only make sense in a plain .txt — disable the box otherwise
        fmt = self.format_combo.currentData() or "txt"
        self.timestamps_check.setEnabled(fmt == "txt")
        self._on_settings_changed()

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_frame.setVisible(checked)
        self.advanced_toggle.setText(_("▾ Advanced options") if checked else _("▸ Advanced options"))
        self.settings.advanced_open = bool(checked)
        self.settings.save()

    def _update_output_hint(self) -> None:
        output_dir = self.current_output_dir()
        fmt = self.format_combo.currentData() or "txt"
        if fmt == "txt" and self.timestamps_check.isChecked():
            fmt = "txt-time"
        sample = paths.sanitize_stem(_("lecture 1.mp4"))
        suffix = paths.output_suffix(fmt)
        sample_name = f"{sample} {paths.datetime_stamp()}{suffix}"
        note = _(
            "(original name + date and time — results of different runs\n"
            "never overwrite each other)"
        )
        try:
            outside = paths.ROOT_DIR.resolve() not in output_dir.resolve().parents
        except OSError:
            outside = False
        if outside:
            note = _("⚠ the folder is OUTSIDE the project — make sure you chose it deliberately.\n") + note
        self.output_hint.setText(_("Result: {path} / \"{name}\" {note}").format(
            path=paths.short_path(output_dir, 46), name=sample_name, note=note))

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

    # ---------------- window drag&drop ---------------------------------- #
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
        log.debug("Drag&drop: %d object(s): %s", len(urls), "; ".join(urls[:6]))
        if urls:
            added = self.file_list.add_paths(urls)
            self._log(_("Added by drag&drop: {n}").format(n=added))
            event.acceptProposedAction()

    # ---------------- geometry/closing ----------------------------------- #
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
            log.debug("Failed to restore window geometry: %s", exc)

    def _save_geometry_now(self) -> None:
        """Write the current window geometry/state to settings (atomically)."""
        try:
            self.settings.window_geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
            self.settings.window_state = bytes(self.saveState().toBase64()).decode("ascii")
            self.settings.splitter_sizes = [int(v) for v in self.splitter.sizes()]
            self.settings.show_log_panel = self.log_dock.isVisible()
            self.settings.save()
        except Exception as exc:  # noqa: BLE001
            log.debug("Failed to save geometry: %s", exc)

    def _schedule_geometry_save(self) -> None:
        """Debounce: the window can be moved continuously — save once after a pause.

        Any change (size, position, minimized/maximized state, panels) thus
        survives even an abnormal termination.
        """
        try:
            self._geo_timer.start()
        except RuntimeError:
            pass    # the window is already deleted — nowhere to save

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._schedule_geometry_save()

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        self._schedule_geometry_save()

    def changeEvent(self, event) -> None:  # noqa: N802
        # window state change (minimized/maximized/normal)
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
                self, _("Processing in progress"),
                _("Recognition has not finished yet. Stop and exit?"),
                compat.message_box_yes() | compat.message_box_no(),
                compat.message_box_no(),
            )
            if answer != compat.message_box_yes():
                event.ignore()
                return

        # All background threads are stopped GUARANTEED (cancel → wait →
        # terminate). Otherwise Qt logs "QThread: Destroyed while thread is
        # still running" on exit, and network retries/pip may outlive the window.
        from .workers import stop_worker

        log.debug("closeEvent: stopping background threads")
        stop_worker(self.worker, 6000, "TranscribeWorker")
        for name, worker in (
            ("download", self.download_worker),
            ("package", self.package_worker),
            ("restore", self.restore_worker),
            ("catalog", self.catalog_worker),
            ("torch_probe", self.torch_probe),
            ("update_check", self.update_worker),
            ("update_apply", self.update_apply_worker),
        ):
            stop_worker(worker, 4000, name)

        self._collect_settings()
        self._save_geometry_now()

        detach_qt_sink(self.log_handler)
        registry.unload()
        log.debug("closeEvent: finished cleanly")
        event.accept()


# --------------------------------------------------------------------------- #
#  Results table
# --------------------------------------------------------------------------- #


class ResultsDialog(QDialog):
    def __init__(self, results: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.results = results
        self.setWindowTitle(_("Results"))
        self.setModal(True)
        self.resize(920, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(10)
        root.addWidget(make_label(_("What came out"), "H2"))

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            [_("File name"), _("Status"), _("Result"), _("Words"), _("Speed"), _("Time")])
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
            status = QTableWidgetItem(_("✓ done") if result.ok else _("✗ error"))
            status.setForeground(compat.COLOR_GREEN if result.ok else compat.COLOR_RED)
            self.table.setItem(row, 1, status)
            outputs = QTableWidgetItem(", ".join(p.name for p in result.outputs) or (result.error or "—"))
            outputs.setToolTip(result.error or "\n".join(str(p) for p in result.outputs))
            self.table.setItem(row, 2, outputs)
            self.table.setItem(row, 3, QTableWidgetItem(str(result.words)))
            self.table.setItem(row, 4, QTableWidgetItem(f"{result.speed_x:.1f}×"))
            self.table.setItem(row, 5, QTableWidgetItem(_("{:.0f} s").format(result.elapsed_seconds)))
            if result.ok and result.outputs:
                self.table.item(row, 2).setData(compat.ROLE_USER, str(result.outputs[0]))

        root.addWidget(self.table, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        open_button = QPushButton(_("Open file"))
        open_button.setObjectName("Primary")
        open_button.clicked.connect(self._open_current)
        buttons.addWidget(open_button)
        folder_button = QPushButton(_("Open folder"))
        folder_button.setObjectName("Ghost")
        folder_button.clicked.connect(self._open_folder)
        buttons.addWidget(folder_button)
        close_button = QPushButton(_("Close"))
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
            QMessageBox.information(self, _("Results"), _("Select a row with a finished file."))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_folder(self) -> None:
        path = self._current_path()
        open_in_file_manager(path or paths.outputs_dir())
