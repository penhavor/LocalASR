"""Dialogs: progress, custom model, settings, hardware, diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import i18n, paths
from ..constants import ENGINE_LABELS, ENGINES, LANGUAGES
from ..core.models_catalog import ModelEntry
from ..i18n import _
from ..settings import Settings
from . import compat
from .widgets import make_label, open_url, stars


# --------------------------------------------------------------------------- #
#  Progress with cancellation
# --------------------------------------------------------------------------- #


class ProgressDialog(QDialog):
    """Universal progress window (model download, package install, work)."""

    cancelRequested = Signal()

    def __init__(self, title: str, description: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(620)
        self.setWindowFlag(compat.WINDOW_CONTEXT_HELP_HINT, False)

        self._cancelled = False
        self._done = False

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        self.title_label = make_label(title, "H2")
        root.addWidget(self.title_label)

        self.description = make_label(description or "", "Dim", word_wrap=True)
        self.description.setVisible(bool(description))
        root.addWidget(self.description)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.status_label = make_label(_("Preparing…"), "Dim", word_wrap=True)
        root.addWidget(self.status_label)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setFixedHeight(150)
        font = QFont("Consolas" if os.name == "nt" else "Monospace")
        font.setPointSize(9)
        self.log.setFont(font)
        root.addWidget(self.log)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QPushButton(_("Cancel"))
        self.cancel_button.setObjectName("Danger")
        self.cancel_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.cancel_button.clicked.connect(self.cancel)
        buttons.addWidget(self.cancel_button)
        root.addLayout(buttons)

    # ------------------------------------------------------------------ #
    def cancel(self) -> None:
        if self._cancelled:
            self.reject()
            return
        self._cancelled = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText(_("Stopping…"))
        self.status_label.setText(_("Cancelling — waiting for the current operation to finish…"))
        self.cancelRequested.emit()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def set_status(self, text: object) -> None:
        # type guard: worker signals may carry anything, and QLabel.setText(float)
        # used to crash the app with a TypeError on every tick
        self.status_label.setText("" if text is None else str(text))

    def set_fraction(self, fraction: object) -> None:
        try:
            value = float(fraction)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            value = -1.0
        if value < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(max(0.0, min(1.0, value)) * 1000))

    def append_log(self, line: str) -> None:
        if line:
            self.log.appendPlainText(line.rstrip())

    def finish(self, message: Optional[str] = None) -> None:
        self._done = True
        self.set_fraction(1.0)
        self.status_label.setText(message if message is not None else _("Done"))
        self.cancel_button.setText(_("Close"))
        self.cancel_button.setObjectName("Primary")
        self.cancel_button.style().unpolish(self.cancel_button)
        self.cancel_button.style().polish(self.cancel_button)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setCursor(compat.CURSOR_POINTING_HAND)
        try:
            self.cancel_button.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.cancel_button.clicked.connect(self.accept)

    def closeEvent(self, event) -> None:  # noqa: N802
        """The X button cancels the operation. After cancel/finish it closes."""
        if self._done or self._cancelled:
            event.accept()
            return
        self.cancel()
        event.ignore()


# --------------------------------------------------------------------------- #
#  Custom model by link
# --------------------------------------------------------------------------- #


class CustomModelDialog(QDialog):
    """Add any model: HF repository, direct link or a folder on disk."""

    def __init__(self, parent: Optional[QWidget] = None, models_root: Optional[Path] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Custom model"))
        self.setModal(True)
        self.setMinimumWidth(680)
        self.entry: Optional[ModelEntry] = None
        self._models_root = Path(models_root) if models_root else paths.models_dir()

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        root.addWidget(
            make_label(_("You can add any model, not only the catalog ones."), "H2")
        )
        root.addWidget(
            make_label(
                _("Three kinds of sources are supported:\n"
                  "  • a Hugging Face repository — e.g. coriollon/whisper-large-v3-turbo-russian\n"
                  "    (a subfolder works too: owner/name/tree/main/ct2_int8_float16)\n"
                  "  • a direct file link — ggml-*.bin, *.zip (Vosk), model.bin\n"
                  "  • a folder or file on disk — e.g. D:\\models\\my-whisper\n\n"
                  "The engine is detected automatically from the content."),
                "Dim",
                word_wrap=True,
            )
        )

        form = QFormLayout()
        form.setSpacing(10)

        source_row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText(_("owner/name, https://… or C:\\path\\to\\model"))
        source_row.addWidget(self.source_edit, stretch=1)

        self.browse_button = QPushButton(_("Browse…"))
        self.browse_button.setObjectName("Ghost")
        self.browse_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.browse_button.clicked.connect(self._browse)
        source_row.addWidget(self.browse_button)
        form.addRow(_("Source"), source_row)

        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText(_("leave empty — we will invent one (the folder name in models/)"))
        form.addRow(_("Model ID"), self.id_edit)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(_("the name you will see in the list"))
        form.addRow(_("Name"), self.name_edit)

        self.engine_combo = QComboBox()
        self.engine_combo.addItem(_("Detect automatically"), "")
        for engine in ENGINES:
            self.engine_combo.addItem(_(ENGINE_LABELS[engine]), engine)
        form.addRow(_("Engine"), self.engine_combo)

        self.languages_edit = QLineEdit()
        self.languages_edit.setPlaceholderText(_("ru, en (comma-separated)"))
        self.languages_edit.setText("ru, en")
        form.addRow(_("Languages"), self.languages_edit)

        self.ram_spin = QSpinBox()
        self.ram_spin.setRange(0, 200_000)
        self.ram_spin.setSuffix(_(" MB"))
        self.ram_spin.setSpecialValueText(_("estimate automatically"))
        form.addRow(_("Memory in use"), self.ram_spin)

        root.addLayout(form)

        self.inspect_button = QPushButton(_("🔍 Detect what kind of model this is"))
        self.inspect_button.setObjectName("Ghost")
        self.inspect_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.inspect_button.clicked.connect(self._inspect)
        root.addWidget(self.inspect_button)

        self.result_label = make_label("", "Dim", word_wrap=True)
        self.result_label.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
        root.addWidget(self.result_label)

        buttons = QDialogButtonBox(compat.dialog_buttons("Ok", "Cancel"))
        buttons.button(compat.dialog_buttons("Ok")).setText(_("Add and download"))
        buttons.button(compat.dialog_buttons("Ok")).setObjectName("Primary")
        buttons.button(compat.dialog_buttons("Cancel")).setText(_("Cancel"))
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------ #
    def _browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        picked = QFileDialog.getExistingDirectory(self, _("Model folder"), str(self._models_root))
        if picked:
            self.source_edit.setText(picked)
            self._inspect()

    def _inspect(self) -> None:
        source = self.source_edit.text().strip()
        if not source:
            self.result_label.setText(_("Specify a source first."))
            return
        self.inspect_button.setEnabled(False)
        self.result_label.setText(_("Checking the source…"))
        try:
            from ..core.downloader import inspect_custom_source

            entry = inspect_custom_source(source)
        except Exception as exc:  # noqa: BLE001
            self.result_label.setText(_("Did not work: {exc}").format(exc=exc))
            self.inspect_button.setEnabled(True)
            return

        self.inspect_button.setEnabled(True)
        if not self.id_edit.text().strip():
            self.id_edit.setText(entry.id)
        if not self.name_edit.text().strip():
            self.name_edit.setText(entry.name)
        engine_index = self.engine_combo.findData(entry.engine)
        if engine_index >= 0:
            self.engine_combo.setCurrentIndex(engine_index)
        if entry.ram_mb:
            self.ram_spin.setValue(entry.ram_mb)

        self.result_label.setText(_(
            "Engine: {engine}\nSize: {size}\nFiles: the source is recognized, ready to add."
        ).format(engine=_(ENGINE_LABELS.get(entry.engine, entry.engine)), size=entry.size_text))
        self._probed = entry

    def _accept(self) -> None:
        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, _("Custom model"), _("Specify a source: link, HF repository or path."))
            return

        probed: Optional[ModelEntry] = getattr(self, "_probed", None)
        if probed is None or probed.source != source:
            try:
                from ..core.downloader import inspect_custom_source

                probed = inspect_custom_source(source)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, _("Custom model"), _("Failed to parse the source:\n{exc}").format(exc=exc))
                return

        model_id = self.id_edit.text().strip() or probed.id
        probed.id = paths.sanitize_stem(model_id).replace(" ", "-").lower() or probed.id
        probed.name = self.name_edit.text().strip() or probed.name

        engine = self.engine_combo.currentData()
        if engine:
            probed.engine = engine

        languages = [
            part.strip().lower()
            for part in self.languages_edit.text().replace(";", ",").split(",")
            if part.strip()
        ]
        if languages:
            probed.languages = languages
            probed.languages_note = ""

        if self.ram_spin.value() > 0:
            probed.ram_mb = self.ram_spin.value()

        probed.custom = True
        probed.tags = list(dict.fromkeys([*probed.tags, "custom model"]))
        self.entry = probed
        self.accept()


# --------------------------------------------------------------------------- #
#  Hardware info
# --------------------------------------------------------------------------- #


class HardwareDialog(QDialog):
    def __init__(self, summary: str, ffmpeg_text: str, notes: List[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Hardware & environment"))
        self.setModal(True)
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(10)
        root.addWidget(make_label(_("What LocalASR sees"), "H2"))

        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setPlainText(summary + "\n\nffmpeg: " + ffmpeg_text)
        box.setMinimumHeight(180)
        root.addWidget(box)

        for note in notes:
            root.addWidget(make_label("• " + note, "Dim", word_wrap=True))

        buttons = QDialogButtonBox(compat.dialog_buttons("Close"))
        buttons.button(compat.dialog_buttons("Close")).setText(_("Close"))
        buttons.rejected.connect(self.accept)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)


# --------------------------------------------------------------------------- #
#  Settings
# --------------------------------------------------------------------------- #


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings: Settings,
        hardware: dict,
        parent: Optional[QWidget] = None,
        on_reset_defaults=None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.hardware = hardware or {}
        self._on_reset_defaults = on_reset_defaults
        self._restart_requested = False
        self.setWindowTitle(_("Settings"))
        self.setModal(True)
        self.setMinimumSize(700, 640)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll_wrap(self._tab_interface()), _("Interface"))
        self.tabs.addTab(self._scroll_wrap(self._tab_quality()), _("Text quality"))
        self.tabs.addTab(self._scroll_wrap(self._tab_speed()), _("Speed & hardware"))
        self.tabs.addTab(self._scroll_wrap(self._tab_engines()), _("Engines"))
        self.tabs.addTab(self._scroll_wrap(self._tab_logging()), _("Logging & diagnostics"))
        self.tabs.addTab(self._scroll_wrap(self._tab_about()), _("About"))

        buttons = QDialogButtonBox(compat.dialog_buttons("Save", "Cancel"))
        buttons.button(compat.dialog_buttons("Save")).setText(_("Save"))
        buttons.button(compat.dialog_buttons("Save")).setObjectName("Primary")
        buttons.button(compat.dialog_buttons("Cancel")).setText(_("Cancel"))
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)

        self.reset_button = QPushButton(_("Reset settings to defaults"))
        self.reset_button.setObjectName("Ghost")
        self.reset_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.reset_button.setToolTip(_(
            "Return ALL settings to defaults: recognition options, folders,\n"
            "window position and size, model catalog filters."
        ))
        self.reset_button.clicked.connect(self._reset_defaults)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.reset_button)
        buttons_row.addStretch(1)
        buttons_row.addWidget(buttons)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.addWidget(self.tabs, stretch=1)
        root.addLayout(buttons_row)

    # -- settings reset -------------------------------------------------------- #
    def _reset_defaults(self) -> None:
        answer = QMessageBox.question(
            self, _("Reset settings"),
            _("Return all settings to defaults?\n\n"
              "Will be reset: recognition options, results folder,\n"
              "window position and size, model catalog filters.\n"
              "Downloaded models and result files are not affected."),
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_no(),
        )
        if answer != compat.message_box_yes():
            return
        language_before = i18n.current_language()
        self.settings.reset_to_defaults()
        if self._on_reset_defaults is not None:
            try:
                self._on_reset_defaults()
            except Exception:  # noqa: BLE001
                pass
        # A reset returns ui_language to AUTO ("follow the system"). When the
        # ACTIVE language differs from the post-reset one (e.g. an explicit
        # English on a Russian system — or vice versa), the interface would
        # silently stay in the old language until a manual restart. Offer the
        # restart right away, exactly like the Save path does.
        new_pref = (self.settings.ui_language or "").strip().lower()
        new_language = new_pref if new_pref in i18n.SUPPORTED else i18n.detect_system_language()
        language_changed = new_language != language_before
        self.done(0)   # not accept(): stale widget values must not be saved
        if language_changed:
            names = dict((value, label) for label, value in i18n.language_choices())
            question = QMessageBox.question(
                self.parentWidget() or self, _("Interface language"),
                _("Settings were reset to defaults.\n"
                  "The interface language is now \"{new}\", but the running window "
                  "still shows \"{old}\".\n\n"
                  "Restart the application now to apply the language?").format(
                    new=names.get(new_language, new_language),
                    old=names.get(language_before, language_before)),
                compat.message_box_yes() | compat.message_box_no(),
                compat.message_box_yes(),
            )
            if question == compat.message_box_yes():
                self._restart_requested = True
                QTimer.singleShot(0, self._restart_application)

    # -- helpers ------------------------------------------------------------------ #
    @staticmethod
    def _scroll_wrap(page: QWidget) -> QScrollArea:
        """Tab with its own scroller: hints are not clipped in a small window."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(compat.frame_no_frame())
        scroll.setHorizontalScrollBarPolicy(compat.SCROLLBAR_AS_NEEDED)
        scroll.setWidget(page)
        return scroll

    @staticmethod
    def _hinted(form: QFormLayout, title: str, widget: QWidget, hint: str) -> None:
        """A setting row + a "why and how" hint."""
        widget.setToolTip(hint)
        form.addRow(title, widget)
        form.addRow(make_label(hint, "Muted", word_wrap=True))

    # -- tabs ------------------------------------------------------------------ #
    def _tab_interface(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(8)

        self.ui_language_combo = QComboBox()
        for label, value in i18n.language_choices():
            self.ui_language_combo.addItem(label, value)
        index = self.ui_language_combo.findData(self.settings.ui_language or "")
        self.ui_language_combo.setCurrentIndex(index if index >= 0 else 0)
        self._hinted(
            form, _("Interface language"), self.ui_language_combo,
            _("Language of menus, buttons, hints and messages. \"Detect automatically\" "
              "follows the operating system language. A change takes effect after "
              "restarting the application — you will be offered to restart on Save."),
        )
        return page

    def _tab_quality(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(8)

        self.language_combo = QComboBox()
        for code, title in LANGUAGES:
            self.language_combo.addItem(_(title), code)
        index = self.language_combo.findData(self.settings.language)
        self.language_combo.setCurrentIndex(index if index >= 0 else 1)
        self._hinted(
            form, _("Speech language"), self.language_combo,
            _("Language of the recording. \"Detect automatically\" is handy for mixed "
              "folders, but an explicit choice is faster and more reliable on "
              "short/noisy files."),
        )

        self.translate_check = QCheckBox(_("Translate the result to English (task=translate)"))
        self.translate_check.setChecked(self.settings.translate_to_english)
        self._hinted(
            form, "", self.translate_check,
            _("The model outputs an English translation of what it heard instead of the "
              "original text. Keep it off for notes of a Russian lecture."),
        )

        self.vad_check = QCheckBox(_("Cut silence and noise (Silero VAD)"))
        self.vad_check.setChecked(self.settings.use_vad)
        self._hinted(
            form, "", self.vad_check,
            _("Cuts pauses/music/noise before recognition: faster and fewer hallucinations. "
              "Turn off only if the silence itself is meaningful."),
        )

        self.anti_hallucination = QCheckBox(_('Clean "[music]", "Thanks for watching", repeats and loops'))
        self.anti_hallucination.setChecked(self.settings.suppress_hallucinations)
        self._hinted(
            form, "", self.anti_hallucination,
            _("Post-processing: removes junk segments Whisper invents on silence and "
              "music, and glued repeats of one phrase. Recommended always on."),
        )

        self.paragraphs_check = QCheckBox(_("Join phrases into paragraphs"))
        self.paragraphs_check.setChecked(self.settings.paragraphs)
        self._hinted(
            form, "", self.paragraphs_check,
            _("In .txt the text becomes paragraphs split by meaningful pauses instead "
              "of one long line. Turn off if every phrase should start a new line."),
        )

        self.word_ts_check = QCheckBox(_("Compute per-word timestamps"))
        self.word_ts_check.setChecked(self.settings.word_timestamps)
        self._hinted(
            form, "", self.word_ts_check,
            _("Only needed for frame-accurate subtitles; slows recognition down. "
              "Turn off for .txt notes."),
        )

        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 12)
        self.beam_spin.setValue(int(self.settings.beam_size or 5))
        self._hinted(
            form, _("Beam size (decoding quality)"), self.beam_spin,
            _("How many phrase continuations the decoder keeps: 1 — fastest, "
              "5 — optimal, 8+ — maximum accuracy and time. 5 is enough for lectures."),
        )

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlainText(self.settings.initial_prompt)
        self.prompt_edit.setFixedHeight(70)
        self.prompt_edit.setPlaceholderText(_(
            "Model hint: style and terms. Example:\n"
            "Machine learning lecture. Terms: gradient descent, transformer, attention, tokenization."
        ))
        self._hinted(
            form, _("initial_prompt (style/terms)"), self.prompt_edit,
            _("A short style/topic example the model \"continues\". Helps with terminology "
              "and formatting (e.g. \"write in a scientific style, capitalize names\")."),
        )

        self.hotwords_edit = QPlainTextEdit()
        self.hotwords_edit.setPlainText(self.settings.hotwords)
        self.hotwords_edit.setFixedHeight(60)
        self.hotwords_edit.setPlaceholderText(_("Comma-separated hint words: Huffman, RAG, Kubernetes…"))
        self._hinted(
            form, _("hotwords (hint words)"), self.hotwords_edit,
            _("Rare words and terms of the concrete lecture — the model will spell them "
              "correctly. Works in faster-whisper; other engines ignore the hint."),
        )

        return page

    def _tab_speed(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(8)

        self.device_combo = QComboBox()
        self.device_combo.addItem(_("Auto (recommended)"), "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem(_("NVIDIA GPU (CUDA)"), "cuda")
        index = self.device_combo.findData(self.settings.device)
        self.device_combo.setCurrentIndex(index if index >= 0 else 0)
        self._hinted(
            form, _("Device"), self.device_combo,
            _("Where to compute: Auto picks the GPU when CUDA is visible, otherwise CPU. "
              "Forced CPU helps when the GPU is busy with games/rendering. "
              "If you pick a GPU without CUDA libraries installed, the app offers "
              "the installation (Engines tab) and runs on CPU meanwhile."),
        )

        self.compute_combo = QComboBox()
        for value, title in (
            ("auto", _("Auto")),
            ("int8", _("int8 — CPU, minimum memory")),
            ("int8_float16", _("int8_float16 — GPU with small VRAM")),
            ("float16", _("float16 — GPU, fastest")),
            ("default", _("default — whatever the engine decides")),
        ):
            self.compute_combo.addItem(title, value)
        index = self.compute_combo.findData(self.settings.compute_type)
        self.compute_combo.setCurrentIndex(index if index >= 0 else 0)
        self._hinted(
            form, _("Compute precision"), self.compute_combo,
            _("The numeric format of model weights in memory: int8 saves RAM/VRAM at the "
              "cost of ~0.1–0.3% accuracy, float16 is fastest on GPU. Auto picks per device."),
        )

        self.batched_check = QCheckBox(_("Batched decoding (BatchedInferencePipeline)"))
        self.batched_check.setChecked(bool(self.settings.batched))
        self._hinted(
            form, "", self.batched_check,
            _("Up to 4× faster on GPU: audio chunks are computed as a batch. Almost no "
              "gain on CPU and more memory needed — turn off when RAM is tight."),
        )

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(int(self.settings.batch_size or 8))
        self._hinted(
            form, _("Batch size"), self.batch_spin,
            _("How many audio chunks go into one batch. More — faster on GPU but more VRAM. "
              "On \"out of memory\" errors reduce to 4/2."),
        )

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setSpecialValueText(_("auto ({})").format(self.hardware.get("recommended_threads", "?")))
        self.threads_spin.setValue(int(self.settings.cpu_threads or 0))
        self._hinted(
            form, _("CPU threads"), self.threads_spin,
            _("How many CPU threads inference gets. 0 (auto) is usually best; limit it "
              "if you keep working on the machine and it feels slow."),
        )

        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 8)
        self.parallel_spin.setValue(int(self.settings.parallel_files or 1))
        self._hinted(
            form, _("Files at once"), self.parallel_spin,
            _("1 — predictable and memory-friendly. 2–4 — faster on multi-core CPU/GPU "
              "with plenty of RAM: each model copy consumes memory in full."),
        )

        note = make_label(
            _("The result is always saved as .txt (file name = original name). "
              "The bottleneck should be the hardware, not Python: audio decoding runs in a "
              "separate thread ahead of time (prefetch), the model loads once per queue, and "
              "inference itself runs in C++/CTranslate2 without the GIL."),
            "Muted",
            word_wrap=True,
        )
        form.addRow(note)
        return page

    def _tab_engines(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        from ..asr.base import engine_install_size, engine_note
        from ..asr.registry import engine_available

        layout.addWidget(
            make_label(
                _("An engine is the library the app recognizes speech with. "
                  "Nothing needs pre-installing: a missing engine installs itself when "
                  "you start a suitable model (the app asks for confirmation and shows "
                  "the download size)."),
                "Muted",
                word_wrap=True,
            )
        )

        for engine in ENGINES:
            box = QGroupBox(_(ENGINE_LABELS[engine]))
            box.setToolTip(_("Engine status in the project environment"))
            box_layout = QVBoxLayout(box)
            installed = engine_available(engine)
            status = make_label(
                _("✓ engine installed") if installed
                else _("○ not installed — installs automatically when needed"),
                "BadgeInstalled" if installed else "Muted",
            )
            status.setToolTip(_(
                "Checked by the presence of the engine's pip packages in the project env.\n"
                "A missing engine is offered for download when a model is started."
            ))
            status.setWordWrap(True)
            box_layout.addWidget(status)
            note = _(engine_note(engine)) if engine_note(engine) else ""
            if note:
                box_layout.addWidget(make_label(note, "Muted", word_wrap=True))
            size_text = _(engine_install_size(engine)) if engine_install_size(engine) else ""
            if size_text and not installed:
                box_layout.addWidget(
                    make_label(_("Install size: {size}").format(size=size_text), "Dim", word_wrap=True)
                )
            layout.addWidget(box)

        # CUDA for faster-whisper: the install button appears ONLY when it is
        # actually needed (an NVIDIA card exists but the libraries do not).
        cuda_box = QGroupBox(_("NVIDIA CUDA acceleration (for faster-whisper)"))
        cuda_layout = QVBoxLayout(cuda_box)
        cuda_ok = bool(self.hardware.get("cuda_ok"))
        has_nvidia = bool(self.hardware.get("has_nvidia"))
        if cuda_ok:
            cuda_layout.addWidget(
                make_label(_("✓ CUDA is available — Whisper models can run on the GPU"),
                           "BadgeInstalled", word_wrap=True)
            )
        elif not has_nvidia:
            cuda_layout.addWidget(
                make_label(_("○ No NVIDIA GPU detected — this section is not needed"),
                           "Muted", word_wrap=True)
            )
        else:
            cuda_layout.addWidget(
                make_label(
                    _("○ An NVIDIA GPU is present, but the CUDA libraries (cuBLAS/cuDNN 9) "
                      "are not installed — recognition runs on the CPU"),
                    "BadgeWarn",
                    word_wrap=True,
                )
            )
            cuda_layout.addWidget(
                make_label(
                    _("Install: ~1 GB (nvidia-cublas-cu12 + nvidia-cudnn-cu12, one time).\n"
                      "Windows: the system cuBLAS/cuDNN 9 DLLs are additionally required — "
                      "download them from github.com/Purfview/whisper-standalone-win (libs "
                      "release) and put them into the environment's Lib\\site-packages."),
                    "Muted",
                    word_wrap=True,
                )
            )
            install_cuda = QPushButton(_("Install CUDA libraries (~1 GB)"))
            install_cuda.setObjectName("Primary")
            install_cuda.setCursor(compat.CURSOR_POINTING_HAND)
            install_cuda.setToolTip(_(
                "Downloads cuBLAS/cuDNN into the project environment (once, ~1 GB).\n"
                "After that Whisper models can run on the GPU."
            ))
            install_cuda.clicked.connect(
                lambda: self._install_engine(
                    "cuda", ["-r", str(paths.ROOT_DIR / "requirements" / "gpu.txt")]
                )
            )
            cuda_layout.addWidget(install_cuda, alignment=compat.ALIGN_LEFT)
        layout.addWidget(cuda_box)

        layout.addStretch(1)
        return page

    def _install_engine(self, engine: str, packages: List[str]) -> None:
        window = self.parent()
        if window is not None and hasattr(window, "install_packages_interactive"):
            self.accept()
            window.install_packages_interactive(packages, _("Installing: {engine}").format(engine=engine))
        else:
            QMessageBox.information(self, _("Engines"), _("Packages can be installed from the main window."))

    # -- logging tab (file logs) ------------------------------------------- #
    def _tab_logging(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        from .. import logutils

        layout.addWidget(make_label(_("Verbose log file"), "H2"))
        layout.addWidget(
            make_label(
                _("By default the app writes NO log files — logs/ stays empty.\n"
                  "When logging is enabled:\n"
                  "  • each launch creates a SEPARATE file logs/localasr_YYYY-MM-DD_HH-MM-SS.log;\n"
                  "  • it contains everything that happens: startup options, hardware,\n"
                  "    network requests, processing stages, timings, errors with full\n"
                  "    tracebacks — enough to reproduce and fix any bug;\n"
                  "  • old files are pruned automatically (the last 30 are kept).\n\n"
                  "The switch applies IMMEDIATELY — no restart needed."),
                "Dim",
                word_wrap=True,
            )
        )

        self.file_logging_check = QCheckBox(_("Keep a verbose log file (logs/)"))
        self.file_logging_check.setChecked(bool(logutils.file_logging_enabled() or self.settings.file_logging))
        self.file_logging_check.setToolTip(_(
            "Enable/disable verbose logging. Applies instantly:\n"
            "enabling creates the file in logs/ right away, disabling stops the writes."
        ))
        self.file_logging_check.toggled.connect(self._on_file_logging_toggled)
        layout.addWidget(self.file_logging_check)

        self.log_path_label = make_label("", "Muted", word_wrap=True)
        self.log_path_label.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
        layout.addWidget(self.log_path_label)

        buttons = QHBoxLayout()
        open_logs = QPushButton(_("Open the logs folder"))
        open_logs.setObjectName("Ghost")
        open_logs.setCursor(compat.CURSOR_POINTING_HAND)
        open_logs.setToolTip(_("Show logs/ in the file manager"))
        open_logs.clicked.connect(lambda: self._open_logs_folder())
        buttons.addWidget(open_logs)
        copy_path = QPushButton(_("Copy the current log path"))
        copy_path.setObjectName("Ghost")
        copy_path.setCursor(compat.CURSOR_POINTING_HAND)
        copy_path.setToolTip(_("Copy the active log file path to the clipboard (handy for issues)"))
        copy_path.clicked.connect(self._copy_log_path)
        buttons.addWidget(copy_path)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        layout.addWidget(
            make_label(
                _("Tip: if something went wrong — enable logging, reproduce the problem\n"
                  "and attach the fresh file from logs/ to the report (Help → \"Report an issue\").\n"
                  "The event journal in the main window is always shown, regardless of this setting."),
                "Muted",
                word_wrap=True,
            )
        )
        layout.addStretch(1)
        self._refresh_log_path_label()
        return page

    def _on_file_logging_toggled(self, checked: bool) -> None:
        """Apply the log switch IMMEDIATELY (no Save/restart)."""
        from .. import logutils

        path = logutils.set_file_logging(bool(checked))
        self.settings.file_logging = bool(checked)
        self.settings.save()
        self._refresh_log_path_label()
        if checked and path:
            self.log_path_label.setText(_("Writing the log: {path}").format(path=path))

    def _refresh_log_path_label(self) -> None:
        from .. import logutils

        current = logutils.current_log_file()
        if current:
            self.log_path_label.setText(_("Current log file: {path}").format(path=current))
        else:
            files = logutils.list_log_files()
            if files:
                self.log_path_label.setText(_("Logging is off. Last log: {path}").format(path=files[0]))
            else:
                self.log_path_label.setText(_("Logging is off — no files in logs/."))

    def _open_logs_folder(self) -> None:
        from .widgets import open_in_file_manager

        open_in_file_manager(paths.logs_dir())

    def _copy_log_path(self) -> None:
        from .. import logutils

        current = logutils.current_log_file()
        files = logutils.list_log_files()
        target = current or (files[0] if files else None)
        if target is None:
            QMessageBox.information(self, _("Log"), _("No log files yet — enable logging."))
            return
        try:
            QApplication.clipboard().setText(str(target))
        except Exception:  # noqa: BLE001
            pass

    def _tab_about(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        from ..constants import APP_NAME, APP_VERSION, REPO_URL

        layout.addWidget(make_label(f"{APP_NAME} {APP_VERSION}", "H1"))
        layout.addWidget(
            make_label(
                _("Local speech recognition: lectures, videos, podcasts — no cloud and no "
                  "data ever leaves the machine. Video is converted to audio automatically.\n\n"
                  "Engines: faster-whisper (CTranslate2), whisper.cpp (GGML), Vosk (Kaldi),\n"
                  "Qwen3-ASR and GigaAM (torch), hf-transformers (any HF ASR model).\n"
                  "Everything is installed strictly inside the project folder."),
                "Dim",
                word_wrap=True,
            )
        )

        for text, url in (
            (_("Repository and documentation"), REPO_URL),
            ("faster-whisper", "https://github.com/SYSTRAN/faster-whisper"),
            ("whisper.cpp", "https://github.com/ggml-org/whisper.cpp"),
            (_("Vosk and models"), "https://alphacephei.com/vosk/models"),
            (_("Hugging Face model catalog"), "https://huggingface.co/models?pipeline_tag=automatic-speech-recognition"),
        ):
            button = QPushButton(f"{text} ↗")
            button.setObjectName("Link")
            button.setCursor(compat.CURSOR_POINTING_HAND)
            button.clicked.connect(lambda _checked=False, link=url: open_url(link))
            layout.addWidget(button, alignment=compat.ALIGN_LEFT)

        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------ #
    def _save_and_accept(self) -> None:
        new_ui_language = self.ui_language_combo.currentData() or ""
        language_changed = new_ui_language != (self.settings.ui_language or "")

        self.settings.ui_language = new_ui_language
        self.settings.language = self.language_combo.currentData() or ""
        self.settings.translate_to_english = self.translate_check.isChecked()
        self.settings.use_vad = self.vad_check.isChecked()
        self.settings.suppress_hallucinations = self.anti_hallucination.isChecked()
        self.settings.paragraphs = self.paragraphs_check.isChecked()
        self.settings.word_timestamps = self.word_ts_check.isChecked()
        self.settings.beam_size = self.beam_spin.value()
        self.settings.initial_prompt = self.prompt_edit.toPlainText().strip()
        self.settings.hotwords = self.hotwords_edit.toPlainText().strip()
        self.settings.device = self.device_combo.currentData() or "auto"
        self.settings.compute_type = self.compute_combo.currentData() or "auto"
        self.settings.batched = self.batched_check.isChecked()
        self.settings.batch_size = self.batch_spin.value()
        self.settings.cpu_threads = self.threads_spin.value()
        self.settings.parallel_files = self.parallel_spin.value()

        restart_now = False
        if language_changed:
            answer = QMessageBox.question(
                self, _("Interface language"),
                _("The new interface language takes effect after a restart.\n"
                  "Restart the application now?"),
                compat.message_box_yes() | compat.message_box_no(),
                compat.message_box_yes(),
            )
            restart_now = answer == compat.message_box_yes()

        self.settings.save()
        self.accept()

        if restart_now:
            self._restart_requested = True
            QTimer.singleShot(0, self._restart_application)

    def _restart_application(self) -> None:
        from ..core.relaunch import restart_app

        if restart_app():
            app = QApplication.instance()
            if app is not None:
                app.quit()
        else:
            QMessageBox.information(
                self, _("Interface language"),
                _("Automatic restart failed. Please close the application and start it again."),
            )


# --------------------------------------------------------------------------- #
#  Model information
# --------------------------------------------------------------------------- #


class ModelInfoDialog(QDialog):
    """Full technical model information (the "ℹ" button on a card)."""

    def __init__(self, entry: ModelEntry, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(entry.display_name)
        self.setModal(True)
        self.setMinimumSize(640, 520)
        self.resize(720, 620)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(compat.frame_no_frame())
        scroll.setHorizontalScrollBarPolicy(compat.SCROLLBAR_AS_NEEDED)
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 8, 0)
        root.setSpacing(10)

        root.addWidget(make_label(entry.display_name, "H1"))

        rows = [
            (_("ID (folder in models/)"), entry.id),
            (_("Engine"), _(entry.engine_label)),
            (_("Vendor"), _(entry.vendor) if entry.vendor else "—"),
            (_("Source"), entry.source or "—"),
            (_("Subfolder"), entry.subfolder or "—"),
            (_("Parameter count"), entry.params_text),
            (_("Quantization (weight format)"), entry.quantization_text),
            (_("Release year"), entry.released_year_text),
            (_("Download size"), entry.size_text),
            (_("Disk usage"), paths.human_size(entry.installed_size_bytes) if entry.installed else "—"),
            (_("Memory in use"), entry.memory_text),
            (_("Languages"), entry.languages_text),
            (_("Accuracy / speed"), f"{stars(entry.quality)} / {stars(entry.speed)}"),
            (_("Tags"), ", ".join(_(tag) for tag in entry.tags) if entry.tags else "—"),
            (_("License"), entry.license or "—"),
            (_("Installed"), _("yes: {}").format(entry.install_dir) if entry.installed else _("no")),
        ]
        table = QFormLayout()
        table.setSpacing(6)
        for key, value in rows:
            label = make_label(str(value), "Dim", word_wrap=True)
            label.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
            table.addRow(make_label(key, "Muted"), label)
        root.addLayout(table)

        if entry.recommended_for:
            hint = make_label(_("Best for: ") + _(entry.recommended_for), "Dim", word_wrap=True)
            root.addWidget(hint)

        description = make_label(_(entry.description or ""), "Dim", word_wrap=True)
        description.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
        root.addWidget(description)

        if entry.defaults:
            root.addWidget(make_label(_("Default parameters: ") + str(entry.defaults), "Muted", word_wrap=True))

        root.addStretch(1)
        scroll.setWidget(page)
        outer.addWidget(scroll, stretch=1)

        buttons = QHBoxLayout()
        if entry.url:
            link = QPushButton(_("Open the model page ↗"))
            link.setObjectName("Ghost")
            link.setCursor(compat.CURSOR_POINTING_HAND)
            link.setToolTip(_("Opens the model page in the browser (Hugging Face / author site)"))
            link.clicked.connect(lambda: open_url(entry.url))
            buttons.addWidget(link)
        buttons.addStretch(1)
        close = QPushButton(_("Close"))
        close.setObjectName("Primary")
        close.setCursor(compat.CURSOR_POINTING_HAND)
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        outer.addLayout(buttons)


# --------------------------------------------------------------------------- #
#  Diagnostics and recovery (program integrity)
# --------------------------------------------------------------------------- #


class DiagnosticsDialog(QDialog):
    """Tools → Diagnostics & recovery.

    Checks three levels of program resilience:
      1. project files (accidentally deleted — restored from GitHub);
      2. working folders (models/, outputs/… — created automatically);
      3. the environment (missing libraries — repaired; extra packages —
         offered for deletion to free space).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Diagnostics & recovery"))
        self.setModal(True)
        self.setMinimumSize(700, 640)
        self.report: dict = {}
        self.audit_worker = None
        self.restore_worker = None
        self.package_worker = None
        self.package_dialog = None
        self._build_ui()
        self._start_audit()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(10)

        outer.addWidget(make_label(_("Program state"), "H1"))
        outer.addWidget(
            make_label(
                _("Checks project files, working folders and environment libraries. "
                  "Nothing breaks by itself here: the app only reports problems and "
                  "offers one-click fixes."),
                "Dim",
                word_wrap=True,
            )
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(compat.frame_no_frame())
        scroll.setHorizontalScrollBarPolicy(compat.SCROLLBAR_AS_NEEDED)
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 8, 0)
        root.setSpacing(12)

        # --- 1. project files
        files_box = QGroupBox(_("Program files"))
        files_layout = QVBoxLayout(files_box)
        self.files_status = make_label(_("Checking…"), "Muted", word_wrap=True)
        files_layout.addWidget(self.files_status)
        self.files_list = make_label("", "Dim", word_wrap=True)
        self.files_list.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
        self.files_list.setVisible(False)
        files_layout.addWidget(self.files_list)
        self.restore_button = QPushButton(_("Restore missing files from GitHub"))
        self.restore_button.setObjectName("Primary")
        self.restore_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.restore_button.setToolTip(_(
            "Re-downloads deleted program files (a few hundred kilobytes).\n"
            "Models, results and settings are not affected."
        ))
        self.restore_button.setVisible(False)
        self.restore_button.clicked.connect(self._restore_files)
        files_layout.addWidget(self.restore_button, alignment=compat.ALIGN_LEFT)
        root.addWidget(files_box)

        # --- 2. folders
        folders_box = QGroupBox(_("Working folders"))
        folders_layout = QVBoxLayout(folders_box)
        self.folders_status = make_label(_("Checking…"), "Muted", word_wrap=True)
        folders_layout.addWidget(self.folders_status)
        folders_layout.addWidget(
            make_label(
                _("These folders are created automatically when needed. Their absence "
                  "is not an error: e.g. models/ appears on the first model download."),
                "Dim",
                word_wrap=True,
            )
        )
        root.addWidget(folders_box)

        # --- 3. environment
        venv_box = QGroupBox(_("Environment"))
        venv_layout = QVBoxLayout(venv_box)
        self.venv_status = make_label(_("Checking…"), "Muted", word_wrap=True)
        self.venv_status.setWordWrap(True)
        venv_layout.addWidget(self.venv_status)
        self.repair_button = QPushButton(_("Repair the environment (install what is missing)"))
        self.repair_button.setObjectName("Primary")
        self.repair_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.repair_button.setToolTip(_(
            "Re-checks the base libraries in the project environment and installs missing ones.\n"
            "The system Python is not affected."
        ))
        self.repair_button.setVisible(False)
        self.repair_button.clicked.connect(self._repair_env)
        venv_layout.addWidget(self.repair_button, alignment=compat.ALIGN_LEFT)

        self.extra_hint = make_label("", "Muted", word_wrap=True)
        self.extra_hint.setVisible(False)
        venv_layout.addWidget(self.extra_hint)
        self.extra_list = QListWidget()
        self.extra_list.setVisible(False)
        self.extra_list.setMaximumHeight(180)
        self.extra_list.setToolTip(_(
            "Packages required neither by the app nor by other installed packages.\n"
            "Check the ones to delete. If you installed them yourself for other\n"
            "tasks — leave them."
        ))
        self.extra_list.itemChanged.connect(self._update_delete_button)
        venv_layout.addWidget(self.extra_list)
        self.delete_button = QPushButton(_("Delete selected packages"))
        self.delete_button.setObjectName("Danger")
        self.delete_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.delete_button.setVisible(False)
        self.delete_button.clicked.connect(self._delete_extra)
        venv_layout.addWidget(self.delete_button, alignment=compat.ALIGN_LEFT)
        root.addWidget(venv_box)

        root.addStretch(1)
        scroll.setWidget(page)
        outer.addWidget(scroll, stretch=1)

        buttons = QHBoxLayout()
        self.refresh_button = QPushButton(_("⟳ Re-check"))
        self.refresh_button.setObjectName("Ghost")
        self.refresh_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.refresh_button.setToolTip(_("Run the check again"))
        self.refresh_button.clicked.connect(self._start_audit)
        buttons.addWidget(self.refresh_button)
        buttons.addStretch(1)
        close = QPushButton(_("Close"))
        close.setObjectName("Primary")
        close.setCursor(compat.CURSOR_POINTING_HAND)
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        outer.addLayout(buttons)

    # ------------------------------------------------------------------ #
    def done(self, result: int) -> None:
        """Silence all own threads before closing (X/Esc/Close)."""
        from .workers import stop_worker

        stop_worker(self.audit_worker, 3000, "EnvAuditWorker")
        stop_worker(self.restore_worker, 3000, "FileRestoreWorker")
        stop_worker(self.package_worker, 3000, "PackageInstaller")
        self.audit_worker = None
        self.restore_worker = None
        self.package_worker = None
        super().done(result)

    def _start_audit(self) -> None:
        if self.audit_worker is not None and self.audit_worker.isRunning():
            return
        from .workers import EnvAuditWorker

        self.refresh_button.setEnabled(False)
        self.files_status.setText(_("Checking program files…"))
        self.venv_status.setText(_("Checking the environment… (pip list takes a few seconds)"))
        worker = EnvAuditWorker(self)
        worker.finishedWith.connect(self._on_audit)
        self.audit_worker = worker
        worker.start()

    def _on_audit(self, report) -> None:
        worker, self.audit_worker = self.audit_worker, None
        if worker is not None:
            worker.deleteLater()
        self.refresh_button.setEnabled(True)
        self.report = dict(report or {})

        if self.report.get("error"):
            self.files_status.setText(_("Check could not finish: {err}").format(err=self.report["error"]))
            return

        # --- files
        missing = list(self.report.get("missing_files") or [])
        if missing:
            self.files_status.setText(_(
                "⚠ Missing files: {n}. The app may work incompletely — restore them "
                "(internet required, ~hundreds of KB)."
            ).format(n=len(missing)))
            self.files_list.setText("  • " + "\n  • ".join(missing[:30]))
            self.files_list.setVisible(True)
            self.restore_button.setVisible(True)
        else:
            self.files_status.setText(_("✓ All program files are present"))
            self.files_list.setVisible(False)
            self.restore_button.setVisible(False)

        # --- folders
        folders = list(self.report.get("folders") or [])
        lines = [
            _("• {name} — {status} ({purpose})").format(
                name=item["name"], status=_(item["status"]), purpose=_(item["purpose"]))
            for item in folders
        ]
        self.folders_status.setText("\n".join(lines) or "—")

        # --- environment
        base_missing = list(self.report.get("base_missing") or [])
        python = str(self.report.get("python") or "")
        if base_missing:
            self.venv_status.setText(_(
                "⚠ Missing base libraries: {list}\nInterpreter: {py}"
            ).format(list=", ".join(base_missing), py=python))
            self.repair_button.setVisible(True)
        else:
            self.venv_status.setText(_("✓ All base libraries are installed\nInterpreter: {py}").format(py=python))
            self.repair_button.setVisible(False)

        # --- extra packages
        extra = list(self.report.get("extra_packages") or [])
        self.extra_list.clear()
        if extra:
            from ..paths import human_size

            total = sum(int(item.get("size_bytes") or 0) for item in extra)
            self.extra_hint.setText(_(
                "Packages the app does not need (required neither by it nor by other "
                "packages): {n} item(s), {size} total. Deleting is optional — it only saves space."
            ).format(n=len(extra), size=human_size(total)))
            self.extra_hint.setVisible(True)
            self.extra_list.setVisible(True)
            self.delete_button.setVisible(True)
            for item in extra:
                size = human_size(int(item.get("size_bytes") or 0))
                entry = QListWidgetItem(
                    f"{item.get('name')} {item.get('version') or ''} · {size}".strip()
                )
                entry.setData(compat.ROLE_USER, str(item.get("name")))
                entry.setFlags(entry.flags() | compat.ITEM_USER_CHECKABLE)
                entry.setCheckState(compat.CHECK_UNCHECKED)
                self.extra_list.addItem(entry)
            self._update_delete_button()
        else:
            self.extra_hint.setText(_("No extra packages found in the environment."))
            self.extra_hint.setVisible(True)
            self.extra_list.setVisible(False)
            self.delete_button.setVisible(False)

    def _selected_extra(self) -> List[str]:
        names = []
        for index in range(self.extra_list.count()):
            item = self.extra_list.item(index)
            if item.checkState() == compat.CHECK_CHECKED:
                data = item.data(compat.ROLE_USER)
                if data:
                    names.append(str(data))
        return names

    def _update_delete_button(self, *_args) -> None:
        from ..paths import human_size

        names = self._selected_extra()
        total = 0
        for item in self.report.get("extra_packages") or []:
            if item.get("name") in names:
                total += int(item.get("size_bytes") or 0)
        if names:
            self.delete_button.setText(_("Delete selected ({n} item(s), ~{size})").format(
                n=len(names), size=human_size(total)))
            self.delete_button.setEnabled(True)
        else:
            self.delete_button.setText(_("Delete selected packages"))
            self.delete_button.setEnabled(False)

    # ------------------------------------------------------------------ #
    def _restore_files(self) -> None:
        missing = list(self.report.get("missing_files") or [])
        if not missing:
            return
        if self.restore_worker is not None and self.restore_worker.isRunning():
            return
        from .workers import FileRestoreWorker

        dialog = ProgressDialog(
            _("Restoring files"),
            _("Downloading {n} file(s) from the project repository…").format(n=len(missing)),
            parent=self,
        )
        worker = FileRestoreWorker(missing, self)
        worker.progress.connect(dialog.set_fraction)
        worker.progress.connect(lambda fraction, status: dialog.set_status(status))
        worker.outputLine.connect(dialog.append_log)
        dialog.cancelRequested.connect(worker.cancel)
        worker.finishedWith.connect(lambda ok, message: self._on_restore_done(ok, message, dialog))
        self.restore_worker = worker
        worker.start()
        dialog.exec()

    def _on_restore_done(self, ok: bool, message: str, dialog: ProgressDialog) -> None:
        dialog.finish(_("Done") if ok else _("Stopped"))
        dialog.accept()
        worker, self.restore_worker = self.restore_worker, None
        if worker is not None:
            worker.deleteLater()
        if ok:
            QMessageBox.information(self, _("Recovery"), message)
        else:
            QMessageBox.warning(self, _("Recovery"), message)
        self._start_audit()

    def _repair_env(self) -> None:
        window = self.parent()
        requirements = paths.ROOT_DIR / "requirements" / "base.txt"
        if not requirements.is_file():
            QMessageBox.warning(
                self, _("Environment repair"),
                _("The file {req} is missing.\nFirst restore the program files (button above).").format(req=requirements),
            )
            return
        if window is not None and hasattr(window, "install_packages_interactive"):
            self.accept()
            window.install_packages_interactive(["-r", str(requirements)], _("Environment repair"))
        else:
            QMessageBox.information(self, _("Environment repair"), _("Run the repair from the main window."))

    def _delete_extra(self) -> None:
        names = self._selected_extra()
        if not names:
            return
        answer = QMessageBox.question(
            self, _("Deleting packages"),
            _("Delete the selected packages from the environment?\n\n  • ")
            + "\n  • ".join(names[:20])
            + ("\n  …" if len(names) > 20 else "")
            + _("\n\nThe app does not need them; the system Python is not affected."),
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_no(),
        )
        if answer != compat.message_box_yes():
            return
        if self.package_worker is not None and self.package_worker.isRunning():
            return
        from .workers import PackageInstaller

        dialog = ProgressDialog(_("Deleting packages"), ", ".join(names), parent=self)
        worker = PackageInstaller(names, _("Deleting packages"), self, mode="uninstall")
        worker.outputLine.connect(dialog.append_log)
        worker.progress.connect(lambda fraction, status: dialog.set_fraction(fraction))
        worker.progress.connect(lambda fraction, status: dialog.set_status(status))
        dialog.cancelRequested.connect(worker.cancel)
        worker.finishedWith.connect(lambda ok, message: self._on_delete_done(ok, message, dialog))
        self.package_worker = worker
        self.package_dialog = dialog
        worker.start()
        dialog.exec()

    def _on_delete_done(self, ok: bool, message: str, dialog: ProgressDialog) -> None:
        dialog.finish(_("Done") if ok else _("Stopped"))
        dialog.accept()
        worker, self.package_worker = self.package_worker, None
        if worker is not None:
            worker.deleteLater()
        self.package_dialog = None
        if not ok:
            QMessageBox.warning(self, _("Deletion"), message)
        self._start_audit()
