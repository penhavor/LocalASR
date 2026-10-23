"""Диалоги: прогресс, добавление своей модели, первый запуск, настройки, железо."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

from .. import paths
from ..constants import ENGINE_LABELS, ENGINES, LANGUAGES
from ..core.models_catalog import ModelEntry
from ..settings import Settings
from . import compat
from .widgets import FlowLayout, make_label, open_url, stars


# --------------------------------------------------------------------------- #
#  Прогресс с возможностью отмены
# --------------------------------------------------------------------------- #


class ProgressDialog(QDialog):
    """Универсальное окно прогресса (скачивание модели, установка пакетов, работа)."""

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

        self.status_label = make_label("Подготовка…", "Dim", word_wrap=True)
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
        self.cancel_button = QPushButton("Отмена")
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
        self.cancel_button.setText("Останавливаю…")
        self.status_label.setText("Отмена — дожидаемся завершения текущей операции…")
        self.cancelRequested.emit()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def set_status(self, text: object) -> None:
        # Защита от «плавания» типов: сигналы воркеров могут принести что угодно,
        # а QLabel.setText(float) ронял приложение TypeError'ом на каждом тике.
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

    def finish(self, message: str = "Готово") -> None:
        self._done = True
        self.set_fraction(1.0)
        self.status_label.setText(message)
        self.cancel_button.setText("Закрыть")
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
        """Крестик = отмена операции. После отмены/завершения окно закрывается."""
        if self._done or self._cancelled:
            event.accept()
            return
        self.cancel()
        event.ignore()


# --------------------------------------------------------------------------- #
#  Своя модель по ссылке
# --------------------------------------------------------------------------- #


class CustomModelDialog(QDialog):
    """Добавление любой модели: HF-репозиторий, прямая ссылка или папка на диске."""

    def __init__(self, parent: Optional[QWidget] = None, models_root: Optional[Path] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Своя модель")
        self.setModal(True)
        self.setMinimumWidth(680)
        self.entry: Optional[ModelEntry] = None
        self._models_root = Path(models_root) if models_root else paths.models_dir()

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        root.addWidget(
            make_label(
                "Можно добавить любую модель, а не только те, что в каталоге.",
                "H2",
            )
        )
        root.addWidget(
            make_label(
                "Поддерживаются три вида источников:\n"
                "  • репозиторий Hugging Face — например, coriollon/whisper-large-v3-turbo-russian\n"
                "    (можно с подпапкой: owner/name/tree/main/ct2_int8_float16)\n"
                "  • прямая ссылка на файл — ggml-*.bin, *.zip (Vosk), model.bin\n"
                "  • папка или файл на диске — например, D:\\models\\my-whisper\n\n"
                "Движок определим автоматически по содержимому.",
                "Dim",
                word_wrap=True,
            )
        )

        form = QFormLayout()
        form.setSpacing(10)

        source_row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("owner/name, https://… или C:\\путь\\к\\модели")
        source_row.addWidget(self.source_edit, stretch=1)

        self.browse_button = QPushButton("Обзор…")
        self.browse_button.setObjectName("Ghost")
        self.browse_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.browse_button.clicked.connect(self._browse)
        source_row.addWidget(self.browse_button)
        form.addRow("Источник", source_row)

        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("оставь пустым — придумаем сами (это имя папки в models/)")
        form.addRow("ID модели", self.id_edit)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("название, которое увидишь в списке")
        form.addRow("Название", self.name_edit)

        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Определить автоматически", "")
        for engine in ENGINES:
            self.engine_combo.addItem(ENGINE_LABELS[engine], engine)
        form.addRow("Движок", self.engine_combo)

        self.languages_edit = QLineEdit()
        self.languages_edit.setPlaceholderText("ru, en (через запятую)")
        self.languages_edit.setText("ru, en")
        form.addRow("Языки", self.languages_edit)

        self.ram_spin = QSpinBox()
        self.ram_spin.setRange(0, 200_000)
        self.ram_spin.setSuffix(" МБ")
        self.ram_spin.setSpecialValueText("оценить автоматически")
        form.addRow("Память в работе", self.ram_spin)

        root.addLayout(form)

        self.inspect_button = QPushButton("🔍 Определить, что это за модель")
        self.inspect_button.setObjectName("Ghost")
        self.inspect_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.inspect_button.clicked.connect(self._inspect)
        root.addWidget(self.inspect_button)

        self.result_label = make_label("", "Dim", word_wrap=True)
        self.result_label.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
        root.addWidget(self.result_label)

        buttons = QDialogButtonBox(compat.dialog_buttons("Ok", "Cancel"))
        buttons.button(compat.dialog_buttons("Ok")).setText("Добавить и скачать")
        buttons.button(compat.dialog_buttons("Ok")).setObjectName("Primary")
        buttons.button(compat.dialog_buttons("Cancel")).setText("Отмена")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------ #
    def _browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        picked = QFileDialog.getExistingDirectory(self, "Папка с моделью", str(self._models_root))
        if picked:
            self.source_edit.setText(picked)
            self._inspect()

    def _inspect(self) -> None:
        source = self.source_edit.text().strip()
        if not source:
            self.result_label.setText("Сначала укажи источник.")
            return
        self.inspect_button.setEnabled(False)
        self.result_label.setText("Проверяю источник…")
        try:
            from ..core.downloader import inspect_custom_source

            entry = inspect_custom_source(source)
        except Exception as exc:  # noqa: BLE001
            self.result_label.setText(f"Не получилось: {exc}")
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

        self.result_label.setText(
            f"Движок: {ENGINE_LABELS.get(entry.engine, entry.engine)}\n"
            f"Размер: {entry.size_text}\n"
            f"Файлов: источник распознан, можно добавлять."
        )
        self._probed = entry

    def _accept(self) -> None:
        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "Своя модель", "Укажи источник: ссылку, HF-репозиторий или путь.")
            return

        probed: Optional[ModelEntry] = getattr(self, "_probed", None)
        if probed is None or probed.source != source:
            try:
                from ..core.downloader import inspect_custom_source

                probed = inspect_custom_source(source)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Своя модель", f"Не удалось разобрать источник:\n{exc}")
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
        probed.tags = list(dict.fromkeys([*probed.tags, "своя модель"]))
        self.entry = probed
        self.accept()


# --------------------------------------------------------------------------- #
#  Первый запуск: рекомендация модели
# --------------------------------------------------------------------------- #


class FirstRunDialog(QDialog):
    """Показывается, когда ни одна модель не установлена."""

    downloadRequested = Signal(object)

    def __init__(self, hardware_summary: str, recommendations: List[ModelEntry], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LocalASR · первый запуск")
        self.setModal(True)
        self.setMinimumWidth(720)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        root.addWidget(make_label("Привет! Нужна модель", "H1"))
        root.addWidget(
            make_label(
                "Программа работает полностью офлайн, но моделей в комплекте нет — они весят "
                "сотни мегабайт. Выбери одну (или несколько) и скачай: она ляжет в папку "
                "models/ внутри проекта.\n\n"
                "Вот что подойдёт твоему железу:",
                "Dim",
                word_wrap=True,
            )
        )

        hardware_box = QFrame()
        hardware_box.setObjectName("Banner")
        hardware_layout = QVBoxLayout(hardware_box)
        hardware_layout.setContentsMargins(14, 12, 14, 12)
        hardware_label = QLabel(hardware_summary)
        hardware_label.setWordWrap(True)
        hardware_label.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
        hardware_layout.addWidget(hardware_label)
        root.addWidget(hardware_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(compat.frame_no_frame())
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 8, 0)
        container_layout.setSpacing(10)

        for entry in recommendations:
            card = QFrame()
            card.setObjectName("Card")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(6)

            top = QHBoxLayout()
            name = make_label(entry.name, "H2")
            name.setWordWrap(True)
            top.addWidget(name, stretch=1)
            if entry.recommended:
                top.addWidget(make_label("★ рекомендуем", "BadgeRecommended"))
            layout.addLayout(top)

            chips = FlowLayout(spacing=6)
            for text in (
                entry.engine_label,
                f"⬇ {entry.size_text}",
                f"🧠 {entry.memory_text}",
                f"точность {stars(entry.quality)}",
                f"скорость {stars(entry.speed)}",
            ):
                chips.addWidget(make_label(text, "BadgeEngine"))
            layout.addLayout(chips)

            layout.addWidget(make_label(entry.languages_text, "Dim"))
            layout.addWidget(make_label(entry.recommended_for or entry.description, "Dim", word_wrap=True))

            buttons = QHBoxLayout()
            buttons.addStretch(1)
            if entry.url:
                link = QPushButton("Страница модели ↗")
                link.setObjectName("Link")
                link.clicked.connect(lambda _=False, url=entry.url: open_url(url))
                buttons.addWidget(link)
            download = QPushButton("Скачать")
            download.setObjectName("Primary")
            download.setCursor(compat.CURSOR_POINTING_HAND)
            download.clicked.connect(lambda _=False, item=entry: self.downloadRequested.emit(item))
            buttons.addWidget(download)
            layout.addLayout(buttons)

            container_layout.addWidget(card)

        container_layout.addStretch(1)
        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        bottom = QHBoxLayout()
        bottom.addWidget(
            make_label("Полный список моделей с фильтрами — в правой панели окна.", "Muted", word_wrap=True)
        )
        bottom.addStretch(1)
        close = QPushButton("Выбрать позже")
        close.setObjectName("Ghost")
        close.clicked.connect(self.accept)
        bottom.addWidget(close)
        root.addLayout(bottom)


# --------------------------------------------------------------------------- #
#  Информация о железе
# --------------------------------------------------------------------------- #


class HardwareDialog(QDialog):
    def __init__(self, summary: str, ffmpeg_text: str, notes: List[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Железо и окружение")
        self.setModal(True)
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(10)
        root.addWidget(make_label("Что видит LocalASR", "H2"))

        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setPlainText(summary + "\n\nffmpeg: " + ffmpeg_text)
        box.setMinimumHeight(180)
        root.addWidget(box)

        for note in notes:
            root.addWidget(make_label("• " + note, "Dim", word_wrap=True))

        buttons = QDialogButtonBox(compat.dialog_buttons("Close"))
        buttons.button(compat.dialog_buttons("Close")).setText("Закрыть")
        buttons.rejected.connect(self.accept)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)


# --------------------------------------------------------------------------- #
#  Настройки
# --------------------------------------------------------------------------- #


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, hardware: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.hardware = hardware or {}
        self.setWindowTitle("Настройки")
        self.setModal(True)
        self.setMinimumSize(700, 640)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll_wrap(self._tab_quality()), "Качество текста")
        self.tabs.addTab(self._scroll_wrap(self._tab_speed()), "Скорость и железо")
        self.tabs.addTab(self._scroll_wrap(self._tab_engines()), "Движки и CUDA")
        self.tabs.addTab(self._scroll_wrap(self._tab_about()), "О программе")

        buttons = QDialogButtonBox(compat.dialog_buttons("Save", "Cancel"))
        buttons.button(compat.dialog_buttons("Save")).setText("Сохранить")
        buttons.button(compat.dialog_buttons("Save")).setObjectName("Primary")
        buttons.button(compat.dialog_buttons("Cancel")).setText("Отмена")
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.addWidget(self.tabs, stretch=1)
        root.addWidget(buttons)

    # -- помощники --------------------------------------------------------- #
    @staticmethod
    def _scroll_wrap(page: QWidget) -> QScrollArea:
        """Вкладка со своей прокруткой: пояснения не обрезаются в маленьком окне."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(compat.frame_no_frame())
        scroll.setHorizontalScrollBarPolicy(compat.SCROLLBAR_AS_NEEDED)
        scroll.setWidget(page)
        return scroll

    @staticmethod
    def _hinted(form: QFormLayout, title: str, widget: QWidget, hint: str) -> None:
        """Строка настройки + пояснение «зачем это и как настраивать»."""
        widget.setToolTip(hint)
        form.addRow(title, widget)
        form.addRow(make_label(hint, "Muted", word_wrap=True))

    # -- вкладки ---------------------------------------------------------- #
    def _tab_quality(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(8)

        self.language_combo = QComboBox()
        for code, title in LANGUAGES:
            self.language_combo.addItem(title, code)
        index = self.language_combo.findData(self.settings.language)
        self.language_combo.setCurrentIndex(index if index >= 0 else 1)
        self._hinted(
            form, "Язык речи", self.language_combo,
            "Язык записи. «Определить автоматически» удобно для разноязычных папок, "
            "но явный выбор быстрее и надёжнее на коротких/шумных файлах.",
        )

        self.translate_check = QCheckBox("Переводить результат в английский (task=translate)")
        self.translate_check.setChecked(self.settings.translate_to_english)
        self._hinted(
            form, "", self.translate_check,
            "Модель сразу выдаёт перевод услышанного на английский вместо исходного текста. "
            "Для конспекта русской лекции держи выключенным.",
        )

        self.vad_check = QCheckBox("Отрезать тишину и шум (Silero VAD)")
        self.vad_check.setChecked(self.settings.use_vad)
        self._hinted(
            form, "", self.vad_check,
            "До распознавания вырезает паузы/музыку/шум: быстрее и меньше галлюцинаций. "
            "Выключай только если тишина в записи содержательна.",
        )

        self.anti_hallucination = QCheckBox("Чистить «[музыка]», «Спасибо за просмотр», повторы и петли")
        self.anti_hallucination.setChecked(self.settings.suppress_hallucinations)
        self._hinted(
            form, "", self.anti_hallucination,
            "Постобработка: убирает мусорные сегменты, которые Whisper придумывает на тишине "
            "и музыке, и склеенные повторы одной фразы. Рекомендуется всегда включённым.",
        )

        self.paragraphs_check = QCheckBox("Склеивать фразы в абзацы")
        self.paragraphs_check.setChecked(self.settings.paragraphs)
        self._hinted(
            form, "", self.paragraphs_check,
            "В .txt текст будет абзацами по смысловым паузам, а не «простынёй» из фраз. "
            "Выключи, если нужна каждая фраза с новой строки.",
        )

        self.word_ts_check = QCheckBox("Считать таймкоды каждого слова")
        self.word_ts_check.setChecked(self.settings.word_timestamps)
        self._hinted(
            form, "", self.word_ts_check,
            "Нужно только для покадрово точных субтитров; замедляет распознавание. "
            "Для .txt-конспекта выключи.",
        )

        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 12)
        self.beam_spin.setValue(int(self.settings.beam_size or 5))
        self._hinted(
            form, "Beam size (качество декодинга)", self.beam_spin,
            "Сколько вариантов продолжения фразы держит декодер: 1 — быстрее всего, "
            "5 — оптимально, 8+ — максимум точности и времени. Для лекций хватает 5.",
        )

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlainText(self.settings.initial_prompt)
        self.prompt_edit.setFixedHeight(70)
        self.prompt_edit.setPlaceholderText(
            "Подсказка модели: стиль и термины. Например:\n"
            "Лекция по машинному обучению. Термины: градиентный спуск, трансформер, attention, токенизация."
        )
        self._hinted(
            form, "initial_prompt (стиль/термины)", self.prompt_edit,
            "Короткий пример стиля/тематики, который модель «продолжает». Помогает с терминологией "
            "и оформлением (например, «пиши научным стилем, имена собственные с заглавной»).",
        )

        self.hotwords_edit = QPlainTextEdit()
        self.hotwords_edit.setPlainText(self.settings.hotwords)
        self.hotwords_edit.setFixedHeight(60)
        self.hotwords_edit.setPlaceholderText("Слова-подсказки через запятую: Хаффман, RAG, Kubernetes…")
        self._hinted(
            form, "hotwords (слова-подсказки)", self.hotwords_edit,
            "Список редких слов и терминов конкретной лекции — модель будет писать их верно. "
            "Работает в faster-whisper; остальные движки подсказку игнорируют.",
        )

        return page

    def _tab_speed(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(8)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Авто (рекомендуется)", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("NVIDIA GPU (CUDA)", "cuda")
        index = self.device_combo.findData(self.settings.device)
        self.device_combo.setCurrentIndex(index if index >= 0 else 0)
        self._hinted(
            form, "Устройство", self.device_combo,
            "Где считаем: «Авто» выберет GPU, если CUDA видна, иначе CPU. "
            "Принудительный CPU полезен, когда GPU занят играми/рендером.",
        )

        self.compute_combo = QComboBox()
        for value, title in (
            ("auto", "Авто"),
            ("int8", "int8 — CPU, минимум памяти"),
            ("int8_float16", "int8_float16 — GPU с малой VRAM"),
            ("float16", "float16 — GPU, быстрее всего"),
            ("default", "default — как решит движок"),
        ):
            self.compute_combo.addItem(title, value)
        index = self.compute_combo.findData(self.settings.compute_type)
        self.compute_combo.setCurrentIndex(index if index >= 0 else 0)
        self._hinted(
            form, "Точность вычислений", self.compute_combo,
            "Формат чисел модели в памяти: int8 экономит RAM/VRAM ценой ~0.1–0.3% точности, "
            "float16 быстрее на GPU. «Авто» подберёт по устройству.",
        )

        self.batched_check = QCheckBox("Пакетный декодинг (BatchedInferencePipeline)")
        self.batched_check.setChecked(bool(self.settings.batched))
        self._hinted(
            form, "", self.batched_check,
            "До 4× быстрее на GPU: куски аудио считаются батчем. На CPU выигрыша почти нет, "
            "а памяти нужно больше — при нехватке RAM выключи.",
        )

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(int(self.settings.batch_size or 8))
        self._hinted(
            form, "Размер батча", self.batch_spin,
            "Сколько кусков аудио уходит в батч за раз. Больше — быстрее на GPU, но растёт VRAM. "
            "При ошибках «out of memory» уменьшай до 4/2.",
        )

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setSpecialValueText(f"авто ({self.hardware.get('recommended_threads', '?')})")
        self.threads_spin.setValue(int(self.settings.cpu_threads or 0))
        self._hinted(
            form, "Потоки CPU", self.threads_spin,
            "Сколько потоков CPU отдаём инференсу. 0 (авто) — обычно лучше всего; "
            "ограничь, если параллельно работаешь за машиной и она «тормозит».",
        )

        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 8)
        self.parallel_spin.setValue(int(self.settings.parallel_files or 1))
        self._hinted(
            form, "Файлов одновременно", self.parallel_spin,
            "1 — предсказуемо и экономно по памяти. 2–4 — быстрее на многоядерном CPU/GPU "
            "с большим запасом RAM: каждая копия модели ест память целиком.",
        )

        note = make_label(
            "Результат всегда сохраняется в .txt (имя файла = имени оригинала). "
            "Узкое место должно быть железо, а не Python: декодирование звука идёт в отдельном потоке "
            "заранее (prefetch), модель грузится один раз на всю очередь, а сам инференс работает в "
            "C++/CTranslate2 без GIL.",
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

        from ..asr.registry import engine_available, missing_engine_packages
        from ..asr.base import engine_note

        for engine in ENGINES:
            box = QGroupBox(ENGINE_LABELS[engine])
            box_layout = QVBoxLayout(box)
            installed = engine_available(engine)
            status = make_label(
                "✓ установлен" if installed else "✗ не установлен",
                "BadgeInstalled" if installed else "BadgeWarn",
            )
            status.setToolTip("Проверяется по наличию pip-пакетов движка в .venv проекта")
            box_layout.addWidget(status)
            note = engine_note(engine)
            if note:
                box_layout.addWidget(make_label(note, "Muted", word_wrap=True))
            packages = missing_engine_packages(engine)
            if packages:
                box_layout.addWidget(make_label("Пакеты: " + ", ".join(packages), "Muted", word_wrap=True))
            if not installed:
                button = QPushButton("Установить движок")
                button.setObjectName("Primary")
                button.setCursor(compat.CURSOR_POINTING_HAND)
                button.clicked.connect(
                    lambda _=False, pkgs=packages, eng=engine: self._install_engine(eng, pkgs)
                )
                box_layout.addWidget(button, alignment=compat.ALIGN_LEFT)
            layout.addWidget(box)

        cuda_box = QGroupBox("Ускорение NVIDIA CUDA")
        cuda_layout = QVBoxLayout(cuda_box)
        cuda_ok = bool(self.hardware.get("cuda_ok"))
        cuda_layout.addWidget(
            make_label(
                "✓ CTranslate2 видит CUDA — GPU можно использовать"
                if cuda_ok
                else "✗ CUDA недоступна (нет GPU NVIDIA или не установлены cuBLAS/cuDNN 9)",
                "BadgeInstalled" if cuda_ok else "BadgeWarn",
                word_wrap=True,
            )
        )
        cuda_layout.addWidget(
            make_label(
                "Нужны pip-пакеты nvidia-cublas-cu12 и nvidia-cudnn-cu12 (Linux достаточно).\n"
                "Windows: дополнительно скачай DLL cuBLAS/cuDNN 9 из "
                "github.com/Purfview/whisper-standalone-win (релиз libs) и положи в "
                ".venv\\Lib\\site-packages.",
                "Muted",
                word_wrap=True,
            )
        )
        install_cuda = QPushButton("Установить CUDA-библиотеки (pip)")
        install_cuda.setObjectName("Ghost")
        install_cuda.setCursor(compat.CURSOR_POINTING_HAND)
        install_cuda.clicked.connect(lambda: self._install_engine("cuda", ["-r requirements/gpu.txt"]))
        cuda_layout.addWidget(install_cuda, alignment=compat.ALIGN_LEFT)
        layout.addWidget(cuda_box)

        layout.addStretch(1)
        return page

    def _install_engine(self, engine: str, packages: List[str]) -> None:
        window = self.parent()
        if window is not None and hasattr(window, "install_packages_interactive"):
            self.accept()
            window.install_packages_interactive(packages, f"Установка движка {engine}")
        else:
            QMessageBox.information(self, "Движки", "Установить пакеты можно из главного окна программы.")

    def _tab_about(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        from ..constants import APP_NAME, APP_VERSION

        layout.addWidget(make_label(f"{APP_NAME} {APP_VERSION}", "H1"))
        layout.addWidget(
            make_label(
                "Локальное распознавание речи: лекции, видео, подкасты — без облака и без "
                "отправки данных куда-либо. Видео автоматически превращается в звук.\n\n"
                "Движки: faster-whisper (CTranslate2), whisper.cpp (GGML), Vosk (Kaldi).\n"
                "Всё ставится строго в .venv внутри папки проекта.",
                "Dim",
                word_wrap=True,
            )
        )

        for text, url in (
            ("Репозиторий и документация", "https://github.com/local-asr/LocalASR"),
            ("faster-whisper", "https://github.com/SYSTRAN/faster-whisper"),
            ("whisper.cpp", "https://github.com/ggml-org/whisper.cpp"),
            ("Vosk и модели", "https://alphacephei.com/vosk/models"),
            ("Каталог моделей Hugging Face", "https://huggingface.co/models?pipeline_tag=automatic-speech-recognition"),
        ):
            button = QPushButton(f"{text} ↗")
            button.setObjectName("Link")
            button.setCursor(compat.CURSOR_POINTING_HAND)
            button.clicked.connect(lambda _=False, link=url: open_url(link))
            layout.addWidget(button, alignment=compat.ALIGN_LEFT)

        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------ #
    def _save_and_accept(self) -> None:
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
        self.settings.save()
        self.accept()


# --------------------------------------------------------------------------- #
#  Информация о модели
# --------------------------------------------------------------------------- #


class ModelInfoDialog(QDialog):
    """Вся техническая информация о модели по кнопке «ℹ» в карточке."""

    def __init__(self, entry: ModelEntry, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(entry.name)
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

        root.addWidget(make_label(entry.name, "H1"))

        rows = [
            ("ID (папка в models/)", entry.id),
            ("Движок", entry.engine_label),
            ("Производитель", entry.vendor or "—"),
            ("Источник", entry.source or "—"),
            ("Подпапка", entry.subfolder or "—"),
            ("Размер скачивания", entry.size_text),
            ("Занимает на диске", paths.human_size(entry.installed_size_bytes) if entry.installed else "—"),
            ("Память в работе", entry.memory_text),
            ("Языки", entry.languages_text),
            ("Точность / скорость", f"{stars(entry.quality)} / {stars(entry.speed)}"),
            ("Теги", ", ".join(entry.tags) if entry.tags else "—"),
            ("Лицензия", entry.license or "—"),
            ("Релиз", entry.released or "—"),
            ("Установлена", "да: " + str(entry.install_dir) if entry.installed else "нет"),
        ]
        table = QFormLayout()
        table.setSpacing(6)
        for key, value in rows:
            label = make_label(str(value), "Dim", word_wrap=True)
            label.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
            table.addRow(make_label(key, "Muted"), label)
        root.addLayout(table)

        if entry.recommended_for:
            root.addWidget(make_label("Кому подходит: " + entry.recommended_for, "BadgeRecommended", word_wrap=True))

        description = make_label(entry.description or "", "Dim", word_wrap=True)
        description.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
        root.addWidget(description)

        if entry.defaults:
            root.addWidget(make_label("Параметры по умолчанию: " + str(entry.defaults), "Muted", word_wrap=True))

        root.addStretch(1)
        scroll.setWidget(page)
        outer.addWidget(scroll, stretch=1)

        buttons = QHBoxLayout()
        if entry.url:
            link = QPushButton("Открыть страницу модели ↗")
            link.setObjectName("Ghost")
            link.setCursor(compat.CURSOR_POINTING_HAND)
            link.setToolTip("Откроется страница модели в браузере (Hugging Face / сайт автора)")
            link.clicked.connect(lambda: open_url(entry.url))
            buttons.addWidget(link)
        buttons.addStretch(1)
        close = QPushButton("Закрыть")
        close.setObjectName("Primary")
        close.setCursor(compat.CURSOR_POINTING_HAND)
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        outer.addLayout(buttons)
