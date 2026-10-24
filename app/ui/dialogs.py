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
        self.setWindowTitle("Настройки")
        self.setModal(True)
        self.setMinimumSize(700, 640)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll_wrap(self._tab_quality()), "Качество текста")
        self.tabs.addTab(self._scroll_wrap(self._tab_speed()), "Скорость и железо")
        self.tabs.addTab(self._scroll_wrap(self._tab_engines()), "Движки")
        self.tabs.addTab(self._scroll_wrap(self._tab_about()), "О программе")

        buttons = QDialogButtonBox(compat.dialog_buttons("Save", "Cancel"))
        buttons.button(compat.dialog_buttons("Save")).setText("Сохранить")
        buttons.button(compat.dialog_buttons("Save")).setObjectName("Primary")
        buttons.button(compat.dialog_buttons("Cancel")).setText("Отмена")
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)

        self.reset_button = QPushButton("Сбросить настройки по умолчанию")
        self.reset_button.setObjectName("Ghost")
        self.reset_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.reset_button.setToolTip(
            "Вернуть ВСЕ настройки к стандартным: параметры распознавания, папки,\n"
            "положение и размер окна, фильтры каталога моделей."
        )
        self.reset_button.clicked.connect(self._reset_defaults)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.reset_button)
        buttons_row.addStretch(1)
        buttons_row.addWidget(buttons)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.addWidget(self.tabs, stretch=1)
        root.addLayout(buttons_row)

    # -- сброс настроек ----------------------------------------------------- #
    def _reset_defaults(self) -> None:
        answer = QMessageBox.question(
            self, "Сброс настроек",
            "Вернуть все настройки к стандартным?\n\n"
            "Будут сброшены: параметры распознавания, папка результатов,\n"
            "положение и размер окна, фильтры каталога моделей.\n"
            "Скачанные модели и файлы результатов не затрагиваются.",
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_no(),
        )
        if answer != compat.message_box_yes():
            return
        self.settings.reset_to_defaults()
        if self._on_reset_defaults is not None:
            try:
                self._on_reset_defaults()
            except Exception:  # noqa: BLE001
                pass
        self.done(0)   # не accept(): старые значения виджетов сохранять не нужно

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
            "Принудительный CPU полезен, когда GPU занят играми/рендером. "
            "Если выберешь GPU без установленных CUDA-библиотек — программа предложит "
            "установку (вкладка «Движки»), а до тех пор будет работать на CPU.",
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

        from ..asr.base import engine_install_size, engine_note
        from ..asr.registry import engine_available

        layout.addWidget(
            make_label(
                "Движок — это библиотека, которой программа распознаёт речь. "
                "Ставить что-то заранее не нужно: отсутствующий движок установится "
                "сам, когда ты запустишь подходящую модель (программа спросит "
                "подтверждение и покажет объём скачивания).",
                "Muted",
                word_wrap=True,
            )
        )

        for engine in ENGINES:
            box = QGroupBox(ENGINE_LABELS[engine])
            box.setToolTip("Статус движка в виртуальном окружении проекта (.venv)")
            box_layout = QVBoxLayout(box)
            installed = engine_available(engine)
            status = make_label(
                "✓ установлен" if installed else "○ не установлен — установится автоматически при необходимости",
                "BadgeInstalled" if installed else "Muted",
            )
            status.setToolTip(
                "Проверяется по наличию pip-пакетов движка в .venv проекта.\n"
                "Отсутствующий движок программа предложит скачать в момент запуска модели."
            )
            status.setWordWrap(True)
            box_layout.addWidget(status)
            note = engine_note(engine)
            if note:
                box_layout.addWidget(make_label(note, "Muted", word_wrap=True))
            size_text = engine_install_size(engine)
            if size_text and not installed:
                box_layout.addWidget(
                    make_label(f"Объём установки: {size_text}", "Dim", word_wrap=True)
                )
            layout.addWidget(box)

        # CUDA для faster-whisper: кнопка установки появляется ТОЛЬКО когда
        # она действительно нужна (карта NVIDIA есть, а библиотек нет).
        cuda_box = QGroupBox("Ускорение NVIDIA CUDA (для faster-whisper)")
        cuda_layout = QVBoxLayout(cuda_box)
        cuda_ok = bool(self.hardware.get("cuda_ok"))
        has_nvidia = bool(self.hardware.get("has_nvidia"))
        if cuda_ok:
            cuda_layout.addWidget(
                make_label("✓ CUDA доступна — Whisper-модели могут работать на GPU",
                           "BadgeInstalled", word_wrap=True)
            )
        elif not has_nvidia:
            cuda_layout.addWidget(
                make_label("○ Видеокарта NVIDIA не обнаружена — раздел не требуется",
                           "Muted", word_wrap=True)
            )
        else:
            cuda_layout.addWidget(
                make_label(
                    "○ Видеокарта NVIDIA есть, но библиотеки CUDA (cuBLAS/cuDNN 9) "
                    "не установлены — распознавание идёт на процессоре",
                    "BadgeWarn",
                    word_wrap=True,
                )
            )
            cuda_layout.addWidget(
                make_label(
                    "Установка: ~1 ГБ (nvidia-cublas-cu12 + nvidia-cudnn-cu12, один раз).\n"
                    "Windows: дополнительно нужны системные DLL cuBLAS/cuDNN 9 — скачать из "
                    "github.com/Purfview/whisper-standalone-win (релиз libs) и положить в "
                    ".venv\\Lib\\site-packages.",
                    "Muted",
                    word_wrap=True,
                )
            )
            install_cuda = QPushButton("Установить CUDA-библиотеки (~1 ГБ)")
            install_cuda.setObjectName("Primary")
            install_cuda.setCursor(compat.CURSOR_POINTING_HAND)
            install_cuda.setToolTip(
                "Скачает cuBLAS/cuDNN в .venv проекта (разово, ~1 ГБ).\n"
                "После этого Whisper-модели смогут работать на видеокарте."
            )
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
            window.install_packages_interactive(packages, f"Установка: {engine}")
        else:
            QMessageBox.information(self, "Движки", "Установить пакеты можно из главного окна программы.")

    def _tab_about(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        from ..constants import APP_NAME, APP_VERSION, REPO_URL

        layout.addWidget(make_label(f"{APP_NAME} {APP_VERSION}", "H1"))
        layout.addWidget(
            make_label(
                "Локальное распознавание речи: лекции, видео, подкасты — без облака и без "
                "отправки данных куда-либо. Видео автоматически превращается в звук.\n\n"
                "Движки: faster-whisper (CTranslate2), whisper.cpp (GGML), Vosk (Kaldi),\n"
                "Qwen3-ASR и GigaAM (torch), hf-transformers (любые ASR-модели HF).\n"
                "Всё ставится строго в .venv внутри папки проекта.",
                "Dim",
                word_wrap=True,
            )
        )

        for text, url in (
            ("Репозиторий и документация", REPO_URL),
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
            ("Число параметров", entry.params_text),
            ("Квантование (формат весов)", entry.quantization_text),
            ("Год релиза", entry.released_year_text),
            ("Размер скачивания", entry.size_text),
            ("Занимает на диске", paths.human_size(entry.installed_size_bytes) if entry.installed else "—"),
            ("Память в работе", entry.memory_text),
            ("Языки", entry.languages_text),
            ("Точность / скорость", f"{stars(entry.quality)} / {stars(entry.speed)}"),
            ("Теги", ", ".join(entry.tags) if entry.tags else "—"),
            ("Лицензия", entry.license or "—"),
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
            hint = make_label("Кому подходит: " + entry.recommended_for, "Dim", word_wrap=True)
            root.addWidget(hint)

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


# --------------------------------------------------------------------------- #
#  Диагностика и восстановление (целостность программы)
# --------------------------------------------------------------------------- #


class DiagnosticsDialog(QDialog):
    """Сервис → Диагностика и восстановление.

    Проверяет три уровня устойчивости программы:
      1. файлы самого проекта (удалил случайно — восстановим с GitHub);
      2. рабочие папки (models/, outputs/… — создаются автоматически);
      3. окружение .venv (не хватает библиотек — починим; лишние пакеты —
         предложим удалить, чтобы освободить место).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Диагностика и восстановление")
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

        outer.addWidget(make_label("Состояние программы", "H1"))
        outer.addWidget(
            make_label(
                "Проверка файлов проекта, рабочих папок и библиотек в .venv. "
                "Здесь ничего не ломается само: программа только сообщает о проблемах "
                "и предлагает исправить их одной кнопкой.",
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

        # --- 1. файлы проекта
        files_box = QGroupBox("Файлы программы")
        files_layout = QVBoxLayout(files_box)
        self.files_status = make_label("Проверяю…", "Muted", word_wrap=True)
        files_layout.addWidget(self.files_status)
        self.files_list = make_label("", "Dim", word_wrap=True)
        self.files_list.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
        self.files_list.setVisible(False)
        files_layout.addWidget(self.files_list)
        self.restore_button = QPushButton("Восстановить недостающие файлы с GitHub")
        self.restore_button.setObjectName("Primary")
        self.restore_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.restore_button.setToolTip(
            "Скачает удалённые файлы программы заново (несколько сотен килобайт).\n"
            "Модели, результаты и настройки не затрагиваются."
        )
        self.restore_button.setVisible(False)
        self.restore_button.clicked.connect(self._restore_files)
        files_layout.addWidget(self.restore_button, alignment=compat.ALIGN_LEFT)
        root.addWidget(files_box)

        # --- 2. папки
        folders_box = QGroupBox("Рабочие папки")
        folders_layout = QVBoxLayout(folders_box)
        self.folders_status = make_label("Проверяю…", "Muted", word_wrap=True)
        folders_layout.addWidget(self.folders_status)
        folders_layout.addWidget(
            make_label(
                "Эти папки создаются автоматически, когда нужны. Их отсутствие "
                "не ошибка: например, models/ появится при первом скачивании модели.",
                "Dim",
                word_wrap=True,
            )
        )
        root.addWidget(folders_box)

        # --- 3. окружение
        venv_box = QGroupBox("Окружение (.venv)")
        venv_layout = QVBoxLayout(venv_box)
        self.venv_status = make_label("Проверяю…", "Muted", word_wrap=True)
        self.venv_status.setWordWrap(True)
        venv_layout.addWidget(self.venv_status)
        self.repair_button = QPushButton("Починить окружение (доустановить недостающее)")
        self.repair_button.setObjectName("Primary")
        self.repair_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.repair_button.setToolTip(
            "Перепроверит базовые библиотеки в .venv и доустановит отсутствующие.\n"
            "Системный Python не затрагивается."
        )
        self.repair_button.setVisible(False)
        self.repair_button.clicked.connect(self._repair_env)
        venv_layout.addWidget(self.repair_button, alignment=compat.ALIGN_LEFT)

        self.extra_hint = make_label("", "Muted", word_wrap=True)
        self.extra_hint.setVisible(False)
        venv_layout.addWidget(self.extra_hint)
        self.extra_list = QListWidget()
        self.extra_list.setVisible(False)
        self.extra_list.setMaximumHeight(180)
        self.extra_list.setToolTip(
            "Пакеты, которые не требуются ни программе, ни другим установленным\n"
            "пакетам. Отметь галочками то, что удалить. Если ставил их сам для\n"
            "других задач — оставь."
        )
        self.extra_list.itemChanged.connect(self._update_delete_button)
        venv_layout.addWidget(self.extra_list)
        self.delete_button = QPushButton("Удалить выбранные пакеты")
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
        self.refresh_button = QPushButton("⟳ Перепроверить")
        self.refresh_button.setObjectName("Ghost")
        self.refresh_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.refresh_button.setToolTip("Запустить проверку заново")
        self.refresh_button.clicked.connect(self._start_audit)
        buttons.addWidget(self.refresh_button)
        buttons.addStretch(1)
        close = QPushButton("Закрыть")
        close.setObjectName("Primary")
        close.setCursor(compat.CURSOR_POINTING_HAND)
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        outer.addLayout(buttons)

    # ------------------------------------------------------------------ #
    def _start_audit(self) -> None:
        if self.audit_worker is not None and self.audit_worker.isRunning():
            return
        from .workers import EnvAuditWorker

        self.refresh_button.setEnabled(False)
        self.files_status.setText("Проверяю файлы программы…")
        self.venv_status.setText("Проверяю окружение .venv… (pip list занимает несколько секунд)")
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
            self.files_status.setText(f"Не удалось завершить проверку: {self.report['error']}")
            return

        # --- файлы
        missing = list(self.report.get("missing_files") or [])
        if missing:
            self.files_status.setText(
                f"⚠ Отсутствует файлов: {len(missing)}. Программа может работать "
                "неполноценно — восстанови их (нужен интернет, вес ~сотни КБ)."
            )
            self.files_list.setText("  • " + "\n  • ".join(missing[:30]))
            self.files_list.setVisible(True)
            self.restore_button.setVisible(True)
        else:
            self.files_status.setText("✓ Все файлы программы на месте")
            self.files_list.setVisible(False)
            self.restore_button.setVisible(False)

        # --- папки
        folders = list(self.report.get("folders") or [])
        lines = [f"• {item['name']} — {item['status']} ({item['purpose']})" for item in folders]
        self.folders_status.setText("\n".join(lines) or "—")

        # --- venv
        base_missing = list(self.report.get("base_missing") or [])
        python = str(self.report.get("python") or "")
        if base_missing:
            self.venv_status.setText(
                f"⚠ Не хватает базовых библиотек: {', '.join(base_missing)}\n"
                f"Интерпретатор: {python}"
            )
            self.repair_button.setVisible(True)
        else:
            self.venv_status.setText(f"✓ Все базовые библиотеки установлены\nИнтерпретатор: {python}")
            self.repair_button.setVisible(False)

        # --- лишние пакеты
        extra = list(self.report.get("extra_packages") or [])
        self.extra_list.clear()
        if extra:
            from ..paths import human_size

            total = sum(int(item.get("size_bytes") or 0) for item in extra)
            self.extra_hint.setText(
                f"Пакеты, которые программе не нужны (не требуются ни ей, ни другим "
                f"пакетам): {len(extra)} шт., всего {human_size(total)}. "
                "Удалять не обязательно — это только экономия места."
            )
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
            self.extra_hint.setText("Лишних пакетов в .venv не найдено.")
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
            self.delete_button.setText(f"Удалить выбранные ({len(names)} шт., ~{human_size(total)})")
            self.delete_button.setEnabled(True)
        else:
            self.delete_button.setText("Удалить выбранные пакеты")
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
            "Восстановление файлов",
            f"Скачиваем {len(missing)} файл(ов) из репозитория проекта…",
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
        dialog.finish("Готово" if ok else "Остановлено")
        dialog.accept()
        worker, self.restore_worker = self.restore_worker, None
        if worker is not None:
            worker.deleteLater()
        if ok:
            QMessageBox.information(self, "Восстановление", message)
        else:
            QMessageBox.warning(self, "Восстановление", message)
        self._start_audit()

    def _repair_env(self) -> None:
        window = self.parent()
        requirements = paths.ROOT_DIR / "requirements" / "base.txt"
        if not requirements.is_file():
            QMessageBox.warning(
                self, "Ремонт окружения",
                f"Файл {requirements} отсутствует.\n"
                "Сначала восстанови файлы программы (кнопка выше).",
            )
            return
        if window is not None and hasattr(window, "install_packages_interactive"):
            self.accept()
            window.install_packages_interactive(["-r", str(requirements)], "Ремонт окружения (.venv)")
        else:
            QMessageBox.information(self, "Ремонт окружения", "Запусти ремонт из главного окна программы.")

    def _delete_extra(self) -> None:
        names = self._selected_extra()
        if not names:
            return
        answer = QMessageBox.question(
            self, "Удаление пакетов",
            "Удалить из .venv выбранные пакеты?\n\n  • " + "\n  • ".join(names[:20])
            + ("\n  …" if len(names) > 20 else "")
            + "\n\nПрограмме они не нужны; системный Python не затрагивается.",
            compat.message_box_yes() | compat.message_box_no(),
            compat.message_box_no(),
        )
        if answer != compat.message_box_yes():
            return
        if self.package_worker is not None and self.package_worker.isRunning():
            return
        from .workers import PackageInstaller

        dialog = ProgressDialog("Удаление пакетов", ", ".join(names), parent=self)
        worker = PackageInstaller(names, "Удаление пакетов", self, mode="uninstall")
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
        dialog.finish("Готово" if ok else "Остановлено")
        dialog.accept()
        worker, self.package_worker = self.package_worker, None
        if worker is not None:
            worker.deleteLater()
        self.package_dialog = None
        if not ok:
            QMessageBox.warning(self, "Удаление", message)
        self._start_audit()
