"""Панель моделей: поиск, мощные фильтры, сортировка, карточки моделей."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..constants import ENGINE_LABELS, ENGINES
from ..core.models_catalog import ModelCatalog, ModelEntry
from . import compat
from .widgets import ModelCard, make_label

GB = 1024 ** 3

# Весовые категории: (подпись, минимум байт, максимум байт)
SIZE_CLASSES = [
    ("", 0, 0),
    ("до 500 МБ", 0, 500 * 1024 ** 2),
    ("500 МБ – 1 ГБ", 500 * 1024 ** 2, 1 * GB),
    ("1 – 2 ГБ", 1 * GB, 2 * GB),
    ("2 – 4 ГБ", 2 * GB, 4 * GB),
    ("больше 4 ГБ", 4 * GB, 0),
]

RAM_CLASSES = [
    ("", 0, 0),
    ("до 2 ГБ RAM", 0, 2048),
    ("2 – 4 ГБ RAM", 2048, 4096),
    ("4 – 8 ГБ RAM", 4096, 8192),
    ("больше 8 ГБ RAM", 8192, 0),
]

SORTS = [
    ("recommended", "Сначала рекомендуемые"),
    ("quality", "По точности"),
    ("size", "По размеру (меньше → больше)"),
    ("speed", "По скорости"),
    ("name", "По названию"),
]


class ModelListPanel(QFrame):
    """Список моделей с фильтрами. Ничего не качает сам — только сигналы."""

    modelSelected = Signal(object)        # ModelEntry
    downloadRequested = Signal(object)    # ModelEntry
    deleteRequested = Signal(object)
    infoRequested = Signal(object)
    customRequested = Signal()
    refreshRequested = Signal()

    def __init__(self, catalog: ModelCatalog, hardware: Optional[dict] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SidePanel")
        self.catalog = catalog
        self.hardware = hardware or {}
        self._cards: Dict[str, ModelCard] = {}

        self._build_ui()
        self.rebuild()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = make_label("Модели", "H1")
        header.addWidget(title)
        header.addStretch(1)

        self.count_label = make_label("", "Muted")
        header.addWidget(self.count_label)
        root.addLayout(header)

        subtitle = make_label(
            "По умолчанию ничего не установлено. Скачай модель, затем выбери её — "
            "кликом по карточке (или в выпадающем списке «Модель» в блоке «3 · Работа»: "
            "там перечислены ВСЕ скачанные модели, выбрать можно одну). "
            "Пока модель не выбрана, кнопка «РАБОТАТЬ» неактивна.",
            "Dim",
            word_wrap=True,
        )
        root.addWidget(subtitle)

        # --- фильтры ------------------------------------------------------ #
        filters = QFrame()
        filters_layout = QGridLayout(filters)
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.setSpacing(8)
        filters_layout.setHorizontalSpacing(10)

        row = 0
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск: название, тег, язык, описание…")
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip("Ищет по названию, ID, описанию, тегам и языкам модели")
        self.search.textChanged.connect(self._apply_filters)
        filters_layout.addWidget(self.search, row, 0, 1, 2)

        self.engine_filter = QComboBox()
        self.engine_filter.addItem("Все движки", "")
        for engine in ENGINES:
            self.engine_filter.addItem(ENGINE_LABELS[engine], engine)
        self.engine_filter.setToolTip(
            "Движок = чем считаем модель:\n"
            "• faster-whisper — быстрее всех на CPU/GPU (основной);\n"
            "• whisper.cpp — минимум RAM;\n"
            "• Vosk — мгновенный старт, простее по качеству;\n"
            "• Qwen3-ASR / GigaAM — torch-модели нового поколения (ставятся по требованию)."
        )
        self.engine_filter.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self.engine_filter, row, 2, 1, 2)

        row += 1
        self.vendor_filter = QComboBox()
        self.vendor_filter.setToolTip(
            "Кто сделал модель: OpenAI (Whisper), Qwen (Alibaba), Sber (GigaAM), "
            "fine-tune'ы сообщества и т.д."
        )
        self.vendor_filter.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self._filter_label("Производитель"), row, 0)
        filters_layout.addWidget(self.vendor_filter, row, 1)

        self.size_filter = QComboBox()
        self.size_filter.setToolTip(
            "Весовой класс модели — сколько места займёт скачивание на диске. "
            "Прикидывай по свободному месту и скорости сети."
        )
        self.size_filter.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self._filter_label("Размер"), row, 2)
        filters_layout.addWidget(self.size_filter, row, 3)

        row += 1
        self.quality_filter = QComboBox()
        self.quality_filter.setToolTip(
            "Минимальная оценка точности (звёзды в карточке). "
            "★★★★ и выше — модели, пригодные для конспекта без вычитки."
        )
        self.quality_filter.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self._filter_label("Точность"), row, 0)
        filters_layout.addWidget(self.quality_filter, row, 1)

        self.ram_filter = QComboBox()
        self.ram_filter.setToolTip(
            "Сколько оперативной памяти модель ест в работе. "
            "Смотри свою свободную RAM в статус-баре внизу окна."
        )
        self.ram_filter.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self._filter_label("Память"), row, 2)
        filters_layout.addWidget(self.ram_filter, row, 3)

        row += 1
        self.sort_combo = QComboBox()
        self.sort_combo.setToolTip("Порядок карточек в списке")
        self.sort_combo.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self._filter_label("Сортировка"), row, 0)
        filters_layout.addWidget(self.sort_combo, row, 1, 1, 2)

        self.reset_button = QPushButton("Сбросить")
        self.reset_button.setObjectName("Ghost")
        self.reset_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.reset_button.setToolTip("Вернуть все фильтры и поиск в исходное состояние")
        self.reset_button.clicked.connect(self._reset_filters)
        filters_layout.addWidget(self.reset_button, row, 3)

        row += 1
        toggles = QHBoxLayout()
        toggles.setSpacing(14)
        self.only_installed = QCheckBox("Только установленные")
        self.only_installed.setToolTip("Показать модели, которые уже скачаны в папку models/")
        self.only_installed.toggled.connect(self._apply_filters)
        self.only_russian = QCheckBox("С русским языком")
        self.only_russian.setChecked(True)
        self.only_russian.setToolTip("Оставить модели, понимающие русскую речь (сняв галочку, увидишь и EN-only)")
        self.only_russian.toggled.connect(self._apply_filters)
        toggles.addWidget(self.only_installed)
        toggles.addWidget(self.only_russian)
        toggles.addStretch(1)
        filters_layout.addLayout(toggles, row, 0, 1, 4)

        root.addWidget(filters)

        # --- кнопки ------------------------------------------------------- #
        actions = QHBoxLayout()
        actions.setSpacing(8)

        self.custom_button = QPushButton("＋ Своя модель (ссылка / путь)")
        self.custom_button.setObjectName("Ghost")
        self.custom_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.custom_button.setToolTip(
            "Добавить модель, которой нет в каталоге: HF-репозиторий, прямую ссылку или папку на диске"
        )
        self.custom_button.clicked.connect(self.customRequested)
        actions.addWidget(self.custom_button, stretch=1)

        self.refresh_button = QPushButton("⟳ Обновить каталог")
        self.refresh_button.setObjectName("Ghost")
        self.refresh_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.refresh_button.setToolTip("Проверить наличие новых моделей в онлайн-каталоге")
        self.refresh_button.clicked.connect(self.refreshRequested)
        actions.addWidget(self.refresh_button)
        root.addLayout(actions)

        # --- скролл с карточками ------------------------------------------ #
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(compat.SCROLLBAR_AS_NEEDED)
        self.scroll.setFrameShape(compat.frame_no_frame())

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 6, 0)
        self.container_layout.setSpacing(10)
        self.container_layout.addStretch(1)
        self.scroll.setWidget(self.container)
        root.addWidget(self.scroll, stretch=1)

        self.empty_label = make_label("", "Dim", word_wrap=True)
        self.empty_label.setAlignment(compat.ALIGN_CENTER)
        self.empty_label.setVisible(False)
        root.addWidget(self.empty_label)

        self._fill_filter_options()

    @staticmethod
    def _filter_label(text: str) -> QLabel:
        label = make_label(text, "Muted")
        label.setAlignment(compat.ALIGN_RIGHT_VCENTER)
        # иначе в узкой панелиGridLayout сжимает подпись и она обрезается слева
        label.setMinimumWidth(label.sizeHint().width())
        return label

    # ------------------------------------------------------------------ #
    def minimumSizeHint(self):
        """Панель может сужаться до 340 px: фильтры и карточки переносятся."""
        from PySide6.QtCore import QSize

        return QSize(340, 420)

    def _fill_filter_options(self) -> None:
        self.vendor_filter.addItem("Все производители", "")
        vendors = sorted(
            {entry.vendor for entry in self.catalog.all() if entry.vendor},
            key=str.lower,
        )
        for vendor in vendors:
            self.vendor_filter.addItem(vendor, vendor)

        self.size_filter.addItem("Любой размер", "")
        for title, _lo, _hi in SIZE_CLASSES[1:]:
            self.size_filter.addItem(title, title)

        self.quality_filter.addItem("Любая точность", 0)
        for stars in (3, 4, 5):
            self.quality_filter.addItem("★" * stars + " и выше", stars)

        self.ram_filter.addItem("Любая память", "")
        for title, _lo, _hi in RAM_CLASSES[1:]:
            self.ram_filter.addItem(title, title)

        for key, title in SORTS:
            self.sort_combo.addItem(title, key)

    # ------------------------------------------------------------------ #
    def rebuild(self) -> None:
        """Полностью перестраивает список карточек (после установки/удаления)."""
        for card in self._cards.values():
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

        while self.container_layout.count() > 1:
            item = self.container_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        entries = self.catalog.sorted_for_display()
        installed_count = sum(1 for entry in entries if entry.installed)

        for entry in entries:
            card = ModelCard(entry, self.hardware)
            card.downloadRequested.connect(self.downloadRequested)
            card.selectRequested.connect(self.modelSelected)
            card.deleteRequested.connect(self.deleteRequested)
            card.infoRequested.connect(self.infoRequested)
            self._cards[entry.id] = card
            self.container_layout.insertWidget(self.container_layout.count() - 1, card)

        self.count_label.setText(
            f"{len(entries)} в каталоге · {installed_count} установлено"
        )
        self._apply_filters()

    def card(self, model_id: str) -> Optional[ModelCard]:
        return self._cards.get(model_id)

    def set_selected(self, model_id: str) -> None:
        for card_id, card in self._cards.items():
            card.set_selected(card_id == model_id)

    def set_hardware(self, hardware: dict) -> None:
        self.hardware = hardware or {}

    # ------------------------------------------------------------------ #
    def _reset_filters(self) -> None:
        self.search.clear()
        self.engine_filter.setCurrentIndex(0)
        self.vendor_filter.setCurrentIndex(0)
        self.size_filter.setCurrentIndex(0)
        self.quality_filter.setCurrentIndex(0)
        self.ram_filter.setCurrentIndex(0)
        self.sort_combo.setCurrentIndex(0)
        self.only_installed.setChecked(False)
        self.only_russian.setChecked(True)

    def _sort_key(self, entry: ModelEntry):
        mode = self.sort_combo.currentData() or "recommended"
        if mode == "quality":
            return (-entry.quality, -entry.speed, entry.size_bytes or 0)
        if mode == "size":
            return (entry.size_bytes or 0, -entry.quality)
        if mode == "speed":
            return (-entry.speed, -entry.quality)
        if mode == "name":
            return (entry.name.lower(),)
        return (
            0 if entry.installed else 1,
            0 if entry.recommended else 1,
            0 if entry.supports_russian else 1,
            -entry.quality,
            entry.ram_mb or 99_999,
        )

    def _apply_filters(self, *_args) -> None:
        query = self.search.text().strip().lower()
        engine = self.engine_filter.currentData() or ""
        vendor = self.vendor_filter.currentData() or ""
        size_title = self.size_filter.currentData() or ""
        ram_title = self.ram_filter.currentData() or ""
        min_quality = int(self.quality_filter.currentData() or 0)
        installed_only = self.only_installed.isChecked()
        russian_only = self.only_russian.isChecked()

        size_lo = size_hi = 0
        for title, lo, hi in SIZE_CLASSES:
            if title == size_title:
                size_lo, size_hi = lo, hi
        ram_lo = ram_hi = 0
        for title, lo, hi in RAM_CLASSES:
            if title == ram_title:
                ram_lo, ram_hi = lo, hi

        visible_entries: List[ModelEntry] = []
        for card in self._cards.values():
            entry = card.entry
            show = True
            if engine and entry.engine != engine:
                show = False
            if show and vendor and entry.vendor != vendor:
                show = False
            if show and installed_only and not entry.installed:
                show = False
            if show and russian_only and not entry.supports_russian:
                show = False
            if show and min_quality and entry.quality < min_quality:
                show = False
            if show and (size_lo or size_hi):
                size = entry.size_bytes or 0
                if size_lo and size < size_lo:
                    show = False
                if size_hi and size > size_hi:
                    show = False
            if show and (ram_lo or ram_hi):
                ram = entry.ram_mb or 0
                if ram_lo and ram < ram_lo:
                    show = False
                if ram_hi and ram > ram_hi:
                    show = False
            if show and query:
                haystack = " ".join(
                    [
                        entry.name,
                        entry.id,
                        entry.engine,
                        entry.vendor,
                        entry.description,
                        entry.languages_text,
                        " ".join(entry.tags),
                        " ".join(entry.languages),
                    ]
                ).lower()
                show = all(part in haystack for part in query.split())
            card.setVisible(show)
            if show:
                visible_entries.append(entry)

        # переставляем карточки в порядке сортировки (без пересоздания)
        ordered = sorted(visible_entries, key=self._sort_key)
        for index, entry in enumerate(ordered):
            card = self._cards[entry.id]
            self.container_layout.removeWidget(card)
            self.container_layout.insertWidget(index, card)

        visible = len(ordered)
        self.empty_label.setVisible(visible == 0)
        if visible == 0:
            self.empty_label.setText(
                "Под эти фильтры ничего не нашлось.\n"
                "Попробуй снять галочку «С русским языком», ослабить «Точность» "
                "или нажать «Сбросить» — в каталоге есть и мультиязычные модели."
            )
