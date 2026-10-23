"""Панель моделей: поиск, фильтры, карточки, кнопки «скачать / выбрать / удалить»."""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
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


class ModelListPanel(QFrame):
    """Список моделей с фильтрами. Ничего не качает сам — только сигналы."""

    modelSelected = Signal(object)        # ModelEntry
    downloadRequested = Signal(object)    # ModelEntry
    deleteRequested = Signal(object)      # ModelEntry
    infoRequested = Signal(object)        # ModelEntry
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
            "По умолчанию ничего не установлено. Скачай модель — она появится в списке "
            "и кнопка «РАБОТАТЬ» станет активной.",
            "Dim",
            word_wrap=True,
        )
        root.addWidget(subtitle)

        # --- фильтры ------------------------------------------------------ #
        filters = QFrame()
        filters_layout = QVBoxLayout(filters)
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.setSpacing(8)

        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по названию, тегу, языку…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filters)
        search_row.addWidget(self.search, stretch=1)

        self.engine_filter = QComboBox()
        self.engine_filter.addItem("Все движки", "")
        for engine in ENGINES:
            self.engine_filter.addItem(ENGINE_LABELS[engine], engine)
        self.engine_filter.currentIndexChanged.connect(self._apply_filters)
        self.engine_filter.setMinimumWidth(190)
        search_row.addWidget(self.engine_filter)
        filters_layout.addLayout(search_row)

        toggles = QHBoxLayout()
        toggles.setSpacing(14)
        self.only_installed = QCheckBox("Только установленные")
        self.only_installed.toggled.connect(self._apply_filters)
        self.only_russian = QCheckBox("С русским языком")
        self.only_russian.setChecked(True)
        self.only_russian.toggled.connect(self._apply_filters)
        toggles.addWidget(self.only_installed)
        toggles.addWidget(self.only_russian)
        toggles.addStretch(1)
        filters_layout.addLayout(toggles)

        root.addWidget(filters)

        # --- кнопки ------------------------------------------------------- #
        actions = QHBoxLayout()
        actions.setSpacing(8)

        self.custom_button = QPushButton("＋ Своя модель (ссылка / путь)")
        self.custom_button.setObjectName("Ghost")
        self.custom_button.setCursor(compat.CURSOR_POINTING_HAND)
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
        self.scroll.setHorizontalScrollBarPolicy(compat.SCROLLBAR_ALWAYS_OFF)
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

    def refresh_entry(self, entry: ModelEntry) -> None:
        card = self._cards.get(entry.id)
        if card is not None:
            card.refresh_state(entry)

    def card(self, model_id: str) -> Optional[ModelCard]:
        return self._cards.get(model_id)

    def set_selected(self, model_id: str) -> None:
        for card_id, card in self._cards.items():
            card.set_selected(card_id == model_id)

    def set_hardware(self, hardware: dict) -> None:
        self.hardware = hardware or {}

    # ------------------------------------------------------------------ #
    def _apply_filters(self, *_args) -> None:
        query = self.search.text().strip().lower()
        engine = self.engine_filter.currentData() or ""
        installed_only = self.only_installed.isChecked()
        russian_only = self.only_russian.isChecked()

        visible = 0
        for card in self._cards.values():
            entry = card.entry
            show = True
            if engine and entry.engine != engine:
                show = False
            if installed_only and not entry.installed:
                show = False
            if russian_only and not entry.supports_russian:
                show = False
            if show and query:
                haystack = " ".join(
                    [
                        entry.name,
                        entry.id,
                        entry.engine,
                        entry.description,
                        entry.languages_text,
                        " ".join(entry.tags),
                        " ".join(entry.languages),
                    ]
                ).lower()
                show = all(part in haystack for part in query.split())
            card.setVisible(show)
            if show:
                visible += 1

        self.empty_label.setVisible(visible == 0)
        if visible == 0:
            self.empty_label.setText(
                "Под эти фильтры ничего не нашлось.\n"
                "Сними галочку «С русским языком» или очисти поиск — "
                "в каталоге есть и мультиязычные модели."
            )
