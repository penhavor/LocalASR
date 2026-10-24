"""Панель моделей: поиск, фильтры (включая числовые диапазоны), сортировка, карточки."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QSize, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..constants import ENGINE_LABELS, ENGINES
from ..core.models_catalog import ModelCatalog, ModelEntry
from ..logutils import get_logger
from ..settings import Settings
from . import compat
from .widgets import ModelCard, make_label

log = get_logger("ui.models")

# Сортировки: (ключ, подпись). Числовые характеристики sorts поддерживают
# диапазоны-фильтры (см. _apply_filters), чтобы список не зависел от
# «зашитых» весовых категорий и оставался гибким в любом году.
SORTS = [
    ("default", "По умолчанию"),
    ("quality", "По точности (выше → ниже)"),
    ("speed", "По скорости (быстрее → медленнее)"),
    ("size", "По размеру (меньше → больше)"),
    ("ram", "По памяти (меньше → больше)"),
    ("params", "По числу параметров (больше → меньше)"),
    ("year", "По году релиза (новые → старые)"),
    ("name", "По названию (А → Я)"),
]


def _range_row(label_text: str, tooltip: str, lo_widget, hi_widget, suffix: str) -> QWidget:
    """Строка фильтра-диапазона: «подпись: [от] — [до]»."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    for widget in (lo_widget, hi_widget):
        widget.setSpecialValueText("—")
        widget.setToolTip(tooltip)
        widget.setSuffix(suffix)
    layout.addWidget(lo_widget)
    dash = QLabel("—")
    dash.setObjectName("Muted")
    layout.addWidget(dash)
    layout.addWidget(hi_widget)
    return row


class ModelListPanel(QFrame):
    """Список моделей с фильтрами. Ничего не качает сам — только сигналы."""

    modelSelected = Signal(object)        # ModelEntry
    downloadRequested = Signal(object)    # ModelEntry
    deleteRequested = Signal(object)
    infoRequested = Signal(object)
    customRequested = Signal()
    refreshRequested = Signal()

    def __init__(
        self,
        catalog: ModelCatalog,
        hardware: Optional[dict] = None,
        settings: Optional[Settings] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SidePanel")
        self.catalog = catalog
        self.hardware = hardware or {}
        self.settings = settings
        self._cards: Dict[str, ModelCard] = {}
        # дебаунс сохранения состояния фильтров в settings.json
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._save_filters)

        self._build_ui()
        self._load_filters()
        self.rebuild()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = make_label("Модели", "H1")
        title.setToolTip("Каталог моделей распознавания: скачанные и доступные для загрузки")
        header.addWidget(title)
        header.addStretch(1)

        self.count_label = make_label("", "Muted")
        header.addWidget(self.count_label)

        # --- переключатель «свернуть/показать фильтры» ----------------------- #
        # По умолчанию фильтры РАСКРЫТЫ. Когда они мешают (хочется видеть
        # больше карточек моделей за раз) — сворачиваются одним кликом;
        # состояние сохраняется в settings.json (catalog_filters.collapsed).
        self.filters_toggle = QPushButton("▾ Фильтры")
        self.filters_toggle.setObjectName("Link")
        self.filters_toggle.setCursor(compat.CURSOR_POINTING_HAND)
        self.filters_toggle.setToolTip(
            "Свернуть/показать панель фильтров каталога.\n"
            "В свёрнутом виде в списке помещается больше моделей."
        )
        self.filters_toggle.clicked.connect(self._toggle_filters)
        header.addWidget(self.filters_toggle)
        root.addLayout(header)

        subtitle = make_label(
            "Скачай модель, затем выбери её — кликом по карточке или в списке "
            "«Модель» в блоке «3 · Работа». Пока модель не выбрана, кнопка "
            "«РАБОТАТЬ» неактивна.",
            "Dim",
            word_wrap=True,
        )
        root.addWidget(subtitle)

        # --- фильтры ------------------------------------------------------ #
        filters = QFrame()
        filters.setObjectName("FiltersFrame")
        self.filters_frame = filters
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

        self.quality_filter = QComboBox()
        self.quality_filter.setToolTip(
            "Минимальная оценка точности (звёзды в карточке).\n"
            "★★★★ и выше — модели, пригодные для конспекта без вычитки."
        )
        self.quality_filter.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self._filter_label("Точность"), row, 2)
        filters_layout.addWidget(self.quality_filter, row, 3)

        # --- числовые диапазоны (гибкие, не привязаны к «весовым категориям») --- #
        row += 1
        self.size_lo = self._spin(0, 1_000_000, " МБ")
        self.size_hi = self._spin(0, 1_000_000, " МБ")
        filters_layout.addWidget(self._filter_label("Размер, МБ"), row, 0)
        filters_layout.addWidget(
            _range_row(
                "Размер", 
                "Диапазон размера файла модели в мегабайтах.\n"
                "«—» = без ограничения. Например: от 500 до 2000.",
                self.size_lo, self.size_hi, " МБ",
            ),
            row, 1, 1, 3,
        )

        row += 1
        self.ram_lo = self._spin(0, 200_000, " МБ")
        self.ram_hi = self._spin(0, 200_000, " МБ")
        filters_layout.addWidget(self._filter_label("Память, МБ"), row, 0)
        filters_layout.addWidget(
            _range_row(
                "Память",
                "Диапазон потребления оперативной памяти в работе.\n"
                "Свою свободную RAM смотри в статус-баре внизу окна.",
                self.ram_lo, self.ram_hi, " МБ",
            ),
            row, 1, 1, 3,
        )

        row += 1
        self.year_lo = self._spin(2015, 2100, "")
        self.year_hi = self._spin(2015, 2100, "")
        filters_layout.addWidget(self._filter_label("Год релиза"), row, 0)
        filters_layout.addWidget(
            _range_row(
                "Год релиза",
                "Диапазон годов выпуска весов модели (любые числа).\n"
                "Модели с неизвестным годом под фильтр не попадут.",
                self.year_lo, self.year_hi, "",
            ),
            row, 1, 1, 3,
        )

        row += 1
        self.params_lo = self._dspin(0.0, 100.0, " млрд")
        self.params_hi = self._dspin(0.0, 100.0, " млрд")
        filters_layout.addWidget(self._filter_label("Параметры"), row, 0)
        filters_layout.addWidget(
            _range_row(
                "Параметры",
                "Диапазон числа параметров модели (в миллиардах).\n"
                "Больше параметров — обычно точнее, но медленнее и прожорливее.",
                self.params_lo, self.params_hi, " млрд",
            ),
            row, 1, 1, 3,
        )

        row += 1
        self.sort_combo = QComboBox()
        self.sort_combo.setToolTip("Порядок карточек в списке")
        self.sort_combo.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self._filter_label("Сортировка"), row, 0)
        filters_layout.addWidget(self.sort_combo, row, 1, 1, 2)

        self.reset_button = QPushButton("Сбросить")
        self.reset_button.setObjectName("Ghost")
        self.reset_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.reset_button.setToolTip("Вернуть все фильтры, поиск и сортировку в исходное состояние")
        self.reset_button.clicked.connect(self.reset_filters_to_defaults)
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

    # -- конструкторы виджетов диапазонов -------------------------------- #
    @staticmethod
    def _spin(minimum: int, maximum: int, suffix: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSuffix(suffix)
        spin.setValue(minimum)          # минимум = specialValueText «—»
        spin.setKeyboardTracking(False)
        spin.setCursor(compat.CURSOR_POINTING_HAND)
        return spin

    @staticmethod
    def _dspin(minimum: float, maximum: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setSuffix(suffix)
        spin.setValue(minimum)
        spin.setKeyboardTracking(False)
        spin.setCursor(compat.CURSOR_POINTING_HAND)
        return spin

    @staticmethod
    def _filter_label(text: str) -> QLabel:
        label = make_label(text, "Muted")
        label.setAlignment(compat.ALIGN_RIGHT_VCENTER)
        label.setToolTip("Фильтр каталога моделей")
        # иначе в узкой панели GridLayout сжимает подпись и она обрезается слева
        label.setMinimumWidth(label.sizeHint().width())
        return label

    # ------------------------------------------------------------------ #
    def minimumSizeHint(self):
        """Панель может сужаться до 340 px: фильтры и карточки переносятся."""
        return QSize(340, 420)

    def _fill_filter_options(self) -> None:
        self.vendor_filter.addItem("Все производители", "")
        vendors = sorted(
            {entry.vendor for entry in self.catalog.all() if entry.vendor},
            key=str.lower,
        )
        for vendor in vendors:
            self.vendor_filter.addItem(vendor, vendor)

        self.quality_filter.addItem("Любая точность", 0)
        for stars in (3, 4, 5):
            self.quality_filter.addItem("★" * stars + " и выше", stars)

        for key, title in SORTS:
            self.sort_combo.addItem(title, key)

        # диапазоны меняются с клавиатуры/стрелок — реагируем на оба события
        for spin in (self.size_lo, self.size_hi, self.ram_lo, self.ram_hi,
                     self.year_lo, self.year_hi):
            spin.valueChanged.connect(self._apply_filters)
        for spin in (self.params_lo, self.params_hi):
            spin.valueChanged.connect(self._apply_filters)

    # ------------------------------------------------------------------ #
    #  Сохранение/восстановление состояния фильтров (settings.json)
    # ------------------------------------------------------------------ #
    def _filters_dict(self) -> dict:
        return {
            "engine": self.engine_filter.currentData() or "",
            "vendor": self.vendor_filter.currentData() or "",
            "sort": self.sort_combo.currentData() or "default",
            "only_installed": self.only_installed.isChecked(),
            "only_russian": self.only_russian.isChecked(),
            "min_quality": int(self.quality_filter.currentData() or 0),
            "size_mb_lo": int(self.size_lo.value()),
            "size_mb_hi": int(self.size_hi.value()),
            "ram_mb_lo": int(self.ram_lo.value()),
            "ram_mb_hi": int(self.ram_hi.value()),
            "year_lo": int(self.year_lo.value()),
            "year_hi": int(self.year_hi.value()),
            "params_lo": float(self.params_lo.value()),
            "params_hi": float(self.params_hi.value()),
            "collapsed": self.filters_frame.isHidden(),
        }

    # ------------------------------------------------------------------ #
    #  Сворачивание панели фильтров
    # ------------------------------------------------------------------ #
    def _active_filter_count(self) -> int:
        """Сколько фильтров отличается от значений по умолчанию."""
        count = 0
        if self.search.text().strip():
            count += 1
        if self.engine_filter.currentData():
            count += 1
        if self.vendor_filter.currentData():
            count += 1
        if int(self.quality_filter.currentData() or 0):
            count += 1
        if (self.sort_combo.currentData() or "default") != "default":
            count += 1
        if self.only_installed.isChecked():
            count += 1
        if not self.only_russian.isChecked():
            count += 1
        for spin in (self.size_lo, self.size_hi, self.ram_lo, self.ram_hi,
                     self.year_lo, self.year_hi, self.params_lo, self.params_hi):
            if spin.value() != spin.minimum():
                count += 1
        return count

    def _apply_filters_visibility(self, visible: bool, save: bool = True) -> None:
        self.filters_frame.setVisible(visible)
        active = self._active_filter_count()
        if visible:
            self.filters_toggle.setText("▾ Фильтры")
            self.filters_toggle.setToolTip(
                "Свернуть панель фильтров каталога.\n"
                "В свёрнутом виде в списке помещается больше моделей."
            )
        else:
            self.filters_toggle.setText(f"▸ Фильтры ({active})" if active else "▸ Фильтры")
            self.filters_toggle.setToolTip(
                "Показать панель фильтров каталога."
                + (f"\nСейчас активно фильтров: {active} — они продолжают работать в свёрнутом виде."
                   if active else "")
            )
        if save:
            self._save_filters()

    def _toggle_filters(self) -> None:
        # isHidden(), а не isVisible(): isVisible() у детей ещё не показанной
        # панели всегда False, и логика переключения переворачивалась
        self._apply_filters_visibility(self.filters_frame.isHidden())

    def _load_filters(self) -> None:
        if self.settings is None:
            return
        state = dict(self.settings.catalog_filters or {})
        collapsed = bool(state.get("collapsed", False))
        self._apply_filters_visibility(not collapsed, save=False)
        for widget, key, setter in (
            (self.engine_filter, "engine", "data"),
            (self.vendor_filter, "vendor", "data"),
            (self.sort_combo, "sort", "data"),
        ):
            value = state.get(key)
            if value:
                index = widget.findData(value)
                if index >= 0:
                    widget.blockSignals(True)
                    widget.setCurrentIndex(index)
                    widget.blockSignals(False)
        for widget, key in (
            (self.only_installed, "only_installed"),
            (self.only_russian, "only_russian"),
        ):
            if key in state:
                widget.blockSignals(True)
                widget.setChecked(bool(state[key]))
                widget.blockSignals(False)
        index = self.quality_filter.findData(int(state.get("min_quality") or 0))
        if index >= 0:
            self.quality_filter.blockSignals(True)
            self.quality_filter.setCurrentIndex(index)
            self.quality_filter.blockSignals(False)
        for widget, key in (
            (self.size_lo, "size_mb_lo"), (self.size_hi, "size_mb_hi"),
            (self.ram_lo, "ram_mb_lo"), (self.ram_hi, "ram_mb_hi"),
            (self.year_lo, "year_lo"), (self.year_hi, "year_hi"),
            (self.params_lo, "params_lo"), (self.params_hi, "params_hi"),
        ):
            if key in state:
                widget.blockSignals(True)
                try:
                    widget.setValue(type(widget.value())(state[key]))
                except (TypeError, ValueError):
                    pass
                widget.blockSignals(False)

    def _save_filters(self) -> None:
        if self.settings is None:
            return
        self.settings.catalog_filters = self._filters_dict()
        self.settings.save()

    def _schedule_save_filters(self) -> None:
        self._save_timer.start()

    def reset_filters_to_defaults(self) -> None:
        """Сброс фильтров (кнопка «Сбросить» и общий сброс настроек)."""
        for widget in (
            self.engine_filter, self.vendor_filter, self.quality_filter, self.sort_combo,
        ):
            widget.blockSignals(True)
            widget.setCurrentIndex(0)
            widget.blockSignals(False)
        for widget in (
            self.size_lo, self.size_hi, self.ram_lo, self.ram_hi,
            self.year_lo, self.year_hi, self.params_lo, self.params_hi,
        ):
            widget.blockSignals(True)
            widget.setValue(widget.minimum())
            widget.blockSignals(False)
        self.only_installed.blockSignals(True)
        self.only_installed.setChecked(False)
        self.only_installed.blockSignals(False)
        self.only_russian.blockSignals(True)
        self.only_russian.setChecked(True)
        self.only_russian.blockSignals(False)
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self._apply_filters()
        self._save_filters()

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
        log.debug("Панель моделей пересобрана: карточек %d, установлено %d", len(entries), installed_count)
        self._apply_filters()

    def card(self, model_id: str) -> Optional[ModelCard]:
        return self._cards.get(model_id)

    def set_selected(self, model_id: str) -> None:
        for card_id, card in self._cards.items():
            card.set_selected(card_id == model_id)

    def set_hardware(self, hardware: dict) -> None:
        self.hardware = hardware or {}

    # ------------------------------------------------------------------ #
    def _sort_key(self, entry: ModelEntry):
        mode = self.sort_combo.currentData() or "default"
        if mode == "quality":
            return (-entry.quality, -entry.speed, entry.size_bytes or 0)
        if mode == "size":
            return (entry.size_bytes or 0, -entry.quality)
        if mode == "ram":
            return (entry.ram_mb or 0, -entry.quality)
        if mode == "params":
            return (-(entry.params_b or 0.0), -entry.quality)
        if mode == "year":
            return (-entry.released_year, -entry.quality)
        if mode == "speed":
            return (-entry.speed, -entry.quality)
        if mode == "name":
            return (entry.name.lower(),)
        return (
            0 if entry.installed else 1,
            0 if entry.supports_russian else 1,
            -entry.quality,
            entry.ram_mb or 99_999,
        )

    def _apply_filters(self, *_args) -> None:
        self._schedule_save_filters()

        query = self.search.text().strip().lower()
        engine = self.engine_filter.currentData() or ""
        vendor = self.vendor_filter.currentData() or ""
        min_quality = int(self.quality_filter.currentData() or 0)
        installed_only = self.only_installed.isChecked()
        russian_only = self.only_russian.isChecked()

        size_lo_mb = int(self.size_lo.value())
        size_hi_mb = int(self.size_hi.value())
        ram_lo_mb = int(self.ram_lo.value())
        ram_hi_mb = int(self.ram_hi.value())
        year_lo = int(self.year_lo.value())
        year_hi = int(self.year_hi.value())
        params_lo = float(self.params_lo.value())
        params_hi = float(self.params_hi.value())
        # у года минимум 2015 — значение «2015..2100» трактуем как диапазон;
        # если lo == minimum spinbox'а и hi == minimum — фильтр выключен
        year_active = not (year_lo == self.year_lo.minimum() and year_hi == self.year_lo.minimum())

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
            if show and size_lo_mb and (entry.size_bytes or 0) < size_lo_mb * 1024 ** 2:
                show = False
            if show and size_hi_mb and (entry.size_bytes or 0) > size_hi_mb * 1024 ** 2:
                show = False
            if show and ram_lo_mb and (entry.ram_mb or 0) < ram_lo_mb:
                show = False
            if show and ram_hi_mb and (entry.ram_mb or 0) > ram_hi_mb:
                show = False
            if show and year_active:
                year = entry.released_year or 0
                if not year:
                    show = False       # год неизвестен — под диапазон не попадает
                else:
                    lo = year_lo if year_lo > self.year_lo.minimum() else 0
                    hi = year_hi if year_hi > self.year_lo.minimum() else 9999
                    if not (lo <= year <= hi):
                        show = False
            if show and params_lo and (entry.params_b or 0.0) < params_lo:
                show = False
            if show and params_hi and (entry.params_b or 0.0) > params_hi:
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
                        entry.quantization or "",
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
                "Попробуй ослабить условия (например, снять галочку «С русским языком»\n"
                "или обнулить диапазоны) либо нажать «Сбросить»."
            )
        # свёрнутая панель фильтров: держим счётчик активных фильтров на кнопке
        if self.filters_frame.isHidden():
            self._apply_filters_visibility(False, save=False)
        log.debug(
            "Фильтры каталога: видно %d из %d | состояние=%s",
            visible, len(self._cards), self._filters_dict(),
        )
