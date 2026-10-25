"""Models panel: search, filters (including numeric ranges), sorting, cards."""

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
from ..i18n import _
from ..logutils import get_logger
from ..settings import Settings
from . import compat
from .widgets import ModelCard, make_label

log = get_logger("ui.models")

# Sort modes: (key, label). Numeric sorts support range filters so the list
# never depends on hard-coded weight categories.
SORTS = [
    ("default", "Default"),
    ("quality", "By accuracy (high → low)"),
    ("speed", "By speed (fast → slow)"),
    ("size", "By size (small → large)"),
    ("ram", "By memory (low → high)"),
    ("params", "By parameter count (high → low)"),
    ("year", "By release year (new → old)"),
    ("name", "By name (A → Z)"),
]


def _range_row(label_text: str, tooltip: str, lo_widget, hi_widget, suffix: str) -> QWidget:
    """Range filter row: "label: [from] — [to]"."""
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
    """Model list with filters. Downloads nothing itself — signals only."""

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
        # debounce saving the filter state into settings.json
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
        title = make_label(_("Models"), "H1")
        title.setToolTip(_("Recognition model catalog: downloaded and available for download"))
        header.addWidget(title)
        header.addStretch(1)

        self.count_label = make_label("", "Muted")
        header.addWidget(self.count_label)

        # --- collapse/show filters toggle ------------------------------------- #
        # Filters are EXPANDED by default; one click collapses them so more
        # cards fit. The state is saved in settings.json (catalog_filters.collapsed).
        self.filters_toggle = QPushButton(_("▾ Filters"))
        self.filters_toggle.setObjectName("Link")
        self.filters_toggle.setCursor(compat.CURSOR_POINTING_HAND)
        self.filters_toggle.setToolTip(_(
            "Collapse/show the catalog filter panel.\n"
            "Collapsed: more models fit in the list."
        ))
        self.filters_toggle.clicked.connect(self._toggle_filters)
        header.addWidget(self.filters_toggle)
        root.addLayout(header)

        subtitle = make_label(
            _("Download a model, then select it — by clicking its card or in the "
              "\"Model\" list of block \"3 · Work\". Until a model is selected, the "
              "RUN button is inactive."),
            "Dim",
            word_wrap=True,
        )
        root.addWidget(subtitle)

        # --- filters ----------------------------------------------------------- #
        filters = QFrame()
        filters.setObjectName("FiltersFrame")
        self.filters_frame = filters
        filters_layout = QGridLayout(filters)
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.setSpacing(8)
        filters_layout.setHorizontalSpacing(10)

        row = 0
        self.search = QLineEdit()
        self.search.setPlaceholderText(_("Search: name, tag, language, description…"))
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip(_("Searches model name, ID, description, tags and languages"))
        self.search.textChanged.connect(self._apply_filters)
        filters_layout.addWidget(self.search, row, 0, 1, 2)

        self.engine_filter = QComboBox()
        self.engine_filter.addItem(_("All engines"), "")
        for engine in ENGINES:
            self.engine_filter.addItem(_(ENGINE_LABELS[engine]), engine)
        self.engine_filter.setToolTip(_(
            "Engine = what computes the model:\n"
            "• faster-whisper — fastest on CPU/GPU (primary);\n"
            "• whisper.cpp — minimal RAM;\n"
            "• Vosk — instant start, simpler quality;\n"
            "• Qwen3-ASR / GigaAM — new-generation torch models (installed on demand)."
        ))
        self.engine_filter.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self.engine_filter, row, 2, 1, 2)

        row += 1
        self.vendor_filter = QComboBox()
        self.vendor_filter.setToolTip(_(
            "Who made the model: OpenAI (Whisper), Qwen (Alibaba), Sber (GigaAM), "
            "community fine-tunes etc."
        ))
        self.vendor_filter.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self._filter_label(_("Vendor")), row, 0)
        filters_layout.addWidget(self.vendor_filter, row, 1)

        self.quality_filter = QComboBox()
        self.quality_filter.setToolTip(_(
            "Minimum accuracy rating (stars on the card).\n"
            "★★★★ and above — models good enough for notes without proofreading."
        ))
        self.quality_filter.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self._filter_label(_("Accuracy")), row, 2)
        filters_layout.addWidget(self.quality_filter, row, 3)

        # --- numeric ranges (flexible, not tied to weight categories) ----------- #
        row += 1
        self.size_lo = self._spin(0, 1_000_000, _(" MB"))
        self.size_hi = self._spin(0, 1_000_000, _(" MB"))
        filters_layout.addWidget(self._filter_label(_("Size, MB")), row, 0)
        filters_layout.addWidget(
            _range_row(
                _("Size"),
                _("Model file size range in megabytes.\n"
                  "\"—\" = no limit. Example: from 500 to 2000."),
                self.size_lo, self.size_hi, _(" MB"),
            ),
            row, 1, 1, 3,
        )

        row += 1
        self.ram_lo = self._spin(0, 200_000, _(" MB"))
        self.ram_hi = self._spin(0, 200_000, _(" MB"))
        filters_layout.addWidget(self._filter_label(_("Memory, MB")), row, 0)
        filters_layout.addWidget(
            _range_row(
                _("Memory"),
                _("Runtime RAM usage range.\n"
                  "See your free RAM in the status bar at the bottom."),
                self.ram_lo, self.ram_hi, _(" MB"),
            ),
            row, 1, 1, 3,
        )

        row += 1
        self.year_lo = self._spin(2015, 2100, "")
        self.year_hi = self._spin(2015, 2100, "")
        filters_layout.addWidget(self._filter_label(_("Release year")), row, 0)
        filters_layout.addWidget(
            _range_row(
                _("Release year"),
                _("Model weight release year range (any numbers).\n"
                  "Models with an unknown year will not match the filter."),
                self.year_lo, self.year_hi, "",
            ),
            row, 1, 1, 3,
        )

        row += 1
        self.params_lo = self._dspin(0.0, 100.0, _(" B"))
        self.params_hi = self._dspin(0.0, 100.0, _(" B"))
        filters_layout.addWidget(self._filter_label(_("Parameters")), row, 0)
        filters_layout.addWidget(
            _range_row(
                _("Parameters"),
                _("Model parameter count range (in billions).\n"
                  "More parameters usually means more accurate but slower and hungrier."),
                self.params_lo, self.params_hi, _(" B"),
            ),
            row, 1, 1, 3,
        )

        row += 1
        self.sort_combo = QComboBox()
        self.sort_combo.setToolTip(_("Card order in the list"))
        self.sort_combo.currentIndexChanged.connect(self._apply_filters)
        filters_layout.addWidget(self._filter_label(_("Sorting")), row, 0)
        filters_layout.addWidget(self.sort_combo, row, 1, 1, 2)

        self.reset_button = QPushButton(_("Reset"))
        self.reset_button.setObjectName("Ghost")
        self.reset_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.reset_button.setToolTip(_("Return all filters, search and sorting to the initial state"))
        self.reset_button.clicked.connect(self.reset_filters_to_defaults)
        filters_layout.addWidget(self.reset_button, row, 3)

        row += 1
        toggles = QHBoxLayout()
        toggles.setSpacing(14)
        self.only_installed = QCheckBox(_("Installed only"))
        self.only_installed.setToolTip(_("Show models already downloaded into models/"))
        self.only_installed.toggled.connect(self._apply_filters)
        self.only_russian = QCheckBox(_("Supports Russian"))
        self.only_russian.setChecked(True)
        self.only_russian.setToolTip(_("Keep models that understand Russian speech (uncheck to see EN-only ones too)"))
        self.only_russian.toggled.connect(self._apply_filters)
        toggles.addWidget(self.only_installed)
        toggles.addWidget(self.only_russian)
        toggles.addStretch(1)
        filters_layout.addLayout(toggles, row, 0, 1, 4)

        root.addWidget(filters)

        # --- buttons ------------------------------------------------------------- #
        actions = QHBoxLayout()
        actions.setSpacing(8)

        self.custom_button = QPushButton(_("＋ Custom model (link / path)"))
        self.custom_button.setObjectName("Ghost")
        self.custom_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.custom_button.setToolTip(_(
            "Add a model not in the catalog: an HF repo, a direct link or a folder on disk"
        ))
        self.custom_button.clicked.connect(self.customRequested)
        actions.addWidget(self.custom_button, stretch=1)

        self.refresh_button = QPushButton(_("⟳ Refresh catalog"))
        self.refresh_button.setObjectName("Ghost")
        self.refresh_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.refresh_button.setToolTip(_("Check the online catalog for new models"))
        self.refresh_button.clicked.connect(self.refreshRequested)
        actions.addWidget(self.refresh_button)
        root.addLayout(actions)

        # --- card scroller ---------------------------------------------------------- #
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

    # -- range widget constructors ------------------------------------------- #
    @staticmethod
    def _spin(minimum: int, maximum: int, suffix: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSuffix(suffix)
        spin.setValue(minimum)          # minimum = specialValueText "—"
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
        label.setToolTip(_("Model catalog filter"))
        # otherwise the GridLayout squeezes the label in a narrow panel
        label.setMinimumWidth(label.sizeHint().width())
        return label

    # ------------------------------------------------------------------ #
    def minimumSizeHint(self):
        """The panel can shrink to 340 px: filters and cards wrap."""
        return QSize(340, 420)

    def _fill_filter_options(self) -> None:
        self.vendor_filter.addItem(_("All vendors"), "")
        vendors = sorted(
            {entry.vendor for entry in self.catalog.all() if entry.vendor},
            key=str.lower,
        )
        for vendor in vendors:
            self.vendor_filter.addItem(_(vendor), vendor)

        self.quality_filter.addItem(_("Any accuracy"), 0)
        for stars in (3, 4, 5):
            self.quality_filter.addItem(_("{stars} and above").format(stars="★" * stars), stars)

        for key, title in SORTS:
            self.sort_combo.addItem(_(title), key)

        # ranges change via keyboard/arrows — react to both events
        for spin in (self.size_lo, self.size_hi, self.ram_lo, self.ram_hi,
                     self.year_lo, self.year_hi):
            spin.valueChanged.connect(self._apply_filters)
        for spin in (self.params_lo, self.params_hi):
            spin.valueChanged.connect(self._apply_filters)

    # ------------------------------------------------------------------ #
    #  Filter state save/restore (settings.json)
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
    #  Filter panel collapsing
    # ------------------------------------------------------------------ #
    def _active_filter_count(self) -> int:
        """How many filters differ from their defaults."""
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
            self.filters_toggle.setText(_("▾ Filters"))
            self.filters_toggle.setToolTip(_(
                "Collapse the catalog filter panel.\n"
                "Collapsed: more models fit in the list."
            ))
        else:
            self.filters_toggle.setText(
                _("▸ Filters ({n})").format(n=active) if active else _("▸ Filters"))
            self.filters_toggle.setToolTip(
                _("Show the catalog filter panel.")
                + (_("\nActive filters: {n} — they keep working while collapsed.").format(n=active)
                   if active else "")
            )
        if save:
            self._save_filters()

    def _toggle_filters(self) -> None:
        # isHidden(), not isVisible(): a child of a not-yet-shown panel always
        # reports isVisible() == False, which inverted the toggle logic
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
        """Reset filters (the Reset button and the global settings reset)."""
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
        """Fully rebuild the card list (after install/delete)."""
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

        self.count_label.setText(_("{total} in catalog · {installed} installed").format(
            total=len(entries), installed=installed_count))
        log.debug("Models panel rebuilt: %d cards, %d installed", len(entries), installed_count)
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
        # the year spin minimum is 2015 — "2015..2100" means the filter is off
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
                    show = False       # unknown year — does not match a range
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
                        entry.display_name,
                        entry.id,
                        entry.engine,
                        entry.vendor,
                        entry.description,
                        _(entry.description),
                        entry.languages_text,
                        entry.quantization or "",
                        " ".join(entry.tags),
                        " ".join(_(tag) for tag in entry.tags),
                        " ".join(entry.languages),
                    ]
                ).lower()
                show = all(part in haystack for part in query.split())
            card.setVisible(show)
            if show:
                visible_entries.append(entry)

        # reorder cards per the sort mode (without recreating them)
        ordered = sorted(visible_entries, key=self._sort_key)
        for index, entry in enumerate(ordered):
            card = self._cards[entry.id]
            self.container_layout.removeWidget(card)
            self.container_layout.insertWidget(index, card)

        visible = len(ordered)
        self.empty_label.setVisible(visible == 0)
        if visible == 0:
            self.empty_label.setText(_(
                "Nothing matched these filters.\n"
                "Try loosening the conditions (e.g. uncheck \"Supports Russian\" "
                "or zero the ranges) or press \"Reset\"."
            ))
        # collapsed filter panel: keep the active-filter counter on the button
        if self.filters_frame.isHidden():
            self._apply_filters_visibility(False, save=False)
        log.debug(
            "Catalog filters: %d of %d visible | state=%s",
            visible, len(self._cards), self._filters_dict(),
        )
