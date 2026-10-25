"""Reusable widgets: file list, model card, helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import paths
from ..constants import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from ..core.models_catalog import ModelEntry
from ..i18n import _
from . import compat


def make_label(text: str, object_name: str = "", word_wrap: bool = False) -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    label.setWordWrap(word_wrap)
    return label


def open_in_file_manager(path: Path | str) -> None:
    """Open a folder in the system file manager (or reveal a file)."""
    path = Path(path)
    target = path if path.is_dir() else path.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


class ToolTipMenu(QMenu):
    """Menu whose item tooltips appear on hover.

    Stock Qt shows menu item tooltips only on some platforms/styles — this
    makes tooltips work everywhere.
    """

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        super().mouseMoveEvent(event)
        try:
            from PySide6.QtWidgets import QToolTip

            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            action = self.actionAt(pos)
            tip = action.toolTip() if action is not None else ""
            if tip:
                global_pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
                QToolTip.showText(global_pos, tip, self)
            else:
                QToolTip.hideText()
        except (RuntimeError, AttributeError):
            pass


def open_url(url: str) -> None:
    if url:
        QDesktopServices.openUrl(QUrl(url))


def stars(value: int, maximum: int = 5) -> str:
    value = max(0, min(maximum, int(value or 0)))
    return "★" * value + "☆" * (maximum - value)


def chip(text: str, object_name: str = "BadgeEngine") -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    label.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
    return label


class FlowLayout(QLayout):
    """Line-wrapping layout (like CSS flex-wrap).

    Keeps chips/checkboxes/buttons from being clipped in a narrow panel.
    """

    def __init__(self, parent: Optional[QWidget] = None, margin: int = 0, spacing: int = 6) -> None:
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing
        self._items: list = []

    # -- required QLayout overrides ------------------------------------------ #
    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:  # noqa: D102
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    # -- own logic ------------------------------------------------------------- #
    def spacing_for(self) -> int:
        if self._spacing >= 0:
            return self._spacing
        return 6

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        left, top, right, bottom = self.getContentsMargins()
        effective = rect.adjusted(left, top, -right, -bottom)
        x = effective.x()
        y = effective.y()
        line_height = 0
        space = self.spacing_for()

        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            hint = item.sizeHint()
            width = min(hint.width(), max(0, effective.width()))
            if item.hasHeightForWidth():
                height = item.heightForWidth(width)
            else:
                height = hint.height()
            next_x = x + width + space
            if next_x - space > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + space
                next_x = x + width + space
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), QSize(width, height)))
            x = next_x
            line_height = max(line_height, height)
        return y + line_height - rect.y() + bottom


# --------------------------------------------------------------------------- #
#  File list
# --------------------------------------------------------------------------- #


class FileListWidget(QListWidget):
    """Media file list with drag&drop and a per-row remove button."""

    filesChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setUniformItemSizes(False)
        self.setSpacing(4)
        self.setHorizontalScrollBarPolicy(compat.SCROLLBAR_ALWAYS_OFF)
        self._known: set[str] = set()

    # -- public API ------------------------------------------------------------ #
    def add_paths(self, items: Iterable[str]) -> int:
        """Add files and folders (folders expand recursively). Returns new count."""
        expanded = paths.expand_media_inputs(items)
        added = 0
        for path in expanded:
            key = str(path.resolve()).lower()
            if key in self._known:
                continue
            self._known.add(key)
            self._append_row(path)
            added += 1
        if added:
            self.filesChanged.emit()
        return added

    def files(self) -> List[Path]:
        result: List[Path] = []
        for index in range(self.count()):
            item = self.item(index)
            data = item.data(compat.ROLE_USER)
            if data:
                result.append(Path(data))
        return result

    def total_size(self) -> int:
        total = 0
        for path in self.files():
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def clear_all(self) -> None:
        self._known.clear()
        self.clear()
        self.filesChanged.emit()

    def remove_selected(self) -> None:
        for item in self.selectedItems():
            data = item.data(compat.ROLE_USER)
            if data:
                self._known.discard(str(Path(data).resolve()).lower())
            self.takeItem(self.row(item))
        self.filesChanged.emit()

    # -- internal ---------------------------------------------------------------- #
    def _append_row(self, path: Path) -> None:
        item = QListWidgetItem()
        item.setData(compat.ROLE_USER, str(path))
        item.setToolTip(str(path))
        widget = FileRow(path)
        widget.removeRequested.connect(lambda p=path: self._remove_one(p))
        item.setSizeHint(widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, widget)
        # Qt sometimes keeps a stale row height (texts overlap) — force-sync
        widget.setMinimumHeight(widget.sizeHint().height())
        QTimer.singleShot(0, lambda: self._sync_row(item, widget))

    def _sync_row(self, item: QListWidgetItem, widget: QWidget) -> None:
        if item is None or widget is None:
            return
        try:
            if self.row(item) < 0:
                return
            widget.setMinimumHeight(widget.sizeHint().height())
            item.setSizeHint(widget.sizeHint())
            rect = self.visualItemRect(item)
            if widget.geometry() != rect:
                widget.setGeometry(rect)
        except RuntimeError:
            pass

    def _sync_all_rows(self) -> None:
        for index in range(self.count()):
            item = self.item(index)
            widget = self.itemWidget(item)
            if widget is not None:
                self._sync_row(item, widget)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_all_rows)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_all_rows)

    def _remove_one(self, path: Path) -> None:
        key = str(path.resolve()).lower()
        for index in range(self.count()):
            item = self.item(index)
            if str(item.data(compat.ROLE_USER) or "").lower() == key or str(
                Path(str(item.data(compat.ROLE_USER))).resolve()
            ).lower() == key:
                self.takeItem(index)
                break
        self._known.discard(key)
        self.filesChanged.emit()

    # -- drag & drop --------------------------------------------------------------- #
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._urls_from(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._urls_from(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = self._urls_from(event)
        if urls:
            self.add_paths(urls)
            event.acceptProposedAction()

    @staticmethod
    def _urls_from(event) -> List[str]:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return []
        return [url.toLocalFile() for url in mime.urls() if url.toLocalFile()]


class FileRow(QFrame):
    """List row: icon, name, path, size, remove button."""

    removeRequested = Signal(object)

    def __init__(self, path: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.path = Path(path)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 8, 7)
        layout.setSpacing(10)

        kind = self._kind_icon()
        icon = QLabel(kind)
        icon.setFixedWidth(22)
        icon.setAlignment(compat.ALIGN_CENTER)
        icon.setStyleSheet("font-size: 15px;")
        layout.addWidget(icon)

        texts = QVBoxLayout()
        texts.setSpacing(1)
        name = QLabel(self.path.name)
        name.setObjectName("H3")
        name.setTextInteractionFlags(compat.TEXT_SELECTABLE_BY_MOUSE)
        subtitle = QLabel(paths.short_path(self.path.parent, 68))
        subtitle.setObjectName("Muted")
        texts.addWidget(name)
        texts.addWidget(subtitle)
        layout.addLayout(texts, stretch=1)

        size_label = QLabel(self._size_text())
        size_label.setObjectName("Dim")
        size_label.setAlignment(compat.ALIGN_RIGHT_VCENTER)
        layout.addWidget(size_label)

        remove = QToolButton()
        remove.setText("✕")
        remove.setToolTip(_("Remove from the list"))
        remove.setCursor(compat.CURSOR_POINTING_HAND)
        remove.setAutoRaise(True)
        remove.setStyleSheet(
            "QToolButton { color: #6b7688; border: none; font-size: 13px; padding: 2px 6px; }"
            "QToolButton:hover { color: #f2637f; }"
        )
        remove.clicked.connect(lambda: self.removeRequested.emit(self.path))
        layout.addWidget(remove)

    def _kind_icon(self) -> str:
        suffix = self.path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return "🎬"
        if suffix in AUDIO_EXTENSIONS:
            return "🎧"
        return "📄"

    def _size_text(self) -> str:
        try:
            size = self.path.stat().st_size
        except OSError:
            return "—"
        return paths.human_size(size)


# --------------------------------------------------------------------------- #
#  Model card
# --------------------------------------------------------------------------- #


class ModelCard(QFrame):
    """Card of one model: name, size, memory, languages, buttons."""

    downloadRequested = Signal(object)      # ModelEntry
    selectRequested = Signal(object)
    deleteRequested = Signal(object)
    infoRequested = Signal(object)

    def __init__(self, entry: ModelEntry, hardware: Optional[dict] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.hardware = hardware or {}
        self.setObjectName("Card")
        self.setProperty("selected", False)
        self.setCursor(compat.CURSOR_POINTING_HAND)
        self._build()
        self._update_card_tooltip()

    def _update_card_tooltip(self) -> None:
        if self.entry.installed:
            self.setToolTip(_(
                "Click the card (or \"Select for work\") — use this model for the RUN "
                "button. The selected model is shown in the status bar below."
            ))
        else:
            self.setToolTip(_(
                "Click the card for model details. "
                "The Download button saves the model into models/."
            ))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Click = select an installed model (or open info)."""
        if self.entry.installed:
            self.selectRequested.emit(self.entry)
        else:
            self.infoRequested.emit(self.entry)
        super().mousePressEvent(event)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(7)

        # row 1: name + badges
        top = QHBoxLayout()
        top.setSpacing(8)
        self.title = QLabel(self.entry.display_name)
        self.title.setObjectName("H2")
        self.title.setWordWrap(True)
        top.addWidget(self.title, stretch=1)

        if self.entry.installed:
            top.addWidget(chip(_("✓ installed"), "BadgeInstalled"))
        root.addLayout(top)

        # row 2: feature chips (wrap instead of clipping)
        chips = FlowLayout(spacing=6)
        chips.addWidget(chip(_(self.entry.engine_label)))
        chips.addWidget(chip(f"⬇ {self.entry.size_text}", "BadgeEngine"))
        memory = self.entry.memory_text
        if memory != "—":
            chips.addWidget(chip(f"🧠 {memory}", "BadgeEngine"))
        if self.entry.params_b:
            params_chip = chip(_("⚖ {} parameters").format(self.entry.params_text), "BadgeEngine")
            params_chip.setToolTip(_("Number of trainable model parameters"))
            chips.addWidget(params_chip)
        if self.entry.quantization:
            quant_chip = chip(f"🗜 {self.entry.quantization_text}", "BadgeEngine")
            quant_chip.setToolTip(_(
                "Weight storage format (quantization).\n"
                "int8/q5_0 — smaller and faster, slightly less accurate;\n"
                "float16/bfloat16 — full quality format."
            ))
            chips.addWidget(quant_chip)
        if self.entry.released_year:
            year_chip = chip(f"📅 {self.entry.released_year}", "BadgeEngine")
            year_chip.setToolTip(_("Model weight release year"))
            chips.addWidget(year_chip)
        for tag in self.entry.tags[:3]:
            chips.addWidget(chip(_(tag), "BadgeEngine"))
        root.addLayout(chips)

        # row 3: languages + quality/speed
        meta = FlowLayout(spacing=12)
        languages = QLabel(f"🌐 {self.entry.languages_text}")
        languages.setObjectName("Dim")
        languages.setWordWrap(True)
        meta.addWidget(languages)

        quality = QLabel(_("<span style='color:#98a2b3'>accuracy</span> <span style='color:#f2b544'>{}</span>").format(stars(self.entry.quality)))
        speed = QLabel(_("<span style='color:#98a2b3'>speed</span> <span style='color:#3ecf8e'>{}</span>").format(stars(self.entry.speed)))
        meta.addWidget(quality)
        meta.addWidget(speed)
        root.addLayout(meta)

        # row 4: memory warning
        warning = self._memory_warning()
        if warning:
            self.warn_label = chip(warning, "BadgeWarn")
            self.warn_label.setWordWrap(True)
            root.addWidget(self.warn_label)

        # row 5: short description
        self.description = QLabel(self._short_description())
        self.description.setObjectName("Dim")
        self.description.setWordWrap(True)
        self.description.setTextFormat(compat.TEXT_FORMAT_RICH)
        root.addWidget(self.description)

        # row 6: download progress
        self.progress = QProgressBar()
        self.progress.setObjectName("Subtle")
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(6)
        root.addWidget(self.progress)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("Muted")
        self.progress_label.setVisible(False)
        self.progress_label.setWordWrap(True)
        root.addWidget(self.progress_label)

        # row 7: buttons
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)

        self.info_button = QPushButton("ℹ")
        self.info_button.setObjectName("InfoButton")
        self.info_button.setFixedSize(30, 30)
        self.info_button.setToolTip(_(
            "Model details: full description, source, license, languages, parameters"
        ))
        self.info_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.info_button.clicked.connect(lambda: self.infoRequested.emit(self.entry))
        buttons.addWidget(self.info_button)

        self.delete_button = QPushButton(_("Delete"))
        self.delete_button.setObjectName("Danger")
        self.delete_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.delete_button.setToolTip(_("Delete the model from disk (free space)"))
        self.delete_button.clicked.connect(lambda: self.deleteRequested.emit(self.entry))
        self.delete_button.setVisible(self.entry.installed)
        buttons.addWidget(self.delete_button)

        self.action_button = QPushButton(_("Select for work") if self.entry.installed else _("Download"))
        self.action_button.setObjectName("Primary" if not self.entry.installed else "")
        self.action_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.action_button.setToolTip(
            _("Download the model into models/") if not self.entry.installed
            else _("Use exactly this model for recognition (it appears in the status bar)")
        )
        self.action_button.clicked.connect(self._on_action)
        buttons.addWidget(self.action_button)
        buttons.addStretch(1)

        root.addLayout(buttons)

    # ------------------------------------------------------------------ #
    def _short_description(self) -> str:
        text = _(self.entry.description or "")
        if len(text) <= 190:
            return text
        cut = text[:190]
        space = cut.rfind(" ")
        if space > 120:
            cut = cut[:space]
        return cut.rstrip(",.;:") + "…"

    def _memory_warning(self) -> str:
        if not self.entry.ram_mb or not self.hardware:
            return ""
        available = int(self.hardware.get("ram_available_mb") or 0)
        if available and self.entry.ram_mb > available:
            return _("⚠ needs ~{need:.1f} GB of memory, only ~{free:.1f} GB free — "
                     "expect slowdowns or an error").format(
                need=self.entry.ram_mb / 1024, free=available / 1024)
        if self.entry.vram_mb and self.hardware.get("cuda_ok"):
            vram = int(self.hardware.get("vram_mb") or 0)
            if vram and self.entry.vram_mb > vram * 0.9:
                return _("⚠ VRAM {vram:.1f} GB — the model will run on CPU").format(vram=vram / 1024)
        return ""

    def _on_action(self) -> None:
        if self.entry.installed:
            self.selectRequested.emit(self.entry)
        else:
            self.downloadRequested.emit(self.entry)

    # ------------------------------------------------------------------ #
    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)
        if self.entry.installed:
            self.action_button.setText(_("✓ Selected") if selected else _("Select for work"))
            self.action_button.setObjectName("Primary" if selected else "")
            self.action_button.style().unpolish(self.action_button)
            self.action_button.style().polish(self.action_button)

    def set_downloading(self, downloading: bool) -> None:
        self.progress.setVisible(downloading)
        self.progress_label.setVisible(downloading)
        self.action_button.setEnabled(not downloading)
        self.delete_button.setEnabled(not downloading)
        if downloading:
            self.progress.setRange(0, 0)
            self.action_button.setText(_("Downloading…"))
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(0)
            self.action_button.setText(
                _("Download") if not self.entry.installed else _("Select for work")
            )
            self.action_button.setEnabled(True)
            self.delete_button.setEnabled(True)

    def update_progress(self, fraction: float, status: str) -> None:
        if not self.progress.isVisible():
            self.progress.setVisible(True)
            self.progress_label.setVisible(True)
        if fraction <= 0:
            self.progress.setRange(0, 0)  # busy indicator, size unknown
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        self.progress_label.setText(status)
