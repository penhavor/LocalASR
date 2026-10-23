"""Переиспользуемые виджеты: список файлов, карточка модели, диалог прогресса."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import paths
from ..constants import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from ..core.models_catalog import ModelEntry
from . import compat


# --------------------------------------------------------------------------- #
#  Мелочи
# --------------------------------------------------------------------------- #


def make_label(text: str, object_name: str = "", word_wrap: bool = False) -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    label.setWordWrap(word_wrap)
    return label


def elide(text: str, width: int, font: Optional[QFont] = None) -> str:
    metrics = QFontMetrics(font or QFont())
    return metrics.elidedText(text, compat.elide_mode(), width)


def open_in_file_manager(path: Path | str) -> None:
    """Открывает папку в системном проводнике (или выделяет файл)."""
    path = Path(path)
    target = path if path.is_dir() else path.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


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


def hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #2a3244; background-color: #2a3244; max-height: 1px;")
    return line


# --------------------------------------------------------------------------- #
#  Список файлов
# --------------------------------------------------------------------------- #


class FileListWidget(QListWidget):
    """Список медиафайлов с drag&drop и кнопкой удаления у каждой строки."""

    filesChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setUniformItemSizes(False)
        self.setSpacing(4)
        self.setHorizontalScrollBarPolicy(compat.SCROLLBAR_ALWAYS_OFF)
        self._known: set[str] = set()

    # -- публичное API ---------------------------------------------------- #
    def add_paths(self, items: Iterable[str]) -> int:
        """Добавляет файлы и папки (папки раскрываются рекурсивно). Возвращает кол-во новых."""
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

    def add_files(self, file_list: Iterable[str]) -> int:
        return self.add_paths(file_list)

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

    # -- внутреннее ------------------------------------------------------- #
    def _append_row(self, path: Path) -> None:
        item = QListWidgetItem()
        item.setData(compat.ROLE_USER, str(path))
        item.setToolTip(str(path))
        widget = FileRow(path)
        widget.removeRequested.connect(lambda p=path: self._remove_one(p))
        item.setSizeHint(widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, widget)

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

    # -- drag & drop ------------------------------------------------------ #
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
    """Строка списка: иконка, имя, путь, размер/длительность, крестик."""

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
        remove.setToolTip("Убрать из списка")
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
#  Карточка модели
# --------------------------------------------------------------------------- #


class ModelCard(QFrame):
    """Карточка одной модели: название, размер, память, языки, кнопки."""

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

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(7)

        # --- строка 1: название + бейджи
        top = QHBoxLayout()
        top.setSpacing(8)
        self.title = QLabel(self.entry.name)
        self.title.setObjectName("H2")
        self.title.setWordWrap(True)
        top.addWidget(self.title, stretch=1)

        if self.entry.recommended:
            top.addWidget(chip("★ рекомендуем", "BadgeRecommended"))
        if self.entry.installed:
            top.addWidget(chip("✓ установлена", "BadgeInstalled"))
        root.addLayout(top)

        # --- строка 2: чипы характеристик
        chips = QHBoxLayout()
        chips.setSpacing(6)
        chips.addWidget(chip(self.entry.engine_label))
        chips.addWidget(chip(f"⬇ {self.entry.size_text}", "BadgeEngine"))
        memory = self.entry.memory_text
        if memory != "—":
            chips.addWidget(chip(f"🧠 {memory}", "BadgeEngine"))
        for tag in self.entry.tags[:3]:
            chips.addWidget(chip(tag, "BadgeEngine"))
        chips.addStretch(1)
        root.addLayout(chips)

        # --- строка 3: языки + качество/скорость
        meta = QHBoxLayout()
        meta.setSpacing(12)
        languages = QLabel(f"🌐 {self.entry.languages_text}")
        languages.setObjectName("Dim")
        languages.setWordWrap(True)
        meta.addWidget(languages, stretch=1)

        quality = QLabel(f"<span style='color:#98a2b3'>точность</span> <span style='color:#f2b544'>{stars(self.entry.quality)}</span>")
        speed = QLabel(f"<span style='color:#98a2b3'>скорость</span> <span style='color:#3ecf8e'>{stars(self.entry.speed)}</span>")
        meta.addWidget(quality)
        meta.addWidget(speed)
        root.addLayout(meta)

        # --- строка 4: предупреждение про память
        warning = self._memory_warning()
        if warning:
            self.warn_label = chip(warning, "BadgeWarn")
            self.warn_label.setWordWrap(True)
            root.addWidget(self.warn_label)

        # --- строка 5: краткое описание
        self.description = QLabel(self._short_description())
        self.description.setObjectName("Dim")
        self.description.setWordWrap(True)
        self.description.setTextFormat(compat.TEXT_FORMAT_RICH)
        root.addWidget(self.description)

        self.full_description = QLabel(self.entry.description)
        self.full_description.setObjectName("Dim")
        self.full_description.setWordWrap(True)
        self.full_description.setVisible(False)
        root.addWidget(self.full_description)

        # --- строка 6: прогресс загрузки
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

        # --- строка 7: кнопки
        buttons = QHBoxLayout()
        buttons.setSpacing(8)

        self.more_button = QPushButton("Подробнее")
        self.more_button.setObjectName("Ghost")
        self.more_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.more_button.clicked.connect(self._toggle_more)
        buttons.addWidget(self.more_button)

        if self.entry.url:
            page = QPushButton("Страница модели ↗")
            page.setObjectName("Link")
            page.setCursor(compat.CURSOR_POINTING_HAND)
            page.clicked.connect(lambda: open_url(self.entry.url))
            buttons.addWidget(page)

        buttons.addStretch(1)

        self.info_button = QPushButton("ℹ")
        self.info_button.setObjectName("Ghost")
        self.info_button.setFixedWidth(34)
        self.info_button.setToolTip("Техническая информация")
        self.info_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.info_button.clicked.connect(lambda: self.infoRequested.emit(self.entry))
        buttons.addWidget(self.info_button)

        self.delete_button = QPushButton("Удалить")
        self.delete_button.setObjectName("Danger")
        self.delete_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.delete_button.clicked.connect(lambda: self.deleteRequested.emit(self.entry))
        self.delete_button.setVisible(self.entry.installed)
        buttons.addWidget(self.delete_button)

        self.action_button = QPushButton("Выбрать" if self.entry.installed else "Скачать")
        self.action_button.setObjectName("Primary" if not self.entry.installed else "")
        self.action_button.setCursor(compat.CURSOR_POINTING_HAND)
        self.action_button.clicked.connect(self._on_action)
        buttons.addWidget(self.action_button)

        root.addLayout(buttons)

    # ------------------------------------------------------------------ #
    def _short_description(self) -> str:
        text = self.entry.description or ""
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
            return (
                f"⚠ нужно ~{self.entry.ram_mb / 1024:.1f} ГБ памяти, сейчас свободно "
                f"~{available / 1024:.1f} ГБ — возможны тормоза или ошибка"
            )
        if self.entry.vram_mb and self.hardware.get("cuda_ok"):
            vram = int(self.hardware.get("vram_mb") or 0)
            if vram and self.entry.vram_mb > vram * 0.9:
                return f"⚠ VRAM {vram / 1024:.1f} ГБ — модель пойдёт через CPU"
        return ""

    def _toggle_more(self) -> None:
        visible = not self.full_description.isVisible()
        self.full_description.setVisible(visible)
        self.more_button.setText("Свернуть" if visible else "Подробнее")
        self.updateGeometry()
        if self.parentWidget():
            self.parentWidget().updateGeometry()

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
            self.action_button.setText("✓ Выбрана" if selected else "Выбрать")
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
            self.action_button.setText("Скачивание…")
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(0)
            self.action_button.setText("Скачать" if not self.entry.installed else "Выбрать")
            self.action_button.setEnabled(True)
            self.delete_button.setEnabled(True)

    def update_progress(self, fraction: float, status: str) -> None:
        if not self.progress.isVisible():
            self.progress.setVisible(True)
            self.progress_label.setVisible(True)
        if fraction <= 0:
            self.progress.setRange(0, 0)  # «бегунок», размер неизвестен
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        self.progress_label.setText(status)

    def refresh_state(self, entry: ModelEntry) -> None:
        self.entry = entry
        self.delete_button.setVisible(entry.installed)
        self.action_button.setText("Выбрать" if entry.installed else "Скачать")
        self.action_button.setObjectName("Primary" if not entry.installed else "")
        self.action_button.style().unpolish(self.action_button)
        self.action_button.style().polish(self.action_button)
