"""Тема оформления (QSS) — тёмная, без внешних зависимостей и картинок из сети."""

from __future__ import annotations

BG = "#0f1116"
BG_PANEL = "#151922"
BG_CARD = "#1b2130"
BG_CARD_HOVER = "#222a3c"
BG_INPUT = "#0d1017"
BORDER = "#2a3244"
BORDER_LIGHT = "#38415a"
TEXT = "#e8ecf4"
TEXT_DIM = "#98a2b3"
TEXT_MUTED = "#6b7688"
ACCENT = "#5b8cff"
ACCENT_HOVER = "#7aa2ff"
ACCENT_DOWN = "#4a76e0"
SUCCESS = "#3ecf8e"
WARNING = "#f2b544"
DANGER = "#f2637f"

RADIUS = 10


def stylesheet() -> str:
    return f"""
* {{
    font-family: "Segoe UI", "Inter", "Noto Sans", "DejaVu Sans", sans-serif;
    font-size: 13px;
    color: {TEXT};
    outline: none;
}}

QMainWindow, QDialog {{
    background-color: {BG};
}}

QWidget {{
    background-color: transparent;
}}

/* ---------------- панели ---------------- */
QFrame#SidePanel {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: {RADIUS + 2}px;
}}

QFrame#Card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
}}

QFrame#Card:hover {{
    background-color: {BG_CARD_HOVER};
    border-color: {BORDER_LIGHT};
}}

QFrame#Card[selected="true"] {{
    background-color: #1d2a44;
    border: 1px solid {ACCENT};
}}

QFrame#Banner {{
    background-color: #1c2436;
    border: 1px solid {ACCENT};
    border-radius: {RADIUS}px;
}}

QFrame#HeaderBar {{
    background-color: {BG_PANEL};
    border-bottom: 1px solid {BORDER};
}}

/* ---------------- типографика ---------------- */
QLabel#H1 {{
    font-size: 20px;
    font-weight: 700;
    color: {TEXT};
}}

QLabel#H2 {{
    font-size: 15px;
    font-weight: 650;
    color: {TEXT};
}}

QLabel#H3 {{
    font-size: 13px;
    font-weight: 600;
    color: {TEXT};
}}

QLabel#Dim {{
    color: {TEXT_DIM};
}}

QLabel#Muted {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}

QLabel#BadgeRecommended {{
    background-color: #24406b;
    color: #cfe0ff;
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 600;
}}

QLabel#BadgeInstalled {{
    background-color: #17402f;
    color: #b6f2d5;
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 600;
}}

QLabel#BadgeEngine {{
    background-color: #2b2440;
    color: #d7ccff;
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 11px;
}}

QLabel#BadgeWarn {{
    background-color: #3d3220;
    color: #ffd79a;
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 11px;
}}

QLabel#BigButton {{
    font-size: 15px;
    font-weight: 700;
}}

/* ---------------- кнопки ---------------- */
QPushButton {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_LIGHT};
    border-radius: {RADIUS - 2}px;
    padding: 7px 14px;
    color: {TEXT};
}}

QPushButton:hover {{
    background-color: {BG_CARD_HOVER};
    border-color: {ACCENT};
}}

QPushButton:pressed {{
    background-color: #131a29;
}}

QPushButton:disabled {{
    color: {TEXT_MUTED};
    background-color: #141821;
    border-color: {BORDER};
}}

QPushButton#Primary {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: #07101f;
    font-weight: 700;
}}

QPushButton#Primary:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}

QPushButton:disabled#Primary, QPushButton#Primary:disabled {{
    background-color: #23314d;
    border-color: #2a3854;
    color: {TEXT_MUTED};
}}

QPushButton#Danger {{
    border-color: #5a2a37;
    color: #ffc2cf;
}}

QPushButton#Danger:hover {{
    background-color: #3a1d27;
    border-color: {DANGER};
}}

QPushButton#Ghost {{
    background-color: transparent;
    border: 1px solid {BORDER};
    color: {TEXT_DIM};
}}

QPushButton#Ghost:hover {{
    color: {TEXT};
    border-color: {ACCENT};
}}

QPushButton#Link {{
    background: transparent;
    border: none;
    color: {ACCENT};
    padding: 2px 4px;
    text-align: left;
}}

QPushButton#Link:hover {{
    color: {ACCENT_HOVER};
    text-decoration: underline;
}}

/* Кнопка «ℹ» — техническая информация о модели: заметная, круглая */
QPushButton#InfoButton {{
    background-color: #21304d;
    border: 1px solid {ACCENT};
    color: {ACCENT_HOVER};
    border-radius: 15px;
    font-size: 15px;
    font-weight: 700;
    padding: 0;
}}

QPushButton#InfoButton:hover {{
    background-color: {ACCENT};
    color: #07101f;
}}

/* ---------------- поля ввода ---------------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS - 2}px;
    padding: 6px 9px;
    selection-background-color: {ACCENT};
    selection-color: #07101f;
}}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {ACCENT};
}}

QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
    color: {TEXT_MUTED};
    background-color: #12151d;
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_LIGHT};
    selection-background-color: {ACCENT};
    selection-color: #07101f;
    padding: 4px;
}}

/* ---------------- чекбоксы/радио ---------------- */
QCheckBox, QRadioButton {{
    spacing: 8px;
    color: {TEXT};
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 17px;
    height: 17px;
    border: 1px solid {BORDER_LIGHT};
    background-color: {BG_INPUT};
}}

QCheckBox::indicator {{
    border-radius: 5px;
}}

QRadioButton::indicator {{
    border-radius: 9px;
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    image: none;
}}

QRadioButton::indicator:checked {{
    background-color: {BG_INPUT};
    border: 5px solid {ACCENT};
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}

/* ---------------- списки ---------------- */
QListWidget, QTreeView, QTableView {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    padding: 4px;
}}

QListWidget::item {{
    border-radius: 8px;
    padding: 6px;
    margin: 2px 0;
    border: 1px solid transparent;
}}

QListWidget::item:hover {{
    background-color: {BG_CARD_HOVER};
}}

QListWidget::item:selected {{
    background-color: #22335a;
    border: 1px solid {ACCENT};
}}

/* ---------------- прогресс ---------------- */
/* Основной прогресс-бар: высокий, проценты читаются и на тёмном треке,
   и на залитой части (белый жирный текст на насыщенном синем). */
QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 10px;
    min-height: 20px;
    max-height: 20px;
    padding: 0;
    text-align: center;
    color: #f4f7ff;
    font-size: 12px;
    font-weight: 700;
}}

QProgressBar::chunk {{
    background-color: {ACCENT_DOWN};
    border-radius: 9px;
}}

/* Тонкий суб-бар (текущий файл): текст не рисуется вообще — иначе проценты
   вылезали за пределы 6-пиксельной полосы. */
QProgressBar#Subtle {{
    border: none;
    background-color: #10141d;
    min-height: 6px;
    max-height: 6px;
    color: transparent;
    font-size: 1px;
}}

QProgressBar#Subtle::chunk {{
    background-color: {SUCCESS};
    border-radius: 3px;
}}

/* ---------------- скроллбары ---------------- */
QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER_LIGHT};
    border-radius: 5px;
    min-height: 28px;
}}

QScrollBar::handle:vertical:hover {{
    background: {ACCENT};
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 11px;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    background: {BORDER_LIGHT};
    border-radius: 5px;
    min-width: 28px;
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0;
    height: 0;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* ---------------- вкладки, сплиттеры, статус ---------------- */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    top: -1px;
}}

QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 7px 16px;
    border: 1px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}

QTabBar::tab:selected {{
    background-color: {BG_PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-bottom-color: {BG_PANEL};
}}

QSplitter::handle {{
    background-color: transparent;
}}

QSplitter::handle:horizontal {{
    width: 8px;
}}

QStatusBar {{
    background-color: {BG_PANEL};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
}}

QStatusBar::item {{
    border: none;
}}

QMenuBar {{
    background-color: {BG_PANEL};
    color: {TEXT_DIM};
    border-bottom: 1px solid {BORDER};
    padding: 2px;
}}

QMenuBar::item:selected {{
    background-color: {BG_CARD_HOVER};
    color: {TEXT};
    border-radius: 6px;
}}

QMenu {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_LIGHT};
    padding: 6px;
}}

QMenu::item {{
    padding: 7px 22px;
    border-radius: 6px;
}}

QMenu::item:selected {{
    background-color: {ACCENT};
    color: #07101f;
}}

QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 5px 8px;
}}

QDockWidget {{
    titlebar-close-icon: none;
    color: {TEXT_DIM};
}}

QDockWidget::title {{
    background-color: {BG_PANEL};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
}}

QToolTip {{
    background-color: {BG_PANEL};
    color: {TEXT};
    border: 1px solid {BORDER_LIGHT};
    padding: 6px;
    border-radius: 6px;
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    margin-top: 14px;
    padding-top: 8px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT_DIM};
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

/* Подложка под прозрачными скролл-областями: без неё на части платформ
   (offscreen/wayland) вместо фона темы остаются «прозрачные дыры». */
QSplitter {{
    background-color: {BG};
}}

QMainWindow::separator {{
    background-color: {BG};
}}
"""


def palette_hints() -> dict:
    """Цвета для рисования вручную (иконки, бейджи)."""
    return {
        "bg": BG,
        "panel": BG_PANEL,
        "card": BG_CARD,
        "border": BORDER,
        "text": TEXT,
        "dim": TEXT_DIM,
        "muted": TEXT_MUTED,
        "accent": ACCENT,
        "success": SUCCESS,
        "warning": WARNING,
        "danger": DANGER,
    }
