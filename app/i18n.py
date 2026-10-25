"""Localization core (stdlib only).

Source strings in the code are English msgids. Translations live in
resources/i18n/<lang>.json as {msgid: translation}. A missing translation
falls back to the msgid itself, so the source language is always safe.
Plural entries use {"one": .., "few": .., "many": ..} values via tn().
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SUPPORTED: Tuple[str, ...] = ("en", "ru")
SOURCE_LANGUAGE = "en"
AUTO = ""  # preference value meaning "follow the system language"

#: Windows LANGID primary sublanguages
_WIN_LANGID = {0x19: "ru", 0x09: "en"}

_lang: str = SOURCE_LANGUAGE
_pref: str = AUTO
_catalog: Dict[str, Any] = {}
_root: Optional[Path] = None


def project_root() -> Path:
    global _root
    if _root is None:
        env = os.environ.get("LOCALASR_ROOT")
        if env and (Path(env) / "launcher" / "main.py").is_file():
            _root = Path(env).resolve()
        else:
            _root = Path(__file__).resolve().parent.parent
    return _root


def detect_system_language() -> str:
    """Map the OS UI language onto SUPPORTED (falls back to English)."""
    candidates: List[str] = []
    if os.name == "nt":
        try:
            import ctypes

            langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            candidates.append(_WIN_LANGID.get(langid & 0x3FF, ""))
        except Exception:  # noqa: BLE001
            pass
    for var in ("LANGUAGE", "LC_ALL", "LC_CTYPE", "LANG"):
        value = os.environ.get(var)
        if value:
            candidates.extend(value.split(":"))
    try:
        import locale

        code = (locale.getlocale()[0] or "") if hasattr(locale, "getlocale") else ""
        candidates.append(str(code))
    except Exception:  # noqa: BLE001
        pass
    for cand in candidates:
        code = cand.lower().split(".")[0].replace("-", "_").split("_")[0]
        if code in SUPPORTED:
            return code
    return SOURCE_LANGUAGE


def _load_catalog(lang: str) -> Dict[str, Any]:
    if lang == SOURCE_LANGUAGE:
        return {}
    path = project_root() / "resources" / "i18n" / f"{lang}.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_preference(root: Optional[Path] = None) -> str:
    """Read ui_language from settings.json without importing app.settings."""
    path = (root or project_root()) / "settings" / "settings.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        payload = data.get("settings", data) if isinstance(data, dict) else {}
        value = str(payload.get("ui_language", "") or "")
        return value if value in SUPPORTED else AUTO
    except (OSError, ValueError, AttributeError):
        return AUTO


def init(pref: Optional[str] = None, root: Optional[Path] = None) -> str:
    """Activate a language; returns it.

    pref: None = read settings.json, "" = auto (system), or "en"/"ru".
    """
    global _lang, _pref, _catalog, _root
    if root is not None:
        _root = Path(root)
    if pref is None:
        pref = read_preference(_root)
    pref = (pref or "").strip().lower()
    _pref = pref if pref in SUPPORTED else AUTO
    _lang = _pref if _pref else detect_system_language()
    _catalog = _load_catalog(_lang)
    return _lang


def current_language() -> str:
    return _lang


def current_preference() -> str:
    return _pref


def t(msgid: str) -> str:
    value = _catalog.get(msgid)
    return value if isinstance(value, str) else msgid


_ = t


def tn(one: str, many: str, n: int) -> str:
    """Plural-aware translate. The catalog stores Slavic forms under `one`."""
    try:
        n = abs(int(n))
    except (TypeError, ValueError):
        n = 0
    if _lang == "ru":
        forms = _catalog.get(one)
        if isinstance(forms, dict):
            mod10, mod100 = n % 10, n % 100
            if mod10 == 1 and mod100 != 11:
                key = "one"
            elif 2 <= mod10 <= 4 and not 12 <= mod100 <= 14:
                key = "few"
            else:
                key = "many"
            value = forms.get(key) or forms.get("many")
            if isinstance(value, str):
                return value
    return one if n == 1 else many


def qt_translator_names() -> List[str]:
    """Qt built-in translation bases for the active language (empty for en)."""
    if _lang == SOURCE_LANGUAGE:
        return []
    return [f"qtbase_{_lang}"]


def language_choices() -> List[Tuple[str, str]]:
    """(label, settings value) for the UI language selector; labels are endonyms."""
    return [
        (t("Detect automatically (system language)"), AUTO),
        ("English", "en"),
        ("Русский", "ru"),
    ]
