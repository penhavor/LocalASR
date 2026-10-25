# -*- coding: utf-8 -*-
"""Localization integrity tests ("no desync" guarantees).

  * every _()/t()/tr() string literal in the code exists in the Russian
    catalog (resources/i18n/ru.json);
  * every tn() plural msgid has a complete one/few/many form set;
  * the data files (model catalog, engines.json) hold SOURCE-language (English)
    text only — translations live in the catalogs, never mixed into the data;
  * the catalog file parses and has no empty values;
  * i18n.init switches languages and falls back to the msgid when a
    translation is absent.

Run:  cd <project root> && python tests/test_i18n.py
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LOCALASR_QUIET", "1")

CATALOG = ROOT / "resources" / "i18n" / "ru.json"
CYRILLIC = re.compile("[\\u0400-\\u04FF]")

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


def walk_sources():
    """All project .py files that participate in the localization."""
    roots = [ROOT / "app", ROOT / "launcher"]
    for item in roots:
        if item.is_file():
            yield item
        elif item.is_dir():
            for dirpath, _dirs, files in os.walk(item):
                for name in sorted(files):
                    if name.endswith(".py"):
                        yield Path(dirpath) / name


def extract_literals():
    """Collect (msgid, kind) from _()/t()/tr() and tn() calls."""
    singles, plurals = set(), set()
    for path in walk_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if name in ("_", "t", "tr") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    singles.add(arg.value)
            elif name == "tn" and len(node.args) >= 2:
                a, b = node.args[0], node.args[1]
                if isinstance(a, ast.Constant) and isinstance(b, ast.Constant):
                    plurals.add(a.value)
    return singles, plurals


def catalog_strings():
    """User-facing strings inside the data files (must be English)."""
    out = []
    cat = json.loads((ROOT / "resources" / "models_catalog.json").read_text(encoding="utf-8"))
    out.append(cat.get("note", ""))
    out.extend(cat.get("engines", {}).values())
    for model in cat.get("models", []):
        for field in ("name", "description", "languages_note", "recommended_for", "vendor"):
            out.append(model.get(field) or "")
        out.extend(model.get("tags") or [])
    engines = json.loads((ROOT / "requirements" / "engines.json").read_text(encoding="utf-8"))
    for value in engines.values():
        if isinstance(value, dict):
            out.append(value.get("note") or "")
            out.append(value.get("install_size_text") or "")
    return [s for s in out if s]


def main() -> int:
    print("== catalog structure ==")
    raw = CATALOG.read_text(encoding="utf-8")
    catalog = json.loads(raw)
    check(isinstance(catalog, dict) and catalog, "ru.json parses to a non-empty object")
    check(all(isinstance(v, (str, dict)) and v for v in catalog.values()),
          "no empty translation values")

    print("== code msgids are covered ==")
    singles, plurals = extract_literals()
    missing = sorted(k for k in singles if k not in catalog)
    check(not missing, f"all {len(singles)} _()/tr() msgids exist in ru.json")
    for key in missing[:10]:
        print(f"      missing: {key!r}")

    print("== plural forms complete ==")
    for key in sorted(plurals):
        value = catalog.get(key)
        ok = isinstance(value, dict) and all(value.get(form) for form in ("one", "few", "many"))
        check(ok, f"tn() msgid has one/few/many: {key!r}")

    print("== data files hold the source language only ==")
    bad = [s for s in catalog_strings() if CYRILLIC.search(s)]
    check(not bad, "no Cyrillic inside models_catalog.json / engines.json")
    for item in bad[:5]:
        print(f"      cyrillic: {item!r}")

    print("== runtime switching and fallback ==")
    from app import i18n

    i18n.init("en")
    check(i18n.current_language() == "en", "explicit en")
    check(i18n.t("Settings") == "Settings", "source language returns the msgid")

    i18n.init("ru")
    check(i18n.current_language() == "ru", "explicit ru")
    check(i18n.t("Settings") == "Настройки", "ru translation applied")
    check(i18n.t("no-such-msgid-anywhere") == "no-such-msgid-anywhere", "missing key falls back")
    check(i18n.tn("{} file", "{} files", 5).format(5) == "5 файлов", "ru plural (many)")
    check(i18n.tn("{} file", "{} files", 2).format(2) == "2 файла", "ru plural (few)")

    i18n.init("en")
    check(i18n.tn("{} file", "{} files", 5).format(5) == "5 files", "en plural")
    check(i18n.qt_translator_names() == [], "no Qt translator for the source language")
    i18n.init("ru")
    check(i18n.qt_translator_names() == ["qtbase_ru"], "Qt translator name for ru")
    i18n.init("en")

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("I18N INTEGRITY TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
