# -*- coding: utf-8 -*-
"""Paths, settings and result-naming tests.

  * a name collision in outputs never overwrites the old file: the new one
    gets "old name + date and time";
  * overwrite happens only with the overwrite flag;
  * the default value is not stored as an absolute path;
  * an unavailable results folder falls back to the project outputs/;
  * sanitize_stem / unique_path / tn (plurals) / human_size invariants.

Run:  cd <project root> && python tests/test_settings_paths.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("LOCALASR_QUIET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import i18n, paths  # noqa: E402
from app.core import exporters  # noqa: E402
from app.core.postprocess import Segment  # noqa: E402
from app.settings import Settings  # noqa: E402

i18n.init("ru")  # stable language for the naming expectations below

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


def test_export_collision_adds_datetime() -> None:
    print("== result names: original + date and time (always) ==")
    stamp_re = r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "outputs"
        segs = [Segment(0.0, 1.5, "первый запуск")]
        first = exporters.export_all_formats(segs, out, "лекция 1", formats=["txt"])
        check(
            re.fullmatch(rf"лекция 1 {stamp_re}\.txt", first[0].name) is not None,
            f"first file already carries date and time: {first[0].name}",
        )

        segs2 = [Segment(0.0, 1.5, "второй запуск")]
        time.sleep(1.05)   # a new second — the timestamp is guaranteed to differ
        second = exporters.export_all_formats(segs2, out, "лекция 1", formats=["txt"])
        check(second[0].name != first[0].name, "second run — a separate file")
        check(first[0].is_file(), "the old file is still on disk")
        check(
            re.fullmatch(rf"лекция 1 {stamp_re}\.txt", second[0].name) is not None,
            f"second name is also original + date/time: {second[0].name}",
        )
        check(
            "первый запуск" in first[0].read_text(encoding="utf-8").lower()
            and "второй запуск" in second[0].read_text(encoding="utf-8").lower(),
            "contents of both files are correct",
        )

        # overwrite flag -> deterministic clean name
        third = exporters.export_all_formats(segs2, out, "лекция 1", formats=["txt"], overwrite=True)
        check(third[0].name == "лекция 1.txt", "overwrite=True writes the clean name")

        # several formats of one run share ONE timestamp
        multi = exporters.export_all_formats(
            segs, out, "доклад", formats=["txt", "srt", "vtt"],
        )
        stamps = set()
        for path in multi:
            match = re.search(stamp_re, path.name)
            check(match is not None, f"timestamp in {path.name}")
            if match:
                stamps.add(match.group(0))
        check(len(stamps) == 1, f"all formats of the run share one stamp: {stamps}")
        check(sorted(p.suffix for p in multi) == [".srt", ".txt", ".vtt"], "formats present")

        # txt vs txt-time extension collision -> format label + same stamp
        multi2 = exporters.export_all_formats(
            segs, out, "доклад", formats=["txt", "txt-time"],
            labels={"txt-time": i18n._("timecodes")},
        )
        names = sorted(p.name for p in multi2)
        check(any("(таймкоды)" in n for n in names), f"txt-time got the format label: {names}")
        stamps2 = {re.search(stamp_re, n).group(0) for n in names if re.search(stamp_re, n)}
        check(len(stamps2) == 1, "txt and txt-time share one timestamp")


def test_unique_path_datetime() -> None:
    print("== unique_path: date/time too ==")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        first = paths.unique_path(out, "файл", ".txt")
        first.write_text("x", encoding="utf-8")
        second = paths.unique_path(out, "файл", ".txt")
        check(second != first and second.exists() is False, f"collision -> {second.name}")
        check(re.search(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", second.name) is not None,
              "the name contains date and time")
        same = paths.unique_path(out, "файл", ".txt", overwrite=True)
        check(same == first, "overwrite=True returns the original path")


def test_resolved_output_dir_fallback() -> None:
    print("== unavailable results folder -> fallback to the project outputs/ ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "proj"
        (root / "settings").mkdir(parents=True)
        settings = Settings.load(root)
        settings.output_dir = "/proc/self/cannot/create/outputs2" if os.name != "nt" else "Z:\\no-such-disk\\outputs2"
        resolved = settings.resolved_output_dir()
        check(resolved == root / "outputs", f"fallback to outputs/: {resolved}")


def test_settings_resilience() -> None:
    print("== broken settings do not crash the app ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "settings").mkdir(parents=True)
        (root / "settings" / "settings.json").write_text("{not json", encoding="utf-8")
        settings = Settings.load(root)
        check(settings.language == "ru", "broken file -> defaults")
        check((root / "settings" / "settings.json.broken").is_file(), "broken file renamed")
        settings.save()
        reloaded = Settings.load(root)
        check(reloaded.file_logging is False, "save/load works after the repair")


def test_paths_helpers() -> None:
    print("== paths helpers ==")
    check(paths.sanitize_stem(r"C:\Video\лекция 1.mp4") == "лекция 1", "windows path -> name")
    check(paths.sanitize_stem("a/b:c*.mp4") == "b_c", "illegal characters replaced")
    check(paths.sanitize_stem("CON.txt") == "_CON", "reserved name escaped")
    check(i18n.tn("{} file", "{} files", 1) == "{} файл", "plural 1")
    check(i18n.tn("{} file", "{} files", 2) == "{} файла", "plural 2")
    check(i18n.tn("{} file", "{} files", 11) == "{} файлов", "plural 11")
    check(paths.human_size(1536) == "1.5 КБ", "human_size")
    stamp = paths.datetime_stamp()
    check(re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", stamp) is not None,
          f"datetime_stamp: {stamp}")


def test_ui_language_setting() -> None:
    print("== ui_language setting: default auto, saved and loaded ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "settings").mkdir(parents=True)
        settings = Settings.load(root)
        check(settings.ui_language == "", "factory default = auto-detect")
        settings.ui_language = "en"
        settings.save(root)
        reloaded = Settings.load(root)
        check(reloaded.ui_language == "en", "preference round-trips")
        check(i18n.read_preference(root) == "en", "i18n reads the preference")


def main() -> int:
    test_export_collision_adds_datetime()
    test_unique_path_datetime()
    test_resolved_output_dir_fallback()
    test_settings_resilience()
    test_paths_helpers()
    test_ui_language_setting()
    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("PATHS AND SETTINGS TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
