# -*- coding: utf-8 -*-
"""Тесты путей, настроек и именования результатов.

Проверяется (задачи 6a/6b):
  * коллизия имён в outputs НЕ затирает старый файл: новый получает
    «старое имя + дата и время»;
  * перезапись включается только флагом overwrite;
  * устаревший output_dir (проект переехал) мигрирует на outputs/ текущего
    проекта — папка «outputs» вне проекта больше не используется;
  * значение по умолчанию не сохраняется абсолютным путём;
  * недоступная папка результата -> откат на outputs/ проекта;
  * sanitize_stem / unique_path / plural_ru / human_size — инварианты.

Запуск:  cd <корень проекта> && python tests/test_settings_paths.py
"""
from __future__ import annotations

import json
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

from app import paths  # noqa: E402
from app.core import exporters  # noqa: E402
from app.core.postprocess import Segment  # noqa: E402
from app.settings import Settings  # noqa: E402

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


def test_export_collision_adds_datetime() -> None:
    print("== имена результатов: оригинал + дата и время (всегда) ==")
    stamp_re = r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "outputs"
        segs = [Segment(0.0, 1.5, "первый запуск")]
        first = exporters.export_all_formats(segs, out, "лекция 1", formats=["txt"])
        check(
            re.fullmatch(rf"лекция 1 {stamp_re}\.txt", first[0].name) is not None,
            f"первый файл сразу с датой и временем: {first[0].name}",
        )

        segs2 = [Segment(0.0, 1.5, "второй запуск")]
        time.sleep(1.05)   # новая секунда — метка времени гарантированно отличается
        second = exporters.export_all_formats(segs2, out, "лекция 1", formats=["txt"])
        check(second[0].name != first[0].name, "второй прогон — отдельный файл")
        check(first[0].is_file(), "старый файл остался на диске")
        check(
            re.fullmatch(rf"лекция 1 {stamp_re}\.txt", second[0].name) is not None,
            f"второе имя тоже оригинал + дата/время: {second[0].name}",
        )
        check(
            "первый запуск" in first[0].read_text(encoding="utf-8").lower()
            and "второй запуск" in second[0].read_text(encoding="utf-8").lower(),
            "содержимое обоих файлов корректно",
        )

        # перезапись флагом overwrite -> детерминированное чистое имя
        third = exporters.export_all_formats(segs2, out, "лекция 1", formats=["txt"], overwrite=True)
        check(third[0].name == "лекция 1.txt", "overwrite=True пишет в чистое имя")

        # несколько форматов одного прогона делят ОДНУ метку времени
        multi = exporters.export_all_formats(
            segs, out, "доклад", formats=["txt", "srt", "vtt"],
        )
        stamps = set()
        for path in multi:
            match = re.search(stamp_re, path.name)
            check(match is not None, f"метка времени в {path.name}")
            if match:
                stamps.add(match.group(0))
        check(len(stamps) == 1, f"все форматы прогона с одной меткой: {stamps}")
        check(sorted(p.suffix for p in multi) == [".srt", ".txt", ".vtt"], "форматы на месте")

        # коллизия расширений txt и txt-time -> метка формата + та же временная метка
        multi2 = exporters.export_all_formats(
            segs, out, "доклад", formats=["txt", "txt-time"], labels={"txt-time": "таймкоды"}
        )
        names = sorted(p.name for p in multi2)
        check(any("(таймкоды)" in n for n in names), f"txt-time получил метку формата: {names}")
        stamps2 = {re.search(stamp_re, n).group(0) for n in names if re.search(stamp_re, n)}
        check(len(stamps2) == 1, "txt и txt-time делят одну метку времени")


def test_unique_path_datetime() -> None:
    print("== unique_path: тоже дата/время ==")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        first = paths.unique_path(out, "файл", ".txt")
        first.write_text("x", encoding="utf-8")
        second = paths.unique_path(out, "файл", ".txt")
        check(second != first and second.exists() is False, f"коллизия -> {second.name}")
        check(re.search(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", second.name) is not None,
              "в имени есть дата и время")
        same = paths.unique_path(out, "файл", ".txt", overwrite=True)
        check(same == first, "overwrite=True возвращает исходный путь")


def test_stale_output_dir_migrates() -> None:
    print("== переехавший проект: устаревший outputs/ сбрасывается ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "NewProject"
        (root / "settings").mkdir(parents=True)
        desktop_outputs = Path(td) / "Desktop" / "LocalASR" / "outputs"
        payload = {"version": 1, "settings": {"output_dir": str(desktop_outputs)}}
        (root / "settings" / "settings.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        settings = Settings.load(root)
        check(settings.output_dir == "", f"устаревший путь сброшен: {settings.output_dir!r}")
        resolved = settings.resolved_output_dir()
        check(resolved == root / "outputs", f"результат идёт в outputs текущего проекта: {resolved}")
        check(not desktop_outputs.exists(), "старая папка на «рабочем столе» НЕ создаётся")

        # папка, которая уже существует, тоже мигрирует (баг из жизни: app сам
        # её и создал в прошлый раз)
        desktop_outputs.mkdir(parents=True, exist_ok=True)
        (root / "settings" / "settings.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        settings2 = Settings.load(root)
        check(settings2.output_dir == "", "существующий чужой outputs/ тоже мигрирует")

        # намеренная папка с другим именем НЕ трогается
        custom = Path(td) / "Мои расшифровки"
        payload2 = {"version": 1, "settings": {"output_dir": str(custom)}}
        (root / "settings" / "settings.json").write_text(
            json.dumps(payload2, ensure_ascii=False), encoding="utf-8"
        )
        settings3 = Settings.load(root)
        check(settings3.output_dir == str(custom), "своя папка (не outputs) сохраняется")

        # outputs ВНУТРИ текущего проекта не мигрирует
        own = str(root / "outputs")
        payload3 = {"version": 1, "settings": {"output_dir": own}}
        (root / "settings" / "settings.json").write_text(
            json.dumps(payload3, ensure_ascii=False), encoding="utf-8"
        )
        settings4 = Settings.load(root)
        check(settings4.output_dir == own, "собственный outputs/ остаётся как есть")


def test_resolved_output_dir_fallback() -> None:
    print("== недоступная папка результата -> откат на outputs/ проекта ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "proj"
        (root / "settings").mkdir(parents=True)
        settings = Settings.load(root)
        settings.output_dir = "/proc/self/нельзя/создать/outputs2" if os.name != "nt" else "Z:\\нет-такого-диска\\outputs2"
        resolved = settings.resolved_output_dir()
        check(resolved == root / "outputs", f"откат на outputs/: {resolved}")


def test_settings_resilience() -> None:
    print("== битые настройки не роняют приложение ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "settings").mkdir(parents=True)
        (root / "settings" / "settings.json").write_text("{не json", encoding="utf-8")
        settings = Settings.load(root)
        check(settings.language == "ru", "битый файл -> дефолты")
        check((root / "settings" / "settings.json.broken").is_file(), "битый файл переименован")
        settings.save()
        reloaded = Settings.load(root)
        check(reloaded.file_logging is False, "save/load после починки работает")


def test_paths_helpers() -> None:
    print("== paths-хелперы ==")
    check(paths.sanitize_stem(r"C:\Video\лекция 1.mp4") == "лекция 1", "windows-путь -> имя")
    check(paths.sanitize_stem("a/b:c*.mp4") == "b_c", "нелегальные символы заменяются")
    check(paths.sanitize_stem("CON.txt") == "_CON", "зарезервированное имя экранировано")
    check(paths.plural_ru(1, "файл", "файла", "файлов") == "файл", "1 файл")
    check(paths.plural_ru(2, "файл", "файла", "файлов") == "файла", "2 файла")
    check(paths.plural_ru(11, "файл", "файла", "файлов") == "файлов", "11 файлов")
    check(paths.human_size(1536) == "1.5 КБ", "human_size")
    stamp = paths.datetime_stamp()
    check(re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", stamp) is not None,
          f"datetime_stamp: {stamp}")


def main() -> int:
    test_export_collision_adds_datetime()
    test_unique_path_datetime()
    test_stale_output_dir_migrates()
    test_resolved_output_dir_fallback()
    test_settings_resilience()
    test_paths_helpers()
    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("ТЕСТ ПУТЕЙ И НАСТРОЕК ПРОЙДЕН")
    return 0


if __name__ == "__main__":
    sys.exit(main())
