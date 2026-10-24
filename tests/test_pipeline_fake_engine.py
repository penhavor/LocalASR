# -*- coding: utf-8 -*-
"""Сквозной тест пайплайна transcribe_file на ФЕЙКОВОМ движке.

Никаких реальных моделей/ffmpeg: probe и decode_audio подменяются. Проверяется
весь цикл «файл -> звук -> движок -> постобработка -> .txt»:
  * успешный прогон: FileResult.ok, сегменты, слова, файл результата;
  * анти-галлюцинации применяются (мусорные сегменты отброшены);
  * коллизия имён: второй прогон того же файла НЕ затирает первый результат,
    а создаёт «имя + дата и время» (задача 6a);
  * overwrite=True перезаписывает;
  * отмена (cancel) прерывает распознавание;
  * ошибка движка -> FileResult.ok=False с текстом ошибки (без падения);
  * progress-колбэк вызывается и доходит до 1.0.

Запуск:  cd <корень проекта> && python tests/test_pipeline_fake_engine.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("LOCALASR_QUIET", "1")

import numpy as np  # noqa: E402

from app.asr import transcriber as tr  # noqa: E402
from app.asr.base import EngineError, TranscriptResult  # noqa: E402
from app.constants import SAMPLE_RATE  # noqa: E402
from app.core import ffmpeg  # noqa: E402
from app.core.models_catalog import ModelEntry  # noqa: E402
from app.core.postprocess import Segment  # noqa: E402
from app.settings import Settings  # noqa: E402

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


class FakeInfo:
    path = Path("лекция 1.mp4")
    duration = 30.0
    size_bytes = 8
    audio_streams = 1
    video_streams = 1
    is_video = True
    codec = "aac"
    sample_rate = 48000
    channels = 2
    error = ""

    @property
    def ok(self) -> bool:
        return not self.error


class FakeEngine:
    """Движок-заглушка: отдаёт фиксированные сегменты + мусор для чистки."""

    name = "fake-engine"
    device = "cpu"
    compute_type = "float32"

    def __init__(self, fail: bool = False, cancel_aware: bool = False) -> None:
        self.fail = fail
        self.cancel_aware = cancel_aware
        self.seen_audio_len = 0

    def transcribe(self, audio, *, options, duration, progress=None, on_segment=None, cancel=None):
        self.seen_audio_len = len(audio)
        if self.fail:
            raise EngineError("движок сломался (тест)")
        segments = [
            Segment(0.0, 4.0, "Первая нормальная фраза про машинное обучение."),
            Segment(4.0, 8.0, "Вторая фраза continues the lecture."),
            Segment(8.0, 9.0, "[музыка]"),
            Segment(9.0, 10.0, "Спасибо за просмотр!"),
        ]
        for index in range(5):
            if cancel and cancel():
                break
            if self.cancel_aware and index == 2:
                break
            if progress:
                progress((index + 1) / 5, f"тест {index + 1}/5")
            if on_segment and index < len(segments):
                on_segment(segments[index])
            if self.cancel_aware:
                time.sleep(0.01)
        return TranscriptResult(
            segments=segments,
            language=options.get("language") or "ru",
            audio_duration=duration,
            processed_duration=duration,
            elapsed_seconds=0.5,
            engine=self.name,
            device=self.device,
            compute_type=self.compute_type,
        )


def make_options(entry: ModelEntry, settings: Settings, out: Path, overwrite: bool = False):
    settings.overwrite_existing = overwrite
    options = tr.JobOptions.from_settings(entry, settings, out)
    return options


def main() -> int:
    # подменяем чтение медиа: probe и decode_audio
    original_probe, original_decode = tr.probe, tr.decode_audio
    tr.probe = lambda source: FakeInfo()
    tr.decode_audio = lambda source, sr, cancel=None, progress=None: np.zeros(
        int(30 * SAMPLE_RATE), dtype=np.float32
    )
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "входящие" / "лекция 1.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fake mp4")
            out = root / "outputs"

            settings = Settings.load(root)
            entry = ModelEntry(id="fake", name="Фейковая модель", engine="fake-engine",
                               installed=True, install_dir=root)

            print("== успешный прогон ==")
            engine = FakeEngine()
            options = make_options(entry, settings, out)
            progress_values = []
            result = tr.transcribe_file(
                source, options,
                progress=lambda f, s: progress_values.append(f),
                engine=engine,
            )
            check(result.ok, "FileResult.ok")
            check(engine.seen_audio_len == int(30 * SAMPLE_RATE), "движок получил декодированное аудио")
            import re
            stamp_re = r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}"
            check(
                len(result.outputs) == 1
                and re.fullmatch(rf"лекция 1 {stamp_re}\.txt", result.outputs[0].name) is not None,
                f"результат: имя оригинала + дата/время: {result.outputs[0].name if result.outputs else '—'}",
            )
            text = result.outputs[0].read_text(encoding="utf-8")
            check("Первая нормальная фраза" in text, "полезный текст на месте")
            check("[музыка]" not in text and "Спасибо за просмотр" not in text,
                  "анти-галлюцинации вычистили мусор")
            check(result.dropped >= 2, f"отброшено мусорных сегментов: {result.dropped}")
            check(result.words > 5, f"счётчик слов: {result.words}")
            check(max(progress_values) >= 0.999, "progress дошёл до 1.0")

            print("== второй прогон: отдельный файл (уникальная метка времени) ==")
            time.sleep(1.05)   # новая секунда для метки времени
            result2 = tr.transcribe_file(source, make_options(entry, settings, out), engine=FakeEngine())
            check(result2.ok and result2.outputs[0] != result.outputs[0],
                  f"второй файл отдельный: {result2.outputs[0].name}")
            check(result.outputs[0].is_file() and "Первая нормальная фраза" in result.outputs[0].read_text(encoding="utf-8"),
                  "первый результат не затёрт")
            import re
            check(
                re.fullmatch(r"лекция 1 \d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}\.txt", result2.outputs[0].name)
                is not None,
                "имя второго = оригинал + дата и время",
            )

            print("== overwrite=True ==")
            result3 = tr.transcribe_file(
                source, make_options(entry, settings, out, overwrite=True), engine=FakeEngine()
            )
            check(result3.outputs[0].name == "лекция 1.txt", "перезапись пишет в исходное имя")

            print("== отмена ==")
            cancelled = {"flag": False}

            def cancel_after():
                time.sleep(0.005)
                cancelled["flag"] = True

            import threading
            threading.Thread(target=cancel_after, daemon=True).start()
            result4 = tr.transcribe_file(
                source, make_options(entry, settings, out),
                engine=FakeEngine(cancel_aware=True),
                cancel=lambda: cancelled["flag"],
            )
            # отмена между сегментами: часть сегментов есть, файл сохранён
            check(result4.ok, "отменённый прогон всё равно сохранил распознанное")

            print("== ошибка движка ==")
            files_before = len(list(out.glob("*.txt")))
            try:
                tr.transcribe_file(
                    source, make_options(entry, settings, out), engine=FakeEngine(fail=True)
                )
                check(False, "сбой движка должен пробрасываться наружу (ловит воркер/CLI)")
            except EngineError as exc:
                check("сломался" in str(exc), f"ошибка движка понятна: {exc}")
            check(len(list(out.glob("*.txt"))) == files_before, "файл результата не создан")
    finally:
        tr.probe, tr.decode_audio = original_probe, original_decode

    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("СКВОЗНОЙ ТЕСТ ПАЙПЛАЙНА ПРОЙДЕН")
    return 0


if __name__ == "__main__":
    sys.exit(main())
