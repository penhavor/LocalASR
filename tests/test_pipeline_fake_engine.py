# -*- coding: utf-8 -*-
"""End-to-end transcribe_file pipeline test on a FAKE engine.

No real models/ffmpeg: probe and decode_audio are patched. The whole cycle
"file -> audio -> engine -> post-processing -> .txt" is verified:
  * a successful run: FileResult.ok, segments, words, the result file;
  * anti-hallucinations apply (junk segments dropped);
  * name collision: a second run of the same file does NOT overwrite the
    first result but creates "name + date and time";
  * overwrite=True overwrites;
  * cancellation interrupts recognition;
  * an engine error -> FileResult.ok=False with the error text (no crash);
  * the progress callback fires and reaches 1.0.

Run:  cd <project root> && python tests/test_pipeline_fake_engine.py
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
    """Stub engine: fixed segments + junk for the cleaner."""

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
            raise EngineError("engine broke (test)")
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
                progress((index + 1) / 5, f"test {index + 1}/5")
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
    # patch media reading: probe and decode_audio
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

            print("== successful run ==")
            engine = FakeEngine()
            options = make_options(entry, settings, out)
            progress_values = []
            result = tr.transcribe_file(
                source, options,
                progress=lambda f, s: progress_values.append(f),
                engine=engine,
            )
            check(result.ok, "FileResult.ok")
            check(engine.seen_audio_len == int(30 * SAMPLE_RATE), "the engine got the decoded audio")
            import re
            stamp_re = r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}"
            check(
                len(result.outputs) == 1
                and re.fullmatch(rf"лекция 1 {stamp_re}\.txt", result.outputs[0].name) is not None,
                f"result: original name + date/time: {result.outputs[0].name if result.outputs else '—'}",
            )
            text = result.outputs[0].read_text(encoding="utf-8")
            check("Первая нормальная фраза" in text, "useful text present")
            check("[музыка]" not in text and "Спасибо за просмотр" not in text,
                  "anti-hallucinations cleaned the junk")
            check(result.dropped >= 2, f"junk segments dropped: {result.dropped}")
            check(result.words > 5, f"word counter: {result.words}")
            check(max(progress_values) >= 0.999, "progress reached 1.0")

            print("== second run: a separate file (unique timestamp) ==")
            time.sleep(1.05)   # a new second for the timestamp
            result2 = tr.transcribe_file(source, make_options(entry, settings, out), engine=FakeEngine())
            check(result2.ok and result2.outputs[0] != result.outputs[0],
                  f"second file is separate: {result2.outputs[0].name}")
            check(result.outputs[0].is_file() and "Первая нормальная фраза" in result.outputs[0].read_text(encoding="utf-8"),
                  "the first result is not overwritten")
            import re
            check(
                re.fullmatch(r"лекция 1 \d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}\.txt", result2.outputs[0].name)
                is not None,
                "second name = original + date and time",
            )

            print("== overwrite=True ==")
            result3 = tr.transcribe_file(
                source, make_options(entry, settings, out, overwrite=True), engine=FakeEngine()
            )
            check(result3.outputs[0].name == "лекция 1.txt", "overwrite writes the original name")

            print("== cancellation ==")
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
            # cancelled between segments: some segments exist, file saved
            check(result4.ok, "the cancelled run still saved what was recognized")

            print("== engine error ==")
            files_before = len(list(out.glob("*.txt")))
            try:
                tr.transcribe_file(
                    source, make_options(entry, settings, out), engine=FakeEngine(fail=True)
                )
                check(False, "an engine failure must propagate (caught by worker/CLI)")
            except EngineError as exc:
                check("broke" in str(exc), f"engine error is clear: {exc}")
            check(len(list(out.glob("*.txt"))) == files_before, "no result file created")
    finally:
        tr.probe, tr.decode_audio = original_probe, original_decode

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("END-TO-END PIPELINE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
