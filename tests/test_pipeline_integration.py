"""Интеграционный тест пайплайна с подменённым движком.

Проверяет всю цепочку без скачивания моделей и без сети:
  WAV → probe → decode (настоящий ffmpeg/PyAV) → «движок» → анти-галлюцинации
  → экспорт во все 6 форматов → имена файлов = именам оригинала.

Реальные движки (faster-whisper, whisper.cpp, Vosk) прогоняются отдельно
скриптами scripts/dev_e2e_*.py.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.asr import registry as registry_module
from app.asr.base import ASREngine, ModelRef, TranscriptResult
from app.asr.transcriber import JobOptions, transcribe_file
from app.constants import OUTPUT_FORMATS
from app.core.models_catalog import ModelCatalog
from app.settings import Settings

def _audio_decoder_available() -> bool:
    """Нужен ffmpeg или PyAV — без них пайплайн не проверить (например, в голом CI)."""
    try:
        from app.core.ffmpeg import find_ffmpeg

        if find_ffmpeg():
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy  # noqa: F401
        import av  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _audio_decoder_available(),
    reason="нет ни ffmpeg, ни PyAV — декодировать звук нечем",
)

FAKE_ENGINE = "fake-engine"

FAKE_SEGMENTS = [
    # (start, end, text, avg_logprob, no_speech_prob)
    (0.0, 3.0, "сегодня мы разберём градиентный спуск", -0.25, 0.01),
    (3.0, 6.0, "это основной метод оптимизации нейросетей", -0.3, 0.02),
    (6.0, 7.0, "[музыка]", -0.1, 0.05),
    (7.0, 9.0, "это основной метод оптимизации нейросетей", -0.3, 0.02),   # повтор-петля
    (12.0, 15.0, "после долгой паузы перейдём к примерам", -0.28, 0.01),
    (15.0, 16.0, "спасибо за просмотр!", -0.1, 0.03),                     # мусор
    (16.0, 19.0, "домашнее задание на следующей неделе", -0.27, 0.03),
]


class FakeEngine(ASREngine):
    name = FAKE_ENGINE
    packages: List[str] = []
    import_name = ""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.loaded_model_id = model.model_id
        self.device = "cpu"
        self.compute_type = "fake"

    def transcribe(self, audio, *, options, duration, progress=None, on_segment=None, cancel=None):
        self.calls += 1
        assert hasattr(audio, "shape"), "движок должен получать numpy-массив"
        result = TranscriptResult(
            engine=self.name,
            device="cpu",
            compute_type="fake",
            model_name=options.get("model_name", ""),
            language=options.get("language") or "ru",
            language_probability=0.97,
            audio_duration=duration,
            processed_duration=duration,
        )
        total = len(FAKE_SEGMENTS)
        for index, (start, end, text, logprob, no_speech) in enumerate(FAKE_SEGMENTS, start=1):
            if cancel is not None and cancel():
                break
            from app.core.postprocess import Segment

            segment = Segment(
                start=start, end=end, text=text,
                avg_logprob=logprob, no_speech_prob=no_speech,
                language=result.language,
            )
            result.segments.append(segment)
            if on_segment is not None:
                on_segment(segment)
            if progress is not None:
                progress(index / total, f"{index}/{total}")
        result.elapsed_seconds = 0.5
        return result


def _make_wav(path: Path, seconds: float = 20.0, rate: int = 16000) -> Path:
    """Генерирует WAV с «речеподобным» сигналом (амплитудная модуляция тона)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(int(seconds * rate)):
        t = index / rate
        tone = math.sin(2 * math.pi * 180 * t)
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 3 * t)     # «слоги»
        pause = 0.0 if 9.0 <= t < 12.0 else 1.0                   # пауза как в лекции
        value = int(tone * envelope * pause * 12000)
        frames += struct.pack("<h", max(-32767, min(32767, value)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(frames))
    return path


@pytest.fixture()
def fake_model(temp_root: Path) -> Any:
    """«Установленная» модель в models/ + зарегистрированный фейковый движок."""
    folder = temp_root / "models" / "fake-lecture-model"
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"FAKE" * 8)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "installed.json").write_text(
        json.dumps({"id": "fake-lecture-model", "engine": FAKE_ENGINE, "languages": ["ru"]}),
        encoding="utf-8",
    )

    registry_module.ENGINE_CLASSES[FAKE_ENGINE] = FakeEngine
    registry_module.registry.unload()
    catalog = ModelCatalog(root=temp_root)
    entry = catalog.get("fake-lecture-model")
    assert entry is not None and entry.installed
    entry.engine = FAKE_ENGINE
    yield entry
    registry_module.ENGINE_CLASSES.pop(FAKE_ENGINE, None)
    registry_module.registry.unload()


def test_pipeline_end_to_end(tmp_path: Path, temp_root: Path, fake_model) -> None:
    source = _make_wav(tmp_path / "лекция 1.wav")
    out_dir = tmp_path / "results"

    settings = Settings.load(root=temp_root)
    settings.language = "ru"
    settings.output_format = "txt"
    settings.output_dir = str(out_dir)

    options = JobOptions.from_settings(fake_model, settings, out_dir)
    options.formats = list(OUTPUT_FORMATS)
    options.write_header = True

    stages: list[tuple[float, str]] = []
    live: list[str] = []

    result = transcribe_file(
        source,
        options,
        progress=lambda fraction, stage: stages.append((fraction, stage)),
        on_segment=lambda segment: live.append(segment.text),
    )

    # --- сам результат
    assert result.ok, result.error
    assert result.segments > 0
    assert result.language == "ru"
    assert result.audio_duration == pytest.approx(20.0, abs=1.0)
    assert live, "«живая» лента сегментов должна приходить"
    assert stages and stages[-1][0] == pytest.approx(1.0)

    # --- анти-галлюцинации
    text = " ".join(out.read_text(encoding="utf-8") for out in result.outputs if out.suffix == ".md")
    assert "[музыка]" not in text.lower()
    assert "спасибо за просмотр" not in text.lower()
    assert text.lower().count("основной метод оптимизации") == 1, "повтор-петля должен быть удалён"
    assert result.dropped >= 2

    # --- имена файлов = имени оригинала, все 6 форматов на месте
    names = sorted(out.name for out in result.outputs)
    assert names == sorted(
        [
            "лекция 1.txt",
            "лекция 1 (таймкоды).txt",
            "лекция 1.srt",
            "лекция 1.vtt",
            "лекция 1.json",
            "лекция 1.md",
        ]
    ), names
    assert {out.suffix for out in result.outputs} == {".txt", ".srt", ".vtt", ".json", ".md"}
    assert len(result.outputs) == len(OUTPUT_FORMATS)
    assert all(out.is_file() and out.stat().st_size > 0 for out in result.outputs)

    # --- содержимое форматов
    by_name = {out.name: out for out in result.outputs}

    plain = by_name["лекция 1.txt"].read_text(encoding="utf-8")
    assert "градиентный спуск" in plain.lower()
    assert "[" not in plain, "в простом txt не должно быть таймкодов"

    timed = by_name["лекция 1 (таймкоды).txt"].read_text(encoding="utf-8")
    assert "[00:00:00]" in timed
    assert "Модель:" in timed, "шапка должна быть (write_header=True)"

    srt = by_name["лекция 1.srt"].read_text(encoding="utf-8")
    assert srt.startswith("1\n") and "-->" in srt and "," in srt

    vtt = by_name["лекция 1.vtt"].read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT")

    payload = json.loads(by_name["лекция 1.json"].read_text(encoding="utf-8"))
    assert payload["meta"]["model"] == fake_model.name
    assert payload["meta"]["language"] == "ru"
    assert payload["segments"] and payload["text"]
    assert payload["meta"]["audio_duration"] == pytest.approx(20.0, abs=1.0)


def test_pipeline_respects_existing_files(tmp_path: Path, temp_root: Path, fake_model) -> None:
    source = _make_wav(tmp_path / "лекция 2.wav", seconds=6)
    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True)
    (out_dir / "лекция 2.txt").write_text("старый результат", encoding="utf-8")

    settings = Settings.load(root=temp_root)
    settings.output_dir = str(out_dir)
    options = JobOptions.from_settings(fake_model, settings, out_dir)

    result = transcribe_file(source, options)
    assert result.ok
    assert result.outputs[0].name == "лекция 2 (2).txt", "не должны перезаписывать чужой файл"
    assert (out_dir / "лекция 2.txt").read_text(encoding="utf-8") == "старый результат"


def test_pipeline_overwrite_mode(tmp_path: Path, temp_root: Path, fake_model) -> None:
    source = _make_wav(tmp_path / "лекция 3.wav", seconds=6)
    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True)
    (out_dir / "лекция 3.txt").write_text("старый результат", encoding="utf-8")

    settings = Settings.load(root=temp_root)
    settings.output_dir = str(out_dir)
    settings.overwrite_existing = True
    options = JobOptions.from_settings(fake_model, settings, out_dir)

    result = transcribe_file(source, options)
    assert result.ok
    assert result.outputs[0].name == "лекция 3.txt"
    assert "старый результат" not in result.outputs[0].read_text(encoding="utf-8")


def test_pipeline_cancellation(tmp_path: Path, temp_root: Path, fake_model) -> None:
    source = _make_wav(tmp_path / "лекция 4.wav", seconds=6)
    settings = Settings.load(root=temp_root)
    options = JobOptions.from_settings(fake_model, settings, tmp_path / "out")

    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    result = transcribe_file(source, options, cancel=cancel)
    assert not result.ok or result.segments < len(FAKE_SEGMENTS)


def test_pipeline_reports_missing_audio_stream(tmp_path: Path, temp_root: Path, fake_model) -> None:
    """Файл без звука → понятная ошибка, а не падение."""
    broken = tmp_path / "пусто.mp4"
    broken.write_bytes(b"not a video at all")

    settings = Settings.load(root=temp_root)
    options = JobOptions.from_settings(fake_model, settings, tmp_path / "out")
    result = transcribe_file(broken, options)
    assert result.ok is False
    assert result.error
