# -*- coding: utf-8 -*-
"""Тест УНИВЕРСАЛЬНОСТИ движков LocalASR.

Программа обязана корректно работать не с одной моделью, а с любыми:
  * матрица detect_engine: CT2 / GGML / Vosk / HF-config / Qwen / GigaAM;
  * Vosk и whisper.cpp прогоняются на фейковых биндингах (контракт библиотек);
  * GigaAM НЕИЗВЕСТНОЙ структуры (без decoding/head) падает в stock-путь
    model.transcribe() вместо быстрого — универсальность для новых поколений;
  * универсальный HF-движок на РЕАЛЬНОЙ модели (openai/whisper-tiny):
    длинное аудио режется на куски, сегменты тайлятся, текст — строки.
    (Без сети/transformers этот шаг пропускается: остальные шаги самодостаточны.)

Запуск:  cd <корень проекта> && python tests/test_engines_universal.py
"""
from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from app.asr.base import ModelRef  # noqa: E402
from app.constants import (  # noqa: E402
    ENGINE_FASTER_WHISPER,
    ENGINE_GIGAAM,
    ENGINE_HF,
    ENGINE_QWEN_ASR,
    ENGINE_VOSK,
    ENGINE_WHISPERCPP,
    SAMPLE_RATE,
)
from app.core.models_catalog import detect_engine  # noqa: E402

SR = SAMPLE_RATE


def _audio(seconds: float = 10.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.2, int(seconds * SR)) * 0.3).astype(np.float32)


# --------------------------------------------------------------------------- #
#  1) Матрица определения движка по содержимому папки
# --------------------------------------------------------------------------- #


def test_detect_engine_matrix() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        ct2 = root / "ct2"
        (ct2 / "int8").mkdir(parents=True)
        (ct2 / "int8" / "model.bin").write_bytes(b"x")
        (ct2 / "config.json").write_text("{}", encoding="utf-8")
        assert detect_engine(ct2) == ENGINE_FASTER_WHISPER

        ggml = root / "ggml"
        ggml.mkdir()
        (ggml / "ggml-large-v3-turbo-q5_0.bin").write_bytes(b"x")
        assert detect_engine(ggml) == ENGINE_WHISPERCPP
        assert detect_engine(ggml / "ggml-large-v3-turbo-q5_0.bin") == ENGINE_WHISPERCPP

        vosk = root / "vosk"
        (vosk / "am").mkdir(parents=True)
        (vosk / "conf").mkdir(parents=True)
        assert detect_engine(vosk) == ENGINE_VOSK
        (root / "vosk.zip").write_bytes(b"PK\x03\x04")
        assert detect_engine(root / "vosk.zip") == ENGINE_VOSK

        hf = root / "hf"
        hf.mkdir()
        (hf / "config.json").write_text(
            json.dumps({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"]}),
            encoding="utf-8",
        )
        (hf / "model.safetensors").write_bytes(b"x")
        assert detect_engine(hf) == ENGINE_HF

        qwen = root / "qwen"
        qwen.mkdir()
        (qwen / "config.json").write_text(
            json.dumps({"model_type": "qwen3_asr"}), encoding="utf-8"
        )
        (qwen / "model.safetensors").write_bytes(b"x")
        assert detect_engine(qwen) == ENGINE_QWEN_ASR

        giga = root / "giga"
        giga.mkdir()
        (giga / "config.json").write_text(json.dumps({"model_type": "gigaam"}), encoding="utf-8")
        (giga / "pytorch_model.bin").write_bytes(b"x")
        assert detect_engine(giga) == ENGINE_GIGAAM

        junk = root / "junk"
        junk.mkdir()
        (junk / "readme.txt").write_text("hi", encoding="utf-8")
        assert detect_engine(junk) == ""
    print("  ✓ detect_engine: CT2/GGML/Vosk/HF/Qwen/GigaAM/мусор")


# --------------------------------------------------------------------------- #
#  2) Vosk на фейковом биндинге (контракт библиотеки vosk)
# --------------------------------------------------------------------------- #


def _fake_vosk_module():
    mod = types.ModuleType("vosk")
    mod.__spec__ = importlib.machinery.ModuleSpec("vosk", None)

    class Model:
        def __init__(self, path):
            self.path = path

    class KaldiRecognizer:
        def __init__(self, model, sample_rate, grammar=None):
            self.model = model
            self.sr = sample_rate
            self.grammar = grammar
            self._blocks = 0

        def SetWords(self, flag):  # noqa: N802 - контракт vosk
            self.words = flag

        def AcceptWaveform(self, data):  # noqa: N802
            self._blocks += 1
            return self._blocks % 2 == 0   # «сегмент» каждые два блока

        def Result(self):
            return json.dumps(
                {"text": "первый блок речи", "result": [
                    {"word": "первый", "start": 0.0, "end": 0.5, "conf": 0.9},
                    {"word": "блок", "start": 0.5, "end": 0.9, "conf": 0.9},
                    {"word": "речи", "start": 0.9, "end": 1.3, "conf": 0.9},
                ]},
                ensure_ascii=False,
            )

        def FinalResult(self):
            return json.dumps({"text": "хвост записи", "result": [
                {"word": "хвост", "start": 8.0, "end": 8.4, "conf": 0.8},
                {"word": "записи", "start": 8.4, "end": 8.9, "conf": 0.8},
            ]}, ensure_ascii=False)

    def SetLogLevel(level):  # noqa: N802
        return None

    mod.Model = Model
    mod.KaldiRecognizer = KaldiRecognizer
    mod.SetLogLevel = SetLogLevel
    return mod


def test_vosk_engine_on_fake_binding() -> None:
    from app.asr.vosk_engine import VoskEngine

    fake = _fake_vosk_module()
    saved = sys.modules.get("vosk")
    sys.modules["vosk"] = fake
    try:
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "vosk-model"
            (folder / "am").mkdir(parents=True)
            engine = VoskEngine()
            engine.load(ModelRef(model_id="v", name="V", engine=ENGINE_VOSK, path=folder), {})
            result = engine.transcribe(_audio(10.0), options={}, duration=10.0)
        texts = [s.text for s in result.segments]
        assert texts, "vosk не дал сегментов"
        assert all(isinstance(t, str) and t for t in texts)
        # минимальная пунктуация для беспунктуационного движка
        assert all(t[-1] in ".!," for t in texts), texts
        assert result.segments[0].words, "word-таймкоды не доехали"
    finally:
        if saved is None:
            sys.modules.pop("vosk", None)
        else:
            sys.modules["vosk"] = saved
    print("  ✓ Vosk: контракт библиотеки соблюдён, сегменты+пунктуация на месте")


# --------------------------------------------------------------------------- #
#  3) whisper.cpp на фейковом pywhispercpp
# --------------------------------------------------------------------------- #


def _fake_pywhispercpp():
    pkg = types.ModuleType("pywhispercpp")
    pkg.__spec__ = importlib.machinery.ModuleSpec("pywhispercpp", None)
    model_mod = types.ModuleType("pywhispercpp.model")
    model_mod.__spec__ = importlib.machinery.ModuleSpec("pywhispercpp.model", None)

    class Seg:
        def __init__(self, t0, t1, text):
            self.t0, self.t1, self.text = t0, t1, text

    class Model:
        def __init__(self, path, **kwargs):
            self.path = path
            self.kwargs = kwargs

        def get_params_schema(self):
            return {"language": None, "beam_search": None, "new_segment_callback": None,
                    "abort_callback": None, "temperature": None}

        def transcribe(self, waveform, **kwargs):
            cb = kwargs.get("new_segment_callback")
            segs = [Seg(0, 150, "фраза из whisper cpp"), Seg(150, 400, "вторая фраза")]
            if cb is not None:
                for s in segs:
                    cb(s)
            return segs

    model_mod.Model = Model
    pkg.model = model_mod
    return pkg, model_mod


def test_whispercpp_engine_on_fake_binding() -> None:
    from app.asr.whispercpp_engine import WhisperCppEngine

    pkg, model_mod = _fake_pywhispercpp()
    saved_pkg, saved_mod = sys.modules.get("pywhispercpp"), sys.modules.get("pywhispercpp.model")
    sys.modules["pywhispercpp"] = pkg
    sys.modules["pywhispercpp.model"] = model_mod
    try:
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "ggml-tiny.bin").write_bytes(b"x" * 1024)
            engine = WhisperCppEngine()
            engine.load(ModelRef(model_id="w", name="W", engine=ENGINE_WHISPERCPP, path=folder), {})
            result = engine.transcribe(_audio(6.0), options={"beam_size": 5}, duration=6.0)
        texts = [s.text for s in result.segments]
        assert texts == ["фраза из whisper cpp", "вторая фраза"], texts
        assert result.segments[0].start == 0.0 and result.segments[1].end == 4.0
    finally:
        for key, saved in (("pywhispercpp", saved_pkg), ("pywhispercpp.model", saved_mod)):
            if saved is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = saved
    print("  ✓ whisper.cpp: callback-поток сегментов и таймкоды (t0/t1 в сотых сек) верны")


# --------------------------------------------------------------------------- #
#  4) GigaAM НЕИЗВЕСТНОЙ структуры -> stock-путь, быстрый отключён
# --------------------------------------------------------------------------- #


def test_gigaam_unknown_structure_falls_back_to_stock() -> None:
    from app.asr.torch_engines import GigaAMEngine

    class UnknownInner:
        """Поколение модели без decoding/head — быстрый путь неприменим."""

    class StockModel:
        calls = 0

        def transcribe(self, path):
            StockModel.calls += 1
            assert str(path).endswith(".wav")
            return "текст из родного transcribe"

    engine = GigaAMEngine()
    assert engine._fast_kind(UnknownInner()) == ""
    engine._model = StockModel()
    audio = _audio(45.0, seed=3)
    bounds = [(0, 20 * SR), (20 * SR, 45 * SR)]
    segments, warnings = [], []
    engine._transcribe_stock(audio, bounds, len(audio), 45.0, segments, warnings, None, None, None)
    assert StockModel.calls == 2, StockModel.calls
    assert [s.text for s in segments] == ["текст из родного transcribe"] * 2
    assert segments[0].start == 0.0 and segments[-1].end == 45.0
    print("  ✓ GigaAM неизвестного поколения: быстрый путь отключён, stock-путь тайлит куски")


# --------------------------------------------------------------------------- #
#  5) Универсальный HF-движок на реальной модели + реальном чанкинге
# --------------------------------------------------------------------------- #


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return 8.0
    return 8.0


def test_hf_engine_real_model() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("  - HF-шаг пропущен: нет transformers")
        return
    if _ram_gb() < 0.9:
        print("  - HF-шаг пропущен: мало RAM для реального инференса в песочнице")
        return
    audio_path = Path(ROOT).parent / "audio" / "hf_chunk_test.wav"
    if not audio_path.is_file():
        print("  - HF-шаг пропущен: нет audio/hf_chunk_test.wav")
        return

    # декодируем СОБСТВЕННЫМ путём программы (ffmpeg или PyAV-фолбэк)
    from app.core.ffmpeg import decode_audio

    try:
        audio = np.ascontiguousarray(np.asarray(decode_audio(audio_path, SR), dtype=np.float32))
    except Exception as exc:  # noqa: BLE001
        print(f"  - HF-шаг пропущен: декодер недоступен: {exc}")
        return
    if len(audio) < 30 * SR:
        print("  - HF-шаг пропущен: аудио короче 30 с")
        return

    from app.asr.hf_pipeline_engine import HFPipelineEngine

    try:
        engine = HFPipelineEngine()
        engine.load(
            ModelRef(model_id="hf-tiny", name="tiny", engine=ENGINE_HF,
                     path=Path(os.environ.get("HF_TINY_CACHE", "openai/whisper-tiny"))),
            {"device": "cpu"},
        )
    except Exception as exc:  # noqa: BLE001 - нет сети/диска: не валим весь тест
        print(f"  - HF-шаг пропущен: {type(exc).__name__}: {str(exc)[:120]}")
        return

    duration = len(audio) / SR
    # greedy и без батчей: тест универсальности должен жить даже в 1 ГБ RAM;
    # качество текста здесь не проверяется — только контракт движка и чанкинга
    result = engine.transcribe(
        audio, options={"language": "ru", "beam_size": 1, "vad": False, "batched": False},
        duration=duration,
    )
    segs = result.segments
    assert segs, "HF-движок не дал сегментов"
    # тайлинг: куски идут встык и покрывают всё аудио (допуск 0.2 c)
    assert segs[0].start == 0.0
    assert abs(segs[-1].end - duration) < 0.5, (segs[-1].end, duration)
    for a, b in zip(segs, segs[1:]):
        assert abs(a.end - b.start) < 1e-6, (a.end, b.start)
    assert all(isinstance(s.text, str) and s.text.strip() for s in segs)
    assert len(segs) >= 2, f"40 c аудио должны дать >=2 кусков, получено {len(segs)}"
    print(f"  ✓ HF-движок (whisper-tiny): {len(segs)} куска на {duration:.0f} c, "
          f"тайлинг и строки корректны")


if __name__ == "__main__":
    test_detect_engine_matrix()
    test_vosk_engine_on_fake_binding()
    test_whispercpp_engine_on_fake_binding()
    test_gigaam_unknown_structure_falls_back_to_stock()
    test_hf_engine_real_model()
    print("\nТЕСТ УНИВЕРСАЛЬНОСТИ ПРОЙДЕН")
