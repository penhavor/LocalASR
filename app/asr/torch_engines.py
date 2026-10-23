"""Движки на PyTorch/transformers: Qwen3-ASR (Alibaba) и GigaAM (Sber).

Обе модели скачиваются из Hugging Face в папку models/ как обычные репозитории
transformers, а инференс идёт локально через torch:

* Qwen3-ASR — мультимодальная модель (аудио-энкодер + Qwen3-LM): 30 языков,
  автоопределение языка, punctuation из коробки. Работает через
  ``AutoProcessor.apply_transcription_request`` + ``generate``.
* GigaAM — conformer-модель Сбера (CTC/RNNT, best-in-class на русском).
  Грузится через ``AutoModel.from_pretrained(..., trust_remote_code=True)``;
  родной ``transcribe`` умеет только ≤25 секунд, поэтому длинные записи мы
  сами режем на куски по тишине и склеиваем результат.

Длинное аудио оба движка получают кусками (у тишины), чтобы:
  * не съедать гигабайты памяти на трёхчасовой лекции;
  * показывать живой прогресс и поддерживать отмену.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..constants import ENGINE_GIGAAM, ENGINE_QWEN_ASR, SAMPLE_RATE
from ..core.postprocess import Segment
from ..logutils import get_logger
from .base import (
    ASREngine,
    CancelToken,
    EngineError,
    ModelRef,
    ProgressCallback,
    SegmentCallback,
    TranscriptResult,
)

log = get_logger("asr.torch")

# Максимальная длина куска для каждого движка (секунды)
QWEN_CHUNK_SECONDS = 300.0     # Qwen3-ASR умеет длинные окна — режем редко
GIGAAM_CHUNK_SECONDS = 20.0    # родной transcribe GigaAM ограничен ~25 секундами


# --------------------------------------------------------------------------- #
#  Нарезка аудио по тишине
# --------------------------------------------------------------------------- #


def iter_silence_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    search_seconds: float = 2.0,
    window_seconds: float = 0.4,
) -> Iterator[Tuple[int, int]]:
    """Генератор границ (start, end) в сэмплах: куски до chunk_seconds,
    граница ищется в окрестности ±search_seconds по минимуму громкости."""
    total = len(audio)
    if total == 0:
        return
    win = max(1, int(window_seconds * sample_rate))
    start = 0
    while start < total:
        target = start + int(chunk_seconds * sample_rate)
        if target >= total:
            end = total
        else:
            lo = max(start + win, start + int((chunk_seconds - search_seconds) * sample_rate))
            hi = min(total, start + int((chunk_seconds + search_seconds) * sample_rate))
            end = target
            if hi - lo > win:
                segment = audio[lo:hi].astype(np.float32)
                kernel = np.ones(win, dtype=np.float32) / win
                rms = np.sqrt(np.convolve(segment * segment, kernel, mode="valid"))
                if rms.size:
                    end = lo + int(np.argmin(rms)) + win // 2
            if end <= start:
                end = target
        yield start, min(end, total)
        start = min(end, total)
        if start >= total:
            break


def torch_float32():
    import torch

    return torch.float32


def _build_gigaam_manual(mod: Any, config: Any, folder: Path) -> Any:
    """Собирает GigaAM вручную: init на cpu + явная загрузка весов.

    Обход meta-инициализации transformers v5, с которой hydra-внутренности
    (mel-фильтры, регистры RNNT) создаются на cpu и ловят конфликт устройств.
    """
    import torch

    config.name_or_path = str(folder)
    model = mod.GigaAMModel(config)
    weights = folder / "pytorch_model.bin"
    safetensors = folder / "model.safetensors"
    if safetensors.is_file():
        from safetensors.torch import load_file

        state = load_file(str(safetensors))
    elif weights.is_file():
        try:
            # mmap заметно снижает пик памяти на больших моделях
            state = torch.load(str(weights), map_location="cpu", weights_only=True, mmap=True)
        except (RuntimeError, OSError, ValueError):
            state = torch.load(str(weights), map_location="cpu", weights_only=True)
    else:
        raise EngineError(f"В {folder} нет ни pytorch_model.bin, ни model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # префикс base_model_prefix может отличаться — пробуем со снятием «model.»
    if unexpected and all(key.startswith("model.") for key in unexpected):
        state = {key[len("model."):]: value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise EngineError(
            f"В весах GigaAM не хватает {len(missing)} тензоров (например {missing[:3]})"
        )
    return model


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Пишет mono float32 → WAV s16le (стандартная библиотека, без зависимостей)."""
    pcm = np.clip(audio, -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(data.tobytes())


def read_wav_mono(path: Path, sample_rate: int):
    """Читает наш WAV s16le в torch-тензор (замена ffmpeg для GigaAM)."""
    import torch

    with wave.open(str(path), "rb") as fh:
        raw = fh.readframes(fh.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


# --------------------------------------------------------------------------- #
#  Общий базовый класс для torch-движков
# --------------------------------------------------------------------------- #


class TorchASREngine(ASREngine):
    """Общее: проверка доступности, устройство, разгрузка памяти."""

    chunk_seconds: float = 60.0
    #: все модули, без которых движок не заведётся (GUI предложит доустановку)
    import_names: Tuple[str, ...] = ("torch", "transformers")

    @classmethod
    def is_available(cls) -> bool:
        return all(importlib.util.find_spec(name) is not None for name in cls.import_names)

    # -- вспомогательное -------------------------------------------------- #
    def _pick_device(self, hardware: Dict[str, Any]) -> Tuple[str, Any]:
        import torch

        wanted = str(hardware.get("device") or "auto")
        use_cuda = False
        if wanted == "cuda":
            use_cuda = torch.cuda.is_available()
        elif wanted == "auto":
            use_cuda = bool(torch.cuda.is_available()) and bool(hardware.get("cuda_ok", True))
        self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32
        self.compute_type = "float16" if use_cuda else "float32"
        return self.device, dtype

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self.loaded_model_id = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _ensure_model(self) -> Any:
        if self._model is None:
            raise EngineError(f"Модель движка {self.name} не загружена")
        return self._model

    def __init__(self) -> None:
        super().__init__()
        self._processor: Any = None


# --------------------------------------------------------------------------- #
#  Qwen3-ASR
# --------------------------------------------------------------------------- #


class QwenASREngine(TorchASREngine):
    name = ENGINE_QWEN_ASR
    packages = ["torch>=2.6", "transformers>=5.0", "accelerate>=1.0", "soundfile>=0.12"]
    import_name = "transformers"
    import_names = ("torch", "transformers", "accelerate")
    chunk_seconds = QWEN_CHUNK_SECONDS

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Не удалось импортировать torch/transformers: {exc}\n"
                "Установи движок: Сервис → Настройки → Движки → Qwen3-ASR → «Установить»."
            ) from exc

        path = str(model.resolved_path)
        device, dtype = self._pick_device(hardware)
        try:
            self._processor = AutoProcessor.from_pretrained(path)
            try:
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    path, dtype=dtype, device_map=device
                )
            except TypeError:
                # transformers < 5.10: аргумент назывался torch_dtype
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    path, torch_dtype=dtype, device_map=device
                )
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"Не удалось загрузить Qwen3-ASR из {path}: {exc}") from exc
        self.loaded_model_id = model.model_id
        log.info("Qwen3-ASR загружен: %s (%s, %s)", model.model_id, device, self.compute_type)

    # ------------------------------------------------------------------ #
    def transcribe(
        self,
        audio: Any,
        *,
        options: Dict[str, Any],
        duration: float,
        progress: Optional[ProgressCallback] = None,
        on_segment: Optional[SegmentCallback] = None,
        cancel: Optional[CancelToken] = None,
    ) -> TranscriptResult:
        import torch

        model = self._ensure_model()
        started = time.time()
        audio = np.asarray(audio, dtype=np.float32)
        total = len(audio) or 1

        language = (options.get("language") or "").strip() or None
        prompt_parts = [options.get("initial_prompt") or "", options.get("hotwords") or ""]
        prompt = " ".join(part.strip() for part in prompt_parts if part.strip()) or None

        segments: List[Segment] = []
        language_detected = ""
        warnings: List[str] = []

        for start, end in iter_silence_chunks(audio, SAMPLE_RATE, self.chunk_seconds):
            if cancel and cancel():
                warnings.append("Распознавание остановлено пользователем")
                break
            chunk = audio[start:end]
            try:
                inputs = self._processor.apply_transcription_request(
                    audio=(chunk, SAMPLE_RATE), language=language, prompt=prompt
                )
                inputs = inputs.to(model.device, model.dtype)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=2048)
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                parsed = self._processor.decode(generated, return_format="parsed")[0]
            except Exception as exc:  # noqa: BLE001
                raise EngineError(f"Qwen3-ASR не смог распознать фрагмент: {exc}") from exc

            text = str(parsed.get("transcription") or "").strip()
            language_detected = str(parsed.get("language") or language_detected)
            if text:
                segment = Segment(start=start / SAMPLE_RATE, end=end / SAMPLE_RATE, text=text)
                segments.append(segment)
                if on_segment:
                    on_segment(segment)

            if progress:
                progress(end / total, f"Распознаю: {end / SAMPLE_RATE:.0f} с из {duration:.0f} с")

        if options.get("word_timestamps"):
            warnings.append("Qwen3-ASR не даёт таймкоды слов — сегменты по кускам аудио")

        return TranscriptResult(
            segments=segments,
            language=language_detected,
            audio_duration=duration,
            processed_duration=duration,
            elapsed_seconds=time.time() - started,
            engine=self.name,
            device=self.device,
            compute_type=self.compute_type,
            warnings=warnings,
        )


# --------------------------------------------------------------------------- #
#  GigaAM (Sber)
# --------------------------------------------------------------------------- #


class GigaAMEngine(TorchASREngine):
    name = ENGINE_GIGAAM
    packages = [
        "torch>=2.6",
        "torchaudio>=2.6",
        "transformers>=4.52",
        "hydra-core>=1.3,<2",
        "omegaconf>=2.3,<3",
        "sentencepiece>=0.2.2",
    ]
    import_name = "transformers"
    import_names = ("torch", "transformers", "torchaudio", "hydra", "omegaconf", "sentencepiece")
    chunk_seconds = GIGAAM_CHUNK_SECONDS

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        try:
            from transformers import AutoModel
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Не удалось импортировать transformers: {exc}\n"
                "Установи движок: Сервис → Настройки → Движки → GigaAM → «Установить»."
            ) from exc

        # Код GigaAM внутри вызывает ffmpeg из PATH — подскажем ему наш бинарик
        try:
            from ..core.ffmpeg import find_ffmpeg

            ffmpeg = find_ffmpeg()
            if ffmpeg:
                extra = str(Path(ffmpeg).parent)
                current = os.environ.get("PATH", "")
                if extra not in current:
                    os.environ["PATH"] = extra + os.pathsep + current
        except Exception:  # noqa: BLE001
            pass

        path = str(model.resolved_path)
        device, dtype = self._pick_device(hardware)
        # ВАЖНО: dtype при from_pretrained не передаём — внутренности модели
        # (mel-фильтры и т.п.) создаются hydra на cpu и конфликтуют с meta-
        # инициализацией transformers. Точность применяем после загрузки.
        try:
            loaded = AutoModel.from_pretrained(
                path, trust_remote_code=True, low_cpu_mem_usage=False
            )
        except ImportError as exc:
            # modeling_gigaam.py лениво импортирует pyannote (только для ихнего
            # longform с HF-токеном). Нам longform не нужен: длинные записи мы
            # режем сами, поэтому грузим модуль напрямую, без проверки transformers.
            log.info("AutoModel требует %s — гружу modeling_gigaam.py напрямую", exc)
            loaded = self._load_direct(path, dtype)
        except Exception as exc:  # noqa: BLE001
            try:
                loaded = self._load_direct(path, dtype)
            except Exception as direct_exc:  # noqa: BLE001
                raise EngineError(
                    f"Не удалось загрузить GigaAM из {path}: {exc} / {direct_exc}\n"
                    "Нужны пакеты: torch, torchaudio, transformers, hydra-core, omegaconf, sentencepiece."
                ) from exc
        try:
            loaded = loaded.to(device)
            if device == "cuda" and dtype is not None and dtype != torch_float32():
                loaded = loaded.to(dtype)
            loaded.eval()
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"Не удалось перевести GigaAM на {device}: {exc}") from exc
        self._model = loaded
        self._patch_audio_loader()
        self.loaded_model_id = model.model_id
        log.info("GigaAM загружен: %s (%s, %s)", model.model_id, device, self.compute_type)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_direct(path: str, dtype: Any = None) -> Any:
        """Загрузка modeling_gigaam.py напрямую (без dynamic-module проверки)."""
        import importlib.util as iu

        module_path = Path(path) / "modeling_gigaam.py"
        if not module_path.is_file():
            raise EngineError(f"В {path} нет modeling_gigaam.py")
        spec = iu.spec_from_file_location("localasr_gigaam_modeling", str(module_path))
        if spec is None or spec.loader is None:
            raise EngineError(f"Не удалось прочитать {module_path}")
        mod = iu.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
            config = mod.GigaAMConfig.from_pretrained(path)
            try:
                # low_cpu_mem_usage=False: иначе transformers создаёт модель на
                # meta-устройстве, а hydra-внутренности (mel-фильтры) — на cpu,
                # и загрузка падает с «Tensor on device cpu is not on meta»
                return mod.GigaAMModel.from_pretrained(
                    path, config=config, low_cpu_mem_usage=False
                )
            except Exception as exc:  # noqa: BLE001
                # transformers v5 всё равно может увести init на meta-устройство:
                # собираем модель вручную и накладываем веса сами
                log.info("from_pretrained не смог (%s) — собираю GigaAM вручную", exc)
                return _build_gigaam_manual(mod, config, Path(path))
        except Exception:
            sys.modules.pop(spec.name, None)
            raise

    # ------------------------------------------------------------------ #
    def _patch_audio_loader(self) -> None:
        """Подменяем load_audio в модуле модели: читаем WAV сами, без ffmpeg."""
        module = sys.modules.get(type(self._model).__module__ or "")
        if module is not None and hasattr(module, "load_audio"):
            module.load_audio = lambda path, sample_rate=SAMPLE_RATE: read_wav_mono(
                Path(path), sample_rate
            )

    # ------------------------------------------------------------------ #
    def transcribe(
        self,
        audio: Any,
        *,
        options: Dict[str, Any],
        duration: float,
        progress: Optional[ProgressCallback] = None,
        on_segment: Optional[SegmentCallback] = None,
        cancel: Optional[CancelToken] = None,
    ) -> TranscriptResult:
        model = self._ensure_model()
        started = time.time()
        audio = np.asarray(audio, dtype=np.float32)
        total = len(audio) or 1

        segments: List[Segment] = []
        warnings: List[str] = []
        tmp_path: Optional[Path] = None

        try:
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = Path(handle.name)
            handle.close()

            for start, end in iter_silence_chunks(audio, SAMPLE_RATE, self.chunk_seconds, search_seconds=1.0):
                if cancel and cancel():
                    warnings.append("Распознавание остановлено пользователем")
                    break
                write_wav(tmp_path, audio[start:end], SAMPLE_RATE)
                try:
                    text = model.transcribe(str(tmp_path))
                except Exception as exc:  # noqa: BLE001
                    raise EngineError(f"GigaAM не смог распознать фрагмент: {exc}") from exc
                text = str(text or "").strip()
                if text:
                    segment = Segment(start=start / SAMPLE_RATE, end=end / SAMPLE_RATE, text=text)
                    segments.append(segment)
                    if on_segment:
                        on_segment(segment)
                if progress:
                    progress(end / total, f"Распознаю: {end / SAMPLE_RATE:.0f} с из {duration:.0f} с")
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        if options.get("word_timestamps"):
            warnings.append("GigaAM использован без таймкодов слов (режим chunks)")

        return TranscriptResult(
            segments=segments,
            language="ru",
            audio_duration=duration,
            processed_duration=duration,
            elapsed_seconds=time.time() - started,
            engine=self.name,
            device=self.device,
            compute_type=self.compute_type,
            warnings=warnings,
        )
