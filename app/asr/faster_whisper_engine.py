"""Движок faster-whisper (CTranslate2).

Это основной движок LocalASR: в ~4 раза быстрее оригинального openai/whisper,
поддерживает int8-квантование, Silero VAD, hotwords и пакетный (batched)
декодинг на GPU. Работает и на CPU, и на CUDA — без PyTorch.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable, Dict, List, Optional

from ..constants import ENGINE_FASTER_WHISPER
from ..core.postprocess import Segment, Word
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

log = get_logger("asr.faster-whisper")

# Параметры Silero VAD, подобранные под лекции: длинные паузы между блоками,
# но короткие — внутри фразы (иначе текст «слипается» и плывут таймкоды).
VAD_DEFAULTS: Dict[str, Any] = {
    "threshold": 0.5,
    "min_silence_duration_ms": 400,
    "min_speech_duration_ms": 250,
    "speech_pad_ms": 300,
}


def _supported_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Отсекает параметры, которых нет в данной версии библиотеки.

    Именно из-за «неожиданного ключевого аргумента» обычно и падают такие
    обёртки при обновлении faster-whisper — здесь это исключено.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


class FasterWhisperEngine(ASREngine):
    name = ENGINE_FASTER_WHISPER
    packages: List[str] = ["faster-whisper>=1.2.0,<1.3"]
    import_name = "faster_whisper"

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        from faster_whisper import WhisperModel  # type: ignore

        path = model.resolved_path
        if not path.exists():
            raise EngineError(f"Папка модели не найдена: {path}")

        device, compute_type = self._resolve_device(model, hardware)
        threads = int(hardware.get("cpu_threads") or 0)
        if threads <= 0:
            threads = int(hardware.get("recommended_threads") or 0)

        kwargs: Dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": max(0, threads),
            "num_workers": 1,
            "local_files_only": True,
        }
        try:
            device_index = int(hardware.get("gpu_index") or 0)
            if device == "cuda" and device_index:
                kwargs["device_index"] = device_index
        except (TypeError, ValueError):
            pass

        log.info(
            "Загружаю модель %s (device=%s, compute=%s, threads=%s)",
            model.name, kwargs["device"], kwargs["compute_type"], kwargs["cpu_threads"],
        )
        started = time.time()
        try:
            self._model = WhisperModel(str(path), **_supported_kwargs(WhisperModel.__init__, kwargs))
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            hint = ""
            low = message.lower()
            if "cudnn" in low or "cublas" in low or "cuda" in low:
                hint = (
                    "\n\nПохоже на проблему с CUDA: нужны cuBLAS 12 и cuDNN 9.\n"
                    "Вариант 1: Сервис → Диагностика и восстановление → установить "
                    "CUDA-библиотеки (программа сама предложит это при выборе GPU).\n"
                    "Вариант 2 (Windows): скачать DLL из "
                    "https://github.com/Purfview/whisper-standalone-win/releases/tag/libs "
                    "и положить в .venv\\Lib\\site-packages.\n"
                    "Вариант 3: Настройки → Устройство → CPU (работает всегда)."
                )
            elif "memory" in low or "memoryerror" in low or "out of memory" in low:
                hint = "\n\nНе хватает памяти: выбери модель поменьше или включи compute_type=int8."
            elif "does not support" in low or "unsupported" in low:
                hint = (
                    "\n\nМодель не поддерживает выбранный compute_type. "
                    "Поставь в настройках «compute_type: auto» или default."
                )
            raise EngineError(f"Не удалось загрузить модель «{model.name}»: {message}{hint}") from exc

        self.loaded_model_id = model.model_id
        self.device = str(getattr(self._model, "device", kwargs["device"]))
        self.compute_type = str(getattr(self._model, "compute_type", kwargs["compute_type"]))
        log.info("Модель загружена за %.1f с", time.time() - started)

    # ------------------------------------------------------------------ #
    def _resolve_device(self, model: ModelRef, hardware: Dict[str, Any]) -> tuple[str, str]:
        device_pref = str(hardware.get("device") or "auto").lower()
        compute_pref = str(hardware.get("compute_type") or "auto").lower()
        cuda_ok = bool(hardware.get("cuda_ok"))
        vram_mb = int(hardware.get("vram_mb") or 0)

        if device_pref == "cpu":
            device = "cpu"
        elif device_pref == "cuda":
            if not cuda_ok:
                log.warning("CUDA запрошена, но недоступна — переключаюсь на CPU")
                device = "cpu"
            else:
                device = "cuda"
        else:  # auto
            device = "cuda" if cuda_ok else "cpu"

        if compute_pref not in ("auto", "default", ""):
            compute = compute_pref
        elif device == "cuda":
            compute = "float16" if vram_mb >= 6000 else "int8_float16"
        else:
            compute = "int8"

        # int8 на очень старых CPU без AVX может не поддерживаться
        if device == "cpu" and compute in ("int8", "int8_float16") and not _cpu_supports_int8():
            log.warning("CPU без поддержки int8 — переключаюсь на default")
            compute = "default"
        return device, compute

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
        model = self.model
        result = TranscriptResult(
            engine=self.name,
            device=self.device,
            compute_type=self.compute_type,
            model_name=options.get("model_name") or "",
        )

        use_batched = bool(options.get("batched", True))
        batch_size = int(options.get("batch_size") or 8)
        language = options.get("language") or None
        task = "translate" if options.get("translate") else "transcribe"
        use_vad = bool(options.get("vad", True))
        word_timestamps = bool(options.get("word_timestamps"))

        vad_parameters: Optional[Dict[str, Any]] = None
        if use_vad:
            vad_parameters = dict(VAD_DEFAULTS)
            custom_vad = options.get("vad_parameters") or {}
            if isinstance(custom_vad, dict):
                vad_parameters.update(custom_vad)

        common: Dict[str, Any] = {
            "language": language,
            "task": task,
            "beam_size": int(options.get("beam_size") or 5),
            "best_of": max(1, int(options.get("beam_size") or 5)),
            "temperature": [0.0, 0.2, 0.4],
            "vad_filter": use_vad,
            "word_timestamps": word_timestamps,
            "condition_on_previous_text": bool(options.get("condition_on_previous_text", True)),
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "compression_ratio_threshold": 2.4,
            # защита от «петель» и галлюцинаций (поддерживается в 1.1+)
            "repetition_penalty": float(options.get("repetition_penalty", 1.08)),
            "no_repeat_ngram_size": int(options.get("no_repeat_ngram_size", 3)),
        }
        if options.get("initial_prompt"):
            common["initial_prompt"] = str(options["initial_prompt"])
        if options.get("hotwords"):
            common["hotwords"] = str(options["hotwords"])
        if word_timestamps:
            common["hallucination_silence_threshold"] = float(
                options.get("hallucination_silence_threshold", 1.5)
            )

        log.debug(
            "faster-whisper: язык=%s task=%s beam=%s vad=%s words=%s batched=%s batch=%s "
            "prompt=%r hotwords=%r",
            language, task, common["beam_size"], use_vad, word_timestamps,
            use_batched, batch_size, bool(options.get("initial_prompt")), bool(options.get("hotwords")),
        )

        started = time.time()
        info = None
        segments_iter = None
        pipeline = None

        if use_batched:
            try:
                from faster_whisper import BatchedInferencePipeline  # type: ignore

                pipeline = BatchedInferencePipeline(model=model)
                kwargs = _supported_kwargs(pipeline.transcribe, dict(common, batch_size=batch_size))
                segments_iter, info = pipeline.transcribe(audio, **kwargs)
                log.info("Режим: batched (batch_size=%s)", kwargs.get("batch_size"))
            except TypeError as exc:
                log.warning("Batched-режим недоступен (%s) — переключаюсь на последовательный", exc)
                result.warnings.append("batched-режим недоступен, использован последовательный")
                pipeline = None
            except EngineError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("Сбой batched-режима: %s — пробую последовательный", exc)
                result.warnings.append(f"batched-режим упал ({exc}), использован последовательный")
                pipeline = None

        if segments_iter is None:
            kwargs = _supported_kwargs(model.transcribe, common)
            try:
                segments_iter, info = model.transcribe(audio, **kwargs)
            except TypeError as exc:
                # совсем старая версия библиотеки — убираем «новые» параметры
                log.warning("transcribe() не принял часть аргументов: %s — повторяю с минимумом", exc)
                minimal = {
                    key: common[key]
                    for key in ("language", "task", "beam_size", "best_of", "temperature",
                                "vad_filter", "word_timestamps", "initial_prompt")
                    if key in common
                }
                segments_iter, info = model.transcribe(audio, **_supported_kwargs(model.transcribe, minimal))
            except Exception as exc:  # noqa: BLE001
                raise EngineError(f"Ошибка распознавания: {exc}") from exc

        # --- итерация по сегментам --------------------------------------- #
        processed_audio = float(getattr(info, "duration_after_vad", 0) or 0) or duration
        total_audio = float(getattr(info, "duration", 0) or duration)
        result.language = str(getattr(info, "language", "") or (language or ""))
        result.language_probability = float(getattr(info, "language_probability", 0) or 0)
        result.audio_duration = total_audio

        last_report = 0.0
        try:
            for raw in segments_iter:
                if cancel is not None and cancel():
                    result.warnings.append("Прервано пользователем")
                    break

                segment = Segment(
                    start=float(getattr(raw, "start", 0.0)),
                    end=float(getattr(raw, "end", 0.0)),
                    text=(getattr(raw, "text", "") or "").strip(),
                    language=result.language,
                    avg_logprob=float(getattr(raw, "avg_logprob", 0.0) or 0.0),
                    no_speech_prob=float(getattr(raw, "no_speech_prob", 0.0) or 0.0),
                    compression_ratio=float(getattr(raw, "compression_ratio", 0.0) or 0.0),
                    words=[
                        Word(
                            start=float(getattr(w, "start", 0.0)),
                            end=float(getattr(w, "end", 0.0)),
                            text=str(getattr(w, "word", "") or ""),
                            probability=float(getattr(w, "probability", 0.0) or 0.0),
                        )
                        for w in (getattr(raw, "words", None) or [])
                    ],
                )
                result.segments.append(segment)

                if on_segment is not None:
                    try:
                        on_segment(segment)
                    except Exception:  # noqa: BLE001
                        pass

                if progress is not None:
                    now = time.time()
                    if now - last_report >= 0.15 or segment.end >= total_audio - 0.5:
                        last_report = now
                        elapsed = max(now - started, 1e-6)
                        fraction = min(1.0, segment.end / total_audio) if total_audio else 0.0
                        speed = segment.end / elapsed
                        eta = (total_audio - segment.end) / speed if speed > 0 else 0.0
                        progress(
                            fraction,
                            f"{int(segment.end // 60):02d}:{int(segment.end % 60):02d} / "
                            f"{int(total_audio // 60):02d}:{int(total_audio % 60):02d}"
                            f" · {speed:.1f}× · осталось ~{int(eta // 60)}:{int(eta % 60):02d}",
                        )
        except Exception as exc:  # noqa: BLE001
            if result.segments:
                result.warnings.append(f"Распознавание прервано ошибкой: {exc}")
                log.warning("Генератор сегментов упал после %d сегментов: %s", len(result.segments), exc)
            else:
                raise EngineError(f"Ошибка распознавания: {exc}") from exc
        finally:
            if pipeline is not None:
                try:
                    pipeline.release()
                except Exception:  # noqa: BLE001
                    pass

        result.elapsed_seconds = time.time() - started
        result.processed_duration = processed_audio
        log.info(
            "Готово: %d сегментов, язык=%s, %.1f× реального времени за %.1f с",
            len(result.segments), result.language, result.speed_x, result.elapsed_seconds,
        )
        return result

    # ------------------------------------------------------------------ #
    def unload(self) -> None:
        model = self._model
        self._model = None
        self.loaded_model_id = None
        try:
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "unload"):
                inner.unload()
        except Exception:  # noqa: BLE001
            pass
        del model


def _cpu_supports_int8() -> bool:
    """Проверяет поддержку AVX (нужна для int8-ядер CTranslate2)."""
    try:
        import ctranslate2  # type: ignore

        info = ctranslate2.get_cpu_features() if hasattr(ctranslate2, "get_cpu_features") else ""
        if info:
            return "avx" in str(info).lower()
    except Exception:  # noqa: BLE001
        pass
    import platform

    if platform.machine().lower() in ("arm64", "aarch64"):
        return True
    # x86-64 без AVX сегодня встречается редко — разрешаем, движок сам откатится
    return True
