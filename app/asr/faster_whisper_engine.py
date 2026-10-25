"""faster-whisper engine (CTranslate2).

The primary LocalASR engine: ~4x faster than openai/whisper, with int8
quantization, Silero VAD, hotwords and batched GPU decoding. Runs on CPU
and CUDA without PyTorch.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable, Dict, List, Optional

from ..constants import ENGINE_FASTER_WHISPER
from ..core.postprocess import Segment, Word
from ..i18n import _
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

# Silero VAD parameters tuned for lectures: long pauses between blocks,
# short ones inside a phrase (otherwise text merges and timecodes drift).
VAD_DEFAULTS: Dict[str, Any] = {
    "threshold": 0.5,
    "min_silence_duration_ms": 400,
    "min_speech_duration_ms": 250,
    "speech_pad_ms": 300,
}


def _supported_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs the installed library version does not accept — the usual
    cause of wrapper crashes after a faster-whisper update.
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
            raise EngineError(_("Model folder not found: {path}").format(path=path))

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
            _("Loading model %s (device=%s, compute=%s, threads=%s)"),
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
                hint = _(
                    "\n\nLooks like a CUDA problem: cuBLAS 12 and cuDNN 9 are required.\n"
                    "Option 1: Tools → Diagnostics & recovery → install the CUDA "
                    "libraries (the app also offers this when a GPU is selected).\n"
                    "Option 2 (Windows): download the DLLs from "
                    "https://github.com/Purfview/whisper-standalone-win/releases/tag/libs "
                    "and put them into the environment's Lib\\site-packages.\n"
                    "Option 3: Settings → Device → CPU (always works)."
                )
            elif "memory" in low or "memoryerror" in low or "out of memory" in low:
                hint = _("\n\nNot enough memory: pick a smaller model or enable compute_type=int8.")
            elif "does not support" in low or "unsupported" in low:
                hint = _(
                    "\n\nThe model does not support the selected compute_type. "
                    "Set \"compute_type: auto\" or default in Settings."
                )
            raise EngineError(_("Failed to load model \"{name}\": {message}{hint}").format(
                name=model.name, message=message, hint=hint)) from exc

        self.loaded_model_id = model.model_id
        self.device = str(getattr(self._model, "device", kwargs["device"]))
        self.compute_type = str(getattr(self._model, "compute_type", kwargs["compute_type"]))
        log.info(_("Model loaded in %.1f s"), time.time() - started)

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
                log.warning(_("CUDA requested but unavailable — switching to CPU"))
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

        # int8 may be unsupported on very old CPUs without AVX
        if device == "cpu" and compute in ("int8", "int8_float16") and not _cpu_supports_int8():
            log.warning(_("CPU without int8 support — switching to default"))
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
            # loop/hallucination guard (supported since 1.1+)
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
            "faster-whisper: language=%s task=%s beam=%s vad=%s words=%s batched=%s batch=%s "
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
                log.info(_("Mode: batched (batch_size=%s)"), kwargs.get("batch_size"))
            except TypeError as exc:
                log.warning(_("Batched mode unavailable (%s) — switching to sequential"), exc)
                result.warnings.append(_("batched mode unavailable, sequential used"))
                pipeline = None
            except EngineError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning(_("Batched mode failed: %s — trying sequential"), exc)
                result.warnings.append(_("batched mode failed ({exc}), sequential used").format(exc=exc))
                pipeline = None

        if segments_iter is None:
            kwargs = _supported_kwargs(model.transcribe, common)
            try:
                segments_iter, info = model.transcribe(audio, **kwargs)
            except TypeError as exc:
                # a very old library version — drop the "new" parameters
                log.warning(_("transcribe() rejected some arguments: %s — retrying with a minimal set"), exc)
                minimal = {
                    key: common[key]
                    for key in ("language", "task", "beam_size", "best_of", "temperature",
                                "vad_filter", "word_timestamps", "initial_prompt")
                    if key in common
                }
                segments_iter, info = model.transcribe(audio, **_supported_kwargs(model.transcribe, minimal))
            except Exception as exc:  # noqa: BLE001
                raise EngineError(_("Recognition error: {exc}").format(exc=exc)) from exc

        # --- segment iteration --------------------------------------------- #
        processed_audio = float(getattr(info, "duration_after_vad", 0) or 0) or duration
        total_audio = float(getattr(info, "duration", 0) or duration)
        result.language = str(getattr(info, "language", "") or (language or ""))
        result.language_probability = float(getattr(info, "language_probability", 0) or 0)
        result.audio_duration = total_audio

        last_report = 0.0
        try:
            for raw in segments_iter:
                if cancel is not None and cancel():
                    result.warnings.append(_("Interrupted by user"))
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
                        # segment.end may overshoot total_audio (VAD padding) —
                        # clamp so the ETA never shows a negative "time left"
                        eta = max(0.0, (total_audio - segment.end) / speed) if speed > 0 else 0.0
                        progress(
                            fraction,
                            _("{pos} / {total} · {speed:.1f}× · ~{eta} left").format(
                                pos=f"{int(segment.end // 60):02d}:{int(segment.end % 60):02d}",
                                total=f"{int(total_audio // 60):02d}:{int(total_audio % 60):02d}",
                                speed=speed,
                                eta=f"{int(eta // 60)}:{int(eta % 60):02d}",
                            ),
                        )
        except Exception as exc:  # noqa: BLE001
            if result.segments:
                result.warnings.append(_("Recognition interrupted by an error: {exc}").format(exc=exc))
                log.warning(_("Segment generator failed after %d segments: %s"), len(result.segments), exc)
            else:
                raise EngineError(_("Recognition error: {exc}").format(exc=exc)) from exc
        finally:
            if pipeline is not None:
                try:
                    pipeline.release()
                except Exception:  # noqa: BLE001
                    pass

        result.elapsed_seconds = time.time() - started
        result.processed_duration = processed_audio
        log.info(
            _("Done: %d segments, language=%s, %.1f× realtime in %.1f s"),
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
    """Check AVX support (needed by CTranslate2 int8 kernels)."""
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
    # x86-64 without AVX is rare today — allow it, the engine falls back itself
    return True
