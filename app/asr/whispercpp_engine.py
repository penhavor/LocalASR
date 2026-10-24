"""Движок whisper.cpp через pywhispercpp (GGML-модели).

Когда он полезен:
  * слабые CPU и мало RAM — квантованные q5_0/q8_0 модели весят в 3–5 раз меньше;
  * нужен максимум скорости на CPU без установки CTranslate2;
  * хочется «классический» whisper.cpp с flash-attention и tinydiarize.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..constants import ENGINE_WHISPERCPP
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

log = get_logger("asr.whispercpp")

def _filter_params(model: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Оставляет только те ключи, которые понимает установленная pywhispercpp."""
    allowed: Optional[set] = None
    try:
        schema = model.get_params_schema()  # constants.PARAMS_SCHEMA
        if isinstance(schema, dict) and schema:
            allowed = set(schema.keys())
    except Exception:  # noqa: BLE001
        allowed = None

    if not allowed:
        try:
            import inspect

            signature = inspect.signature(type(model).__init__)
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
                return dict(params)
        except (TypeError, ValueError):
            return dict(params)
        return dict(params)

    filtered = {key: value for key, value in params.items() if key in allowed}
    dropped = sorted(set(params) - set(filtered))
    if dropped:
        log.info("pywhispercpp не поддерживает параметры %s — пропускаю", ", ".join(dropped))
    return filtered


def resolve_ggml_file(model: ModelRef) -> Path:
    """Находит .bin-файл модели (папка может содержать несколько ggml-файлов)."""
    path = model.path
    if path.is_file():
        return path
    if model.subfolder:
        candidate = path / model.subfolder
        if candidate.is_file():
            return candidate
    preferred: List[Path] = []
    others: List[Path] = []
    try:
        for child in sorted(path.rglob("*.bin")):
            name = child.name.lower()
            if "encoder" in name or "coreml" in name:
                continue
            (preferred if name.startswith("ggml") else others).append(child)
    except OSError as exc:
        raise EngineError(f"Не удалось прочитать папку модели {path}: {exc}") from exc

    for group in (preferred, others):
        if group:
            # если файлов несколько — берём самый большой (обычно это полная модель)
            return max(group, key=lambda p: p.stat().st_size)
    raise EngineError(f"В папке {path} не найден GGML-файл модели (*.bin)")


class WhisperCppEngine(ASREngine):
    name = ENGINE_WHISPERCPP
    packages = ["pywhispercpp>=1.4.0"]
    import_name = "pywhispercpp"

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        try:
            from pywhispercpp.model import Model as CppModel  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Не удалось импортировать pywhispercpp: {exc}\n"
                "Установи движок: Настройки → Движки → whisper.cpp → «Установить»."
            ) from exc

        ggml = resolve_ggml_file(model)
        threads = int(hardware.get("cpu_threads") or 0) or int(hardware.get("recommended_threads") or 4)
        threads = max(1, min(32, threads))

        # whisper.cpp очень разговорчив: уводим его лог в файл проекта,
        # чтобы не засорять консоль (файл пересоздаётся при каждой загрузке)
        log_target: Any = False
        if not hardware.get("whispercpp_verbose"):
            try:
                from .. import paths as _paths

                cpp_log = _paths.logs_dir() / "whispercpp.log"
                cpp_log.write_text(f"# whisper.cpp log · {ggml.name}\n", encoding="utf-8")
                log_target = str(cpp_log)
            except OSError:
                log_target = False

        params: Dict[str, Any] = {
            "n_threads": threads,
            "print_progress": False,
            "print_realtime": False,
            "print_timestamps": False,
            "print_special": False,
            "suppress_non_speech_tokens": True,
            "no_context": True,     # меньше «залипаний» и петель на длинном аудио
            "temperature_inc": 0.2,
            "entropy_thold": 2.4,
            "logprob_thold": -1.0,
        }
        use_gpu = bool(hardware.get("whispercpp_gpu", True))
        context_params = {"use_gpu": use_gpu, "flash_attn": bool(hardware.get("flash_attn", False))}

        log.info("Загружаю GGML-модель %s (threads=%d)", ggml.name, threads)
        started = time.time()
        try:
            self._model = CppModel(
                str(ggml),
                redirect_whispercpp_logs_to=log_target,
                context_params=context_params,
                **params,
            )
        except TypeError:
            # старая версия pywhispercpp без context_params
            self._model = CppModel(str(ggml), **params)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"Не удалось загрузить GGML-модель: {exc}") from exc

        self.loaded_model_id = model.model_id
        self.device = "cpu/gpu(ggml)"
        self.compute_type = ggml.stem
        log.info("GGML-модель загружена за %.1f с", time.time() - started)

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
        import numpy as np

        model = self.model
        result = TranscriptResult(
            engine=self.name,
            device=self.device,
            compute_type=self.compute_type,
            model_name=options.get("model_name") or "",
            audio_duration=duration,
        )

        waveform = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        if waveform.ndim != 1 or waveform.size == 0:
            raise EngineError("whisper.cpp ожидает одномерный массив float32")

        language = options.get("language") or None
        beam_size = int(options.get("beam_size") or 5)
        transcribe_params: Dict[str, Any] = {
            "language": language if language else "auto",
            "translate": bool(options.get("translate")),
            "no_timestamps": False,
            "single_segment": False,
            "suppress_blank": True,
            "suppress_non_speech_tokens": True,
            "temperature": 0.0,
            "beam_search": {"beam_size": max(1, beam_size), "patience": -1.0},
        }

        # ВАЖНО: в свежем whisper.cpp встроенный VAD требует внешний файл модели
        # (ggml-silero-v5.12.bin). Без него whisper_full молча возвращает ноль
        # сегментов («failed to compute VAD»), поэтому включаем VAD только если
        # файл реально существует.
        vad_model_path = options.get("vad_model_path") or ""
        if options.get("vad") and vad_model_path and Path(vad_model_path).is_file():
            transcribe_params["vad"] = True
            transcribe_params["vad_model_path"] = str(vad_model_path)
        else:
            transcribe_params["vad"] = False
            if options.get("vad"):
                log.info("VAD в whisper.cpp выключен: нет файла модели VAD (не критично)")
        if options.get("initial_prompt"):
            transcribe_params["initial_prompt"] = str(options["initial_prompt"])
            transcribe_params["carry_initial_prompt"] = bool(options.get("carry_initial_prompt", False))

        # разные версии pywhispercpp поддерживают разный набор параметров —
        # лишний ключ привёл бы к KeyError, поэтому фильтруем по схеме
        transcribe_params = _filter_params(model, transcribe_params)

        started = time.time()

        def _progress_callback(segment) -> None:
            try:
                text = (segment.text or "").strip()
                start = float(segment.t0) / 100.0
                end = float(segment.t1) / 100.0
            except Exception:  # noqa: BLE001
                return
            seg = Segment(start=start, end=end, text=text, language=result.language)
            result.segments.append(seg)
            if on_segment is not None:
                try:
                    on_segment(seg)
                except Exception:  # noqa: BLE001
                    pass
            if progress is not None and duration > 0:
                elapsed = max(time.time() - started, 1e-6)
                speed = end / elapsed
                eta = max(0.0, (duration - end) / speed) if speed > 0 else 0.0
                progress(
                    min(1.0, end / duration),
                    f"{int(end // 60):02d}:{int(end % 60):02d} / "
                    f"{int(duration // 60):02d}:{int(duration % 60):02d}"
                    f" · {speed:.1f}× · осталось ~{int(eta // 60)}:{int(eta % 60):02d}",
                )

        def _abort() -> bool:
            return bool(cancel is not None and cancel())

        kwargs: Dict[str, Any] = {
            "new_segment_callback": _progress_callback,
            "abort_callback": _abort,
        }
        # фильтруем параметры под конкретную версию pywhispercpp
        try:
            import inspect

            allowed = set(inspect.signature(model.transcribe).parameters)
            if "kwargs" not in allowed:
                kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        except (TypeError, ValueError):
            pass

        segments = self._run(model, waveform, kwargs, transcribe_params, started)

        # страховка: если VAD/фильтры «съели» всё аудио — повторяем мягче
        if not result.segments:
            log.warning("Первый проход не дал сегментов — повторяю без VAD и с no_timestamps=False")
            result.warnings.append("первый проход пуст, повтор с отключённым VAD")
            retry_params = dict(transcribe_params)
            retry_params["vad"] = False
            retry_params.pop("vad_model_path", None)
            retry_params["suppress_blank"] = False
            retry_params.pop("beam_search", None)
            retry_params["greedy"] = {"best_of": 1}
            segments = self._run(model, waveform, kwargs, retry_params, started)

        if cancel is not None and cancel():
            result.warnings.append("Прервано пользователем")

        # если колбэк не сработал (старая версия), берём сегменты из ответа
        if not result.segments and segments:
            for raw in segments:
                seg = Segment(
                    start=float(getattr(raw, "t0", 0)) / 100.0,
                    end=float(getattr(raw, "t1", 0)) / 100.0,
                    text=(getattr(raw, "text", "") or "").strip(),
                )
                result.segments.append(seg)
                if on_segment is not None:
                    on_segment(seg)

        if result.segments:
            result.processed_duration = max(s.end for s in result.segments)
        else:
            result.processed_duration = duration
        result.elapsed_seconds = time.time() - started
        result.language = language or "auto"
        return result

    # ------------------------------------------------------------------ #
    def _run(self, model, waveform, kwargs: Dict[str, Any], params: Dict[str, Any], started: float):
        """Один вызов transcribe с перехватом несовместимости параметров."""
        try:
            return model.transcribe(waveform, **kwargs, **params)
        except TypeError as exc:
            log.warning("pywhispercpp не принял часть параметров (%s) — повторяю с минимумом", exc)
            minimal = {k: v for k, v in params.items() if k in ("language", "translate")}
            return model.transcribe(waveform, new_segment_callback=kwargs.get("new_segment_callback"), **minimal)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"Ошибка whisper.cpp: {exc}") from exc
