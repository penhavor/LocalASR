"""Универсальный движок: ЛЮБАЯ ASR-модель Hugging Face через transformers.pipeline.

Закрывает модели, которые не относятся к faster-whisper/whisper.cpp/Vosk/
Qwen3-ASR/GigaAM: оригинальные сборки Whisper (openai/whisper-*), wav2vec2,
HuBERT, Data2Vec, MMS, Seamless-M4T, Parakeet/Canary (NVIDIA), Distil-Whisper
и тысячи других репозиториев с pipeline_tag=automatic-speech-recognition.

Как это работает:
  * модель грузится стандартным ``pipeline("automatic-speech-recognition", …)``
    — transformers сам определяет архитектуру (seq2seq / CTC / …);
  * длинное аудио режется по тишине на куски ~27–28.5 секунд (строго короче
    30-секундного окна Whisper-подобных моделей), куски считаются батчами
    (на GPU — заметный выигрыш);
  * язык/перевод передаются в generate_kwargs, если модель их понимает
    (не понимает — тихо повторяем вызов без них);
  * прогресс и отмена — как у остальных движков (между кусками/батчами).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..constants import ENGINE_HF, SAMPLE_RATE
from ..core.postprocess import Segment
from ..logutils import get_logger
from .base import (
    CancelToken,
    EngineError,
    ModelRef,
    ProgressCallback,
    SegmentCallback,
    TranscriptResult,
)
from .torch_engines import TorchASREngine, iter_smart_chunks

log = get_logger("asr.hf")

# Куски СТРОГО короче 30 секунд: у Whisper-подобных seq2seq-моделей вход
# длиннее 3000 mel-кадров переключает generate() в long-form режим, который
# ОБЯЗАТЕЛЬНО требует таймкоды (и падает на моделях без no_timestamps_token_id).
# 27 с + до 1.5 с поиска границы тишины = максимум ~28.5 с — всегда short-form.
HF_CHUNK_SECONDS = 27.0
HF_SEARCH_SECONDS = 1.5
# сколько кусков отправляем в pipeline одним батчем
HF_BATCH_CUDA = 8
HF_BATCH_CPU = 2


class HFPipelineEngine(TorchASREngine):
    name = ENGINE_HF
    packages = [
        "torch>=2.6",
        "transformers>=4.49",
        "accelerate>=1.0",
        "soundfile>=0.12",
    ]
    import_name = "transformers"
    import_names = ("torch", "transformers")
    chunk_seconds = HF_CHUNK_SECONDS

    def __init__(self) -> None:
        super().__init__()
        self._pipe: Any = None
        # ключ аудио-входа pipeline: в старых transformers «array», в новых «raw»
        self._input_key: str = "array"
        # новые transformers по умолчанию просят таймкоды — некоторым моделям
        # для них не хватает настроек generation_config; нам таймкоды pipeline
        # не нужны (сегменты = границы кусков), поэтому отключаем — но ТОЛЬКО
        # для seq2seq: CTC-ветка на return_timestamps=False отвечает ошибкой
        self._return_timestamps_supported: bool = True
        self._pass_no_timestamps: bool = False

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Не удалось импортировать torch/transformers: {exc}\n"
                "Установи движок: Сервис → Настройки → Движки → Transformers → «Установить»."
            ) from exc

        path = str(model.resolved_path)
        device, dtype = self._pick_device(hardware)
        started = time.time()

        last_error: Optional[Exception] = None
        for attempt_dtype in (dtype, torch.float32):
            # transformers 4.x принимает torch_dtype, 5.x переименовал в dtype
            for dtype_kwarg in ("torch_dtype", "dtype"):
                try:
                    self._pipe = hf_pipeline(
                        task="automatic-speech-recognition",
                        model=path,
                        device=torch.device(device),
                        trust_remote_code=True,
                        **{dtype_kwarg: attempt_dtype},
                    )
                    self.compute_type = str(attempt_dtype).replace("torch.", "")
                    break
                except TypeError as exc:
                    if dtype_kwarg not in str(exc):
                        last_error = exc
                        break
                    last_error = exc
                    log.debug("pipeline не принял %s: %s", dtype_kwarg, exc)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    log.warning("pipeline(%s) не собрался с %s: %s", path, attempt_dtype, exc)
                    break
            if self._pipe is not None:
                break
        if self._pipe is None:
            raise EngineError(
                f"Не удалось загрузить модель «{model.name}» через transformers: {last_error}\n\n"
                "Возможно, это не ASR-модель, либо она требует редкие зависимости — "
                "смотри текст ошибки выше."
            ) from last_error

        # Некоторые модели несут в generation_config return_timestamps=True —
        # transformers делает `return_timestamps or config.return_timestamps`,
        # т.е. наш False перебивается, и на моделях без no_timestamps_token_id
        # декодирование падает. Таймкоды pipeline нам не нужны (сегменты =
        # границы кусков), поэтому гасим флаг прямо в конфиге.
        try:
            gen_cfg = getattr(self._pipe, "generation_config", None)
            if gen_cfg is not None and getattr(gen_cfg, "return_timestamps", False):
                gen_cfg.return_timestamps = False
        except Exception:  # noqa: BLE001
            pass

        self._model = getattr(self._pipe, "model", None)
        pipe_type = str(getattr(self._pipe, "type", "") or "")
        self._pass_no_timestamps = pipe_type in ("seq2seq_whisper", "seq2seq")
        self.loaded_model_id = model.model_id
        log.info(
            "HF-pipeline загружен: %s (%s, %s) за %.1f с",
            model.model_id, device, self.compute_type, time.time() - started,
        )

    # ------------------------------------------------------------------ #
    def unload(self) -> None:
        self._pipe = None
        super().unload()

    # ------------------------------------------------------------------ #
    def _generate_kwargs(self, options: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        language = str(options.get("language") or "").strip()
        if language:
            kwargs["language"] = language
        if options.get("translate"):
            kwargs["task"] = "translate"
        return kwargs

    def _call(self, pieces: List[np.ndarray], generate_kwargs: Dict[str, Any]) -> List[str]:
        """Вызов pipeline c каскадом откатов (универсальность важнее скорости).

        Порядок попыток:
          1. батч кусков одним вызовом (быстро на GPU);
          2. то же с другим ключом входа («array» <-> «raw» в разных версиях);
          3. поштучно (батчинг в pipeline капризен: разные наборы ключей у
             последнего элемента, модели без поддержки batch и т.п.);
          4. поштучно без language/task (модель может их не понимать).
        """

        def one_call(items: Any, batch_size: Optional[int], gk: Dict[str, Any]) -> List[Dict]:
            for attempt in (0, 1):
                kwargs: Dict[str, Any] = {}
                if batch_size and isinstance(items, list) and len(items) > 1:
                    kwargs["batch_size"] = batch_size
                if gk:
                    kwargs["generate_kwargs"] = gk
                if self._return_timestamps_supported and self._pass_no_timestamps:
                    kwargs["return_timestamps"] = False
                try:
                    result = self._pipe(items, **kwargs)
                    break
                except TypeError as exc:
                    # старая версия pipeline не знает return_timestamps
                    if attempt == 0 and "return_timestamps" in str(exc):
                        self._return_timestamps_supported = False
                        continue
                    raise
            if isinstance(result, dict):
                return [result]
            return list(result)

        def inputs_for(key: str, batch: bool) -> Any:
            items = [{key: piece, "sampling_rate": SAMPLE_RATE} for piece in pieces]
            return items if (batch and len(items) > 1) else items

        def singles(key: str, gk: Dict[str, Any]) -> List[Dict]:
            outputs: List[Dict] = []
            for item in inputs_for(key, batch=False):
                outputs.extend(one_call(item, None, gk))
            return outputs

        other_key = "raw" if self._input_key == "array" else "array"
        attempts = (
            (lambda: one_call(inputs_for(self._input_key, True), len(pieces), generate_kwargs)),
            (lambda: one_call(inputs_for(other_key, True), len(pieces), generate_kwargs)),
            (lambda: singles(self._input_key, generate_kwargs)),
            (lambda: singles(other_key, generate_kwargs)),
            (lambda: singles(self._input_key, {})),
        )
        last_error: Optional[Exception] = None
        for index, attempt in enumerate(attempts):
            try:
                outputs = attempt()
                if index == 1:
                    self._input_key = other_key   # запоминаем рабочий ключ
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.debug("pipeline-попытка %d не удалась: %s", index + 1, exc)
        else:
            raise EngineError(f"Transformers-pipeline не смог распознать фрагмент: {last_error}")

        texts: List[str] = []
        for output in outputs:
            if isinstance(output, dict):
                texts.append(str(output.get("text") or ""))
            else:
                texts.append(str(output or ""))
        return texts

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
        self._ensure_model()
        if self._pipe is None:
            raise EngineError("HF-pipeline не загружен")
        started = time.time()
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        total = len(audio) or 1

        bounds = list(
            iter_smart_chunks(
                audio, SAMPLE_RATE, self.chunk_seconds,
                search_seconds=HF_SEARCH_SECONDS,
                use_vad=bool(options.get("vad", True)),
            )
        )
        segments: List[Segment] = []
        warnings: List[str] = []
        generate_kwargs = self._generate_kwargs(options)

        batch_size = HF_BATCH_CUDA if self.device == "cuda" else HF_BATCH_CPU
        if options.get("batched") is False:
            batch_size = 1
        elif options.get("batch_size"):
            batch_size = max(1, min(16, int(options["batch_size"])))

        index = 0
        count = len(bounds)
        while index < count:
            if cancel and cancel():
                warnings.append("Распознавание остановлено пользователем")
                break
            batch_bounds = bounds[index: index + batch_size]
            pieces = [audio[start:end] for start, end in batch_bounds]
            try:
                texts = self._call(pieces, dict(generate_kwargs))
            except Exception as exc:  # noqa: BLE001
                raise EngineError(
                    f"Transformers-pipeline не смог распознать фрагмент: {exc}"
                ) from exc

            for (start, end), text in zip(batch_bounds, texts):
                text = str(text or "").strip()
                if text:
                    segment = Segment(start=start / SAMPLE_RATE, end=end / SAMPLE_RATE, text=text)
                    segments.append(segment)
                    if on_segment:
                        on_segment(segment)
            index += len(batch_bounds)
            if progress:
                done = batch_bounds[-1][1]
                progress(
                    min(1.0, done / total),
                    f"Распознаю: {done / SAMPLE_RATE:.0f} с из {duration:.0f} с",
                )

        if options.get("word_timestamps"):
            warnings.append("HF-pipeline: таймкоды слов не считались — сегменты по кускам аудио")

        return TranscriptResult(
            segments=segments,
            language=str(options.get("language") or ""),
            audio_duration=duration,
            processed_duration=duration,
            elapsed_seconds=time.time() - started,
            engine=self.name,
            device=self.device,
            compute_type=self.compute_type,
            warnings=warnings,
        )
