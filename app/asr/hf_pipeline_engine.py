"""Generic engine: ANY Hugging Face ASR model via transformers.pipeline.

Covers models outside faster-whisper/whisper.cpp/Vosk/Qwen3-ASR/GigaAM:
original Whisper builds, wav2vec2, HuBERT, Data2Vec, MMS, Seamless-M4T,
Parakeet/Canary, Distil-Whisper and thousands of other repos with
pipeline_tag=automatic-speech-recognition.

How it works: the model is loaded with the standard pipeline(); long audio
is split on silence into ~27-28.5 s chunks (strictly shorter than the 30 s
window of Whisper-like models) processed in batches; language/translation
go into generate_kwargs when the model accepts them; progress and cancel
behave like the other engines.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..constants import ENGINE_HF, SAMPLE_RATE
from ..core.postprocess import Segment
from ..i18n import _
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

# Chunks STRICTLY shorter than 30 s: for Whisper-like seq2seq models an input
# longer than 3000 mel frames switches generate() to long-form mode which
# REQUIRES timestamps (and crashes models without no_timestamps_token_id).
# 27 s + up to 1.5 s of silence-boundary search = ~28.5 s max — always short-form.
HF_CHUNK_SECONDS = 27.0
HF_SEARCH_SECONDS = 1.5
# chunks sent to the pipeline per batch
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
        # pipeline audio input key: "array" in old transformers, "raw" in new
        self._input_key: str = "array"
        # new transformers request timestamps by default — some models lack the
        # generation_config settings for them; we do not need pipeline
        # timestamps (segments = chunk bounds), so disable — but ONLY for
        # seq2seq: the CTC branch errors on return_timestamps=False
        self._return_timestamps_supported: bool = True
        self._pass_no_timestamps: bool = False

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except Exception as exc:  # noqa: BLE001
            raise EngineError(_(
                "Failed to import torch/transformers: {exc}\n"
                "Missing components are offered for automatic installation when a model "
                "is started (or: Tools → Diagnostics & recovery)."
            ).format(exc=exc)) from exc

        path = str(model.resolved_path)
        device, dtype = self._pick_device(hardware)
        started = time.time()

        last_error: Optional[Exception] = None
        for attempt_dtype in (dtype, torch.float32):
            # transformers 4.x takes torch_dtype, 5.x renamed it to dtype
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
                    log.debug("pipeline rejected %s: %s", dtype_kwarg, exc)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    log.warning("pipeline(%s) failed to build with %s: %s", path, attempt_dtype, exc)
                    break
            if self._pipe is not None:
                break
        if self._pipe is None:
            from .torch_engines import missing_files_hint

            log.error("HF-pipeline failed to build for %s: %s", path, last_error, exc_info=last_error)
            raise EngineError(_(
                "Failed to load model \"{name}\" via transformers: {err}\n\n"
                "It may not be an ASR model, or it needs rare dependencies — "
                "see the error text above."
            ).format(name=model.name, err=last_error)
                + missing_files_hint(last_error or EngineError(""))
            ) from last_error

        # Some models carry return_timestamps=True in generation_config —
        # transformers computes `return_timestamps or config.return_timestamps`,
        # so our False is overridden and decoding crashes on models without
        # no_timestamps_token_id. We do not need pipeline timestamps, so
        # suppress the flag right in the config.
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
            _("HF-pipeline loaded: %s (%s, %s) in %.1f s"),
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
        # Beam search instead of greedy decoding: without it the pipeline
        # generates with num_beams=1, underusing the model weights. If a model
        # rejects the parameter, the _call fallback cascade retries without
        # generate_kwargs.
        try:
            beam = int(options.get("beam_size") or 0)
        except (TypeError, ValueError):
            beam = 0
        if beam > 1:
            kwargs["num_beams"] = min(beam, 8)
        return kwargs

    def _call(self, pieces: List[np.ndarray], generate_kwargs: Dict[str, Any]) -> List[str]:
        """Pipeline call with a fallback cascade (universality over speed).

        Attempts: batched call -> batched with the other input key
        ("array" <-> "raw") -> one by one -> one by one without language/task.
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
                    # an old pipeline version does not know return_timestamps
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
                    self._input_key = other_key   # remember the working key
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.debug("pipeline attempt %d failed: %s", index + 1, exc)
        else:
            raise EngineError(_("Transformers pipeline could not recognize the chunk: {err}").format(err=last_error))

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
            raise EngineError(_("HF-pipeline is not loaded"))
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
                warnings.append(_("Recognition stopped by user"))
                break
            batch_bounds = bounds[index: index + batch_size]
            pieces = [audio[start:end] for start, end in batch_bounds]
            try:
                texts = self._call(pieces, dict(generate_kwargs))
            except Exception as exc:  # noqa: BLE001
                raise EngineError(_(
                    "Transformers pipeline could not recognize the chunk: {err}"
                ).format(err=exc)) from exc

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
                    _("Recognizing: {done:.0f} s of {total:.0f} s").format(
                        done=done / SAMPLE_RATE, total=duration),
                )

        if options.get("word_timestamps"):
            warnings.append(_("HF-pipeline: word timestamps not computed — segments follow audio chunks"))

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
