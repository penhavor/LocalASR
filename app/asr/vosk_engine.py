"""Vosk engine (Kaldi).

The lightest option: a 45 MB model starts instantly and uses ~300 MB RAM.
Accuracy is notably below Whisper, but it runs on very weak hardware offline.
Limitations: no punctuation/capitals (added by simple post-processing),
no translation.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..constants import ENGINE_VOSK, SAMPLE_RATE
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

log = get_logger("asr.vosk")

CHUNK_SECONDS = 4.0
SENTENCE_BREAK_WORDS = {
    "и", "а", "но", "что", "как", "то", "же", "ли", "бы", "в", "на", "с", "к", "из", "от", "до",
    "по", "при", "о", "об", "про", "для", "у", "над", "под", "между", "без",
}


def resolve_model_dir(model: ModelRef) -> Path:
    """Vosk expects a folder with am/, graph/, conf/ (sometimes one level deeper)."""
    path = model.path
    if path.is_file():
        raise EngineError(_("A Vosk model must be a folder, not a file: {path}").format(path=path))
    if (path / "am").is_dir():
        return path
    try:
        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / "am").is_dir():
                return child
    except OSError:
        pass
    raise EngineError(_(
        "Folder {path} has no Vosk model structure (expected an am/ subfolder). "
        "The archive may be extracted incompletely."
    ).format(path=path))


class VoskEngine(ASREngine):
    name = ENGINE_VOSK
    packages = ["vosk>=0.3.45"]
    import_name = "vosk"

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        try:
            from vosk import Model as VoskModel, SetLogLevel  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise EngineError(_(
                "Failed to import vosk: {exc}\n"
                "Missing components are offered for automatic installation when a model "
                "is started (or: Tools → Diagnostics & recovery)."
            ).format(exc=exc)) from exc

        folder = resolve_model_dir(model)
        try:
            SetLogLevel(-1)
        except Exception:  # noqa: BLE001
            pass

        log.info(_("Loading Vosk model %s"), folder)
        started = time.time()
        try:
            self._model = VoskModel(str(folder))
        except Exception as exc:  # noqa: BLE001
            raise EngineError(_(
                "Vosk could not load the model from {folder}: {exc}\n"
                "Check that the archive is fully extracted and there is enough disk space."
            ).format(folder=folder, exc=exc)) from exc
        self.loaded_model_id = model.model_id
        self.device = "cpu"
        self.compute_type = "kaldi"
        log.info(_("Vosk model loaded in %.1f s"), time.time() - started)

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

        try:
            from vosk import KaldiRecognizer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise EngineError(_("Failed to import vosk: {exc}").format(exc=exc)) from exc

        model = self.model
        result = TranscriptResult(
            engine=self.name,
            device="cpu",
            compute_type="kaldi",
            model_name=options.get("model_name") or "",
            audio_duration=duration,
            language="ru",
        )

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise EngineError(_("Vosk expects a 1-D float32 array"))

        try:
            recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        except TypeError:
            recognizer = KaldiRecognizer(model, float(SAMPLE_RATE))

        try:
            recognizer.SetWords(True)
        except Exception:  # noqa: BLE001
            pass

        hotwords = options.get("hotwords")
        if hotwords:
            grammar = _hotwords_to_grammar(hotwords)
            if grammar:
                try:
                    recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)
                    recognizer.SetWords(True)
                    result.warnings.append(_("Hotword grammar applied (restricts vocabulary)"))
                except Exception as exc:  # noqa: BLE001
                    log.debug(_("Hotwords were not applied: %s"), exc)

        pcm = np.clip(waveform, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2").tobytes()

        chunk_bytes = int(SAMPLE_RATE * 2 * CHUNK_SECONDS)
        total = len(pcm16)
        started = time.time()
        last_report = 0.0

        for offset in range(0, total, chunk_bytes):
            if cancel is not None and cancel():
                result.warnings.append(_("Interrupted by user"))
                break
            block = pcm16[offset:offset + chunk_bytes]
            try:
                accepted = recognizer.AcceptWaveform(block)
            except Exception as exc:  # noqa: BLE001
                raise EngineError(_("Vosk failed while processing audio: {exc}").format(exc=exc)) from exc

            if accepted:
                payload = _safe_json(recognizer.Result())
                segment = _segment_from_payload(payload)
                if segment is not None:
                    result.segments.append(segment)
                    if on_segment is not None:
                        on_segment(segment)

            if progress is not None:
                now = time.time()
                if now - last_report >= 0.2:
                    last_report = now
                    position = min(duration, (offset + len(block)) / (SAMPLE_RATE * 2))
                    elapsed = max(now - started, 1e-6)
                    speed = position / elapsed
                    eta = (duration - position) / speed if speed > 0 else 0
                    progress(
                        min(1.0, position / duration) if duration else 0.0,
                        _("{pos} / {total} · {speed:.1f}× · ~{eta} left").format(
                            pos=f"{int(position // 60):02d}:{int(position % 60):02d}",
                            total=f"{int(duration // 60):02d}:{int(duration % 60):02d}",
                            speed=speed,
                            eta=f"{int(eta // 60)}:{int(eta % 60):02d}",
                        ),
                    )

        payload = _safe_json(recognizer.FinalResult())
        segment = _segment_from_payload(payload)
        if segment is not None:
            result.segments.append(segment)
            if on_segment is not None:
                on_segment(segment)

        result.processed_duration = duration
        result.elapsed_seconds = time.time() - started
        _add_punctuation(result.segments)
        return result


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _safe_json(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def _segment_from_payload(payload: Dict[str, Any]) -> Optional[Segment]:
    text = str(payload.get("text") or "").strip()
    words: List[Word] = []
    for item in payload.get("result") or []:
        if not isinstance(item, dict):
            continue
        try:
            words.append(
                Word(
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    text=str(item.get("word", "")),
                    probability=float(item.get("conf", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue
    if not text and not words:
        return None
    start = words[0].start if words else float(payload.get("start", 0.0) or 0.0)
    end = words[-1].end if words else float(payload.get("end", start) or start)
    return Segment(start=start, end=end, text=text, words=words)


def _hotwords_to_grammar(hotwords: str) -> str:
    """Vosk supports vocabulary restriction via a JSON grammar."""
    tokens = [t.strip().lower() for t in re.split(r"[,\n;]+", hotwords) if t.strip()]
    if not tokens:
        return ""
    words = sorted({word for token in tokens for word in token.split()})
    if not words:
        return ""
    payload = json.dumps(words + ["[unk]"], ensure_ascii=False)
    return payload


def _add_punctuation(segments: List[Segment]) -> None:
    """Vosk returns text without punctuation — make it minimally readable."""
    for index, segment in enumerate(segments):
        text = re.sub(r"\s{2,}", " ", segment.text).strip()
        if not text:
            continue
        text = text.capitalize()
        if text[-1] not in ".!?":
            # a long pause before the next segment = end of sentence
            next_start = segments[index + 1].start if index + 1 < len(segments) else None
            gap = (next_start - segment.end) if next_start is not None else 10.0
            text += "." if gap >= 1.2 else ","
        segment.text = text
