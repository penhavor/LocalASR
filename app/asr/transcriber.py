"""Orchestration: file -> audio -> engine -> post-processing -> result file.

No Qt here: the module is used by both the GUI worker and the CLI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import paths
from ..constants import APP_VERSION, SAMPLE_RATE
from ..core import exporters
from ..core.ffmpeg import AudioError, decode_audio, probe
from ..core.models_catalog import ModelEntry
from ..core.postprocess import (
    Segment,
    build_header,
    join_plain,
    sanitize_segments,
)
from ..i18n import _
from ..logutils import get_logger
from ..settings import Settings
from .base import CancelToken, ProgressCallback, TranscriptResult
from .registry import registry

log = get_logger("pipeline")


@dataclass
class JobOptions:
    model: ModelEntry
    settings: Settings
    output_dir: Path
    formats: List[str] = field(default_factory=lambda: ["txt"])
    language: str = "ru"
    translate: bool = False
    beam_size: int = 5
    use_vad: bool = True
    word_timestamps: bool = False
    suppress_hallucinations: bool = True
    paragraphs: bool = True
    initial_prompt: str = ""
    hotwords: str = ""
    batched: bool = True
    batch_size: int = 8
    device: str = "auto"
    compute_type: str = "auto"
    cpu_threads: int = 0
    gpu_index: int = 0
    overwrite: bool = False
    write_header: bool = False
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, model: ModelEntry, settings: Settings, output_dir: Optional[Path] = None) -> "JobOptions":
        merged = dict(model.defaults or {})
        merged.update(
            {
                "language": settings.language,
                "translate": settings.translate_to_english,
                "beam_size": settings.beam_size,
                "use_vad": settings.use_vad,
                "word_timestamps": settings.word_timestamps,
                "suppress_hallucinations": settings.suppress_hallucinations,
                "paragraphs": settings.paragraphs,
                "initial_prompt": settings.initial_prompt,
                "hotwords": settings.hotwords,
                "batched": settings.batched,
                "batch_size": settings.batch_size,
                "device": settings.device,
                "compute_type": settings.compute_type,
                "cpu_threads": settings.cpu_threads,
                "gpu_index": settings.gpu_index,
                "overwrite": settings.overwrite_existing,
            }
        )
        return cls(
            model=model,
            settings=settings,
            output_dir=Path(output_dir) if output_dir else settings.resolved_output_dir(),
            formats=[settings.output_format],
            **{
                key: value
                for key, value in merged.items()
                if key in cls.__dataclass_fields__ and key not in ("model", "settings", "output_dir", "formats")
            },
        )


@dataclass
class FileResult:
    source: Path
    outputs: List[Path] = field(default_factory=list)
    ok: bool = False
    error: str = ""
    segments: int = 0
    words: int = 0
    chars: int = 0
    audio_duration: float = 0.0
    elapsed_seconds: float = 0.0
    decode_seconds: float = 0.0
    language: str = ""
    warnings: List[str] = field(default_factory=list)
    dropped: int = 0
    preview: str = ""
    engine_label: str = ""

    @property
    def speed_x(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.audio_duration / self.elapsed_seconds


def get_engine_for(options: JobOptions, engine=None):
    """Engine for the job.

    Sequential processing uses the shared cached registry engine (the model
    is loaded once per queue). With parallel_files > 1 every thread gets its
    OWN instance — otherwise threads would fight over one model.
    """
    if engine is not None:
        return engine
    return registry.get(
        options.model,
        {
            "device": options.device,
            "compute_type": options.compute_type,
            "cpu_threads": options.cpu_threads,
            "gpu_index": options.gpu_index,
        },
    )


#: labels for formats sharing one extension (txt and txt-time)
FORMAT_LABELS = {
    "txt": "",
    "txt-time": _("timecodes"),
    "srt": "srt",
    "vtt": "vtt",
    "json": "json",
    "md": "md",
}


def export_all(
    segments,
    *,
    formats: List[str],
    directory: Path,
    stem: str,
    meta: Optional[Dict[str, Any]] = None,
    paragraphs: bool = True,
    overwrite: bool = False,
) -> List[Path]:
    """Write the result to several formats at once.

    Naming: one format -> "lecture 1.txt"; several -> per-format suffixes;
    an extension collision (txt/txt-time) -> "lecture 1 (timecodes).txt";
    an existing file with overwrite off -> "lecture 1 (2).txt".
    """
    return exporters.export_all_formats(
        segments,
        directory=directory,
        stem=stem,
        formats=formats,
        meta=meta,
        paragraphs=paragraphs,
        overwrite=overwrite,
        labels=FORMAT_LABELS,
    )


def transcribe_file(
    source: Path | str,
    options: JobOptions,
    *,
    progress: Optional[ProgressCallback] = None,
    on_segment: Optional[Callable[[Segment], None]] = None,
    cancel: Optional[CancelToken] = None,
    audio: Any = None,
    audio_duration: float = 0.0,
    engine: Any = None,
) -> FileResult:
    """Full processing cycle for one file.

    `audio` may be passed pre-decoded (prefetch from another thread) — then
    the decoding step is skipped.
    """
    source = Path(source)
    result = FileResult(source=source)
    started = time.time()

    def report(fraction: float, stage: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, fraction)), stage)

    # ------------------------------------------------------------------ #
    # 1. Metadata
    # ------------------------------------------------------------------ #
    report(0.01, _("Reading {name}").format(name=source.name))
    info = probe(source)
    log.debug(
        "probe(%s): duration=%.1f s, audio streams=%d, video streams=%d, "
        "codec=%s, sample_rate=%d, channels=%d, error=%r",
        source.name, info.duration, info.audio_streams, info.video_streams,
        info.codec, info.sample_rate, info.channels, info.error,
    )
    if info.audio_streams == 0 and info.error:
        result.error = info.error
        return result
    duration = audio_duration or info.duration or 0.0
    result.audio_duration = duration

    # ------------------------------------------------------------------ #
    # 2. Decode to 16 kHz mono float32
    # ------------------------------------------------------------------ #
    if audio is None:
        decode_started = time.time()
        report(0.03, _("Extracting audio (ffmpeg)"))

        def decode_progress(fraction: float) -> None:
            report(0.03 + 0.07 * fraction, _("Extracting audio…"))

        try:
            audio = decode_audio(source, SAMPLE_RATE, cancel=cancel, progress=decode_progress)
        except AudioError as exc:
            result.error = str(exc)
            return result
        result.decode_seconds = time.time() - decode_started
        if duration <= 0:
            duration = len(audio) / SAMPLE_RATE
            result.audio_duration = duration
    else:
        report(0.10, _("Audio already decoded (prefetch)"))

    # ------------------------------------------------------------------ #
    # 3. Recognition
    # ------------------------------------------------------------------ #
    engine = get_engine_for(options, engine)
    result.engine_label = f"{engine.name} · {engine.device} · {engine.compute_type}"

    def asr_progress(fraction: float, stage: str) -> None:
        report(0.10 + 0.85 * fraction, stage)

    engine_options: Dict[str, Any] = {
        "language": options.language or None,
        "translate": options.translate,
        "beam_size": options.beam_size,
        "vad": options.use_vad,
        "word_timestamps": options.word_timestamps,
        "initial_prompt": options.initial_prompt,
        "hotwords": options.hotwords,
        "batched": options.batched,
        "batch_size": options.batch_size,
        "model_name": options.model.name,
    }

    transcript: TranscriptResult = engine.transcribe(
        audio,
        options=engine_options,
        duration=duration,
        progress=asr_progress,
        on_segment=on_segment,
        cancel=cancel,
    )
    # audio is no longer needed — free memory before post-processing
    del audio

    log.debug(
        "ASR(%s): segments=%d, language=%s, warnings=%s, %.1f s",
        source.name, len(transcript.segments), transcript.language or "—",
        transcript.warnings or "—", transcript.elapsed_seconds,
    )

    result.language = transcript.language or (options.language or "")
    result.warnings.extend(transcript.warnings)
    if transcript.audio_duration:
        result.audio_duration = transcript.audio_duration

    # ------------------------------------------------------------------ #
    # 4. Post-processing (anti-hallucinations)
    # ------------------------------------------------------------------ #
    report(0.96, _("Cleaning hallucinations from the text"))
    segments = transcript.segments
    if options.suppress_hallucinations:
        cleaned = sanitize_segments(segments, aggressive=True)
        result.dropped = len(segments) - len(cleaned)
        log.debug(
            "Post-processing(%s): segments %d -> %d (dropped %d)",
            source.name, len(segments), len(cleaned), result.dropped,
        )
        segments = cleaned
    else:
        segments = [segment for segment in segments if segment.text.strip()]
        result.dropped = 0

    if not segments:
        result.error = _(
            "Recognition produced no segments. Possible reasons: no speech in the file, "
            "overly aggressive VAD or a wrong language selection."
        )
        result.elapsed_seconds = time.time() - started
        return result

    # ------------------------------------------------------------------ #
    # 5. Export
    # ------------------------------------------------------------------ #
    report(0.98, _("Saving the result"))
    stem = paths.sanitize_stem(source.name)
    header = ""
    if options.write_header:
        header = build_header(
            source_name=source.name,
            model_name=options.model.name,
            language=result.language,
            duration_seconds=result.audio_duration,
            processing_seconds=max(time.time() - started, 0.001),
            extra={
                _("Engine"): result.engine_label or transcript.engine,
                _("Device"): f"{transcript.device} / {transcript.compute_type}",
                **options.extra_meta,
            },
        )

    meta = {
        "source_file": str(source),
        "source_name": source.name,
        "model": options.model.name,
        "model_id": options.model.id,
        "engine": transcript.engine,
        "device": transcript.device,
        "compute_type": transcript.compute_type,
        "language": result.language,
        "language_probability": round(transcript.language_probability, 4) or None,
        "audio_duration": round(result.audio_duration, 2),
        "sample_rate": SAMPLE_RATE,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "app_version": APP_VERSION,
        "header": header,
        "warnings": result.warnings or None,
    }

    options.output_dir.mkdir(parents=True, exist_ok=True)
    written = export_all(
        segments,
        formats=options.formats or ["txt"],
        directory=options.output_dir,
        stem=stem,
        meta=meta,
        paragraphs=options.paragraphs,
        overwrite=options.overwrite,
    )
    log.debug("Export(%s): %s", source.name, ", ".join(str(p) for p in written))

    result.outputs = written
    result.ok = True
    result.segments = len(segments)
    result.words = sum(len(segment.text.split()) for segment in segments)
    result.chars = sum(len(segment.text) for segment in segments)
    result.elapsed_seconds = time.time() - started
    result.preview = join_plain(segments[:12], paragraphs=False)[:400]
    report(1.0, _("Done: {name}").format(name=written[0].name if written else stem))
    log.info(
        _("%s -> %s | %d segments, %d words, %.1f× in %.1f s"),
        source.name,
        written[0].name if written else "—",
        result.segments,
        result.words,
        result.speed_x,
        result.elapsed_seconds,
    )
    return result
