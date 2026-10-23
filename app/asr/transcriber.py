"""Оркестрация: файл -> звук -> движок -> постобработка -> файл результата.

Здесь нет ни строчки Qt: модуль одинаково используется GUI-воркером и CLI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import paths
from ..constants import SAMPLE_RATE
from ..core import exporters
from ..core.ffmpeg import AudioError, decode_audio, probe
from ..core.models_catalog import ModelEntry
from ..core.postprocess import (
    Segment,
    build_header,
    join_plain,
    sanitize_segments,
)
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
    """Движок под задачу.

    При однопоточной обработке берём общий закэшированный движок из реестра
    (модель грузится один раз на всю очередь). При parallel_files > 1 каждый
    поток получает СВОЙ экземпляр — иначе потоки дрались бы за одну модель.
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


#: метки форматов, у которых одинаковое расширение (txt и txt-time)
FORMAT_LABELS = {
    "txt": "",
    "txt-time": "таймкоды",
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
    """Пишет результат сразу в несколько форматов.

    Правило именования (важно для пользователя):
      * один формат        -> «лекция 1.txt»  (имя = имени оригинала);
      * несколько форматов -> «лекция 1.txt», «лекция 1.srt», «лекция 1.vtt»;
      * коллизия расширений (txt и txt-time) -> «лекция 1 (таймкоды).txt»;
      * файл уже существует и перезапись выключена -> «лекция 1 (2).txt».
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
    """Полный цикл обработки одного файла.

    `audio` можно передать готовым (префетч из соседнего потока) — тогда шаг
    декодирования пропускается.
    """
    source = Path(source)
    result = FileResult(source=source)
    started = time.time()

    def report(fraction: float, stage: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, fraction)), stage)

    # ------------------------------------------------------------------ #
    # 1. Метаданные
    # ------------------------------------------------------------------ #
    report(0.01, f"Читаю {source.name}")
    info = probe(source)
    if info.audio_streams == 0 and info.error:
        result.error = info.error
        return result
    duration = audio_duration or info.duration or 0.0
    result.audio_duration = duration

    # ------------------------------------------------------------------ #
    # 2. Декодирование в 16 кГц моно float32
    # ------------------------------------------------------------------ #
    if audio is None:
        decode_started = time.time()
        report(0.03, "Извлекаю звук (ffmpeg)")

        def decode_progress(fraction: float) -> None:
            report(0.03 + 0.07 * fraction, "Извлекаю звук…")

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
        report(0.10, "Звук уже декодирован (prefetch)")

    # ------------------------------------------------------------------ #
    # 3. Распознавание
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
    # звук больше не нужен — освобождаем память до постобработки
    del audio

    result.language = transcript.language or (options.language or "")
    result.warnings.extend(transcript.warnings)
    if transcript.audio_duration:
        result.audio_duration = transcript.audio_duration

    # ------------------------------------------------------------------ #
    # 4. Постобработка (анти-галлюцинации)
    # ------------------------------------------------------------------ #
    report(0.96, "Чищу текст от галлюцинаций")
    segments = transcript.segments
    if options.suppress_hallucinations:
        cleaned = sanitize_segments(segments, aggressive=True)
        result.dropped = len(segments) - len(cleaned)
        segments = cleaned
    else:
        segments = [segment for segment in segments if segment.text.strip()]
        result.dropped = 0

    if not segments:
        result.error = (
            "Распознавание не дало ни одного сегмента. Возможные причины: в файле нет речи, "
            "слишком агрессивный VAD или неверно выбран язык."
        )
        result.elapsed_seconds = time.time() - started
        return result

    # ------------------------------------------------------------------ #
    # 5. Экспорт
    # ------------------------------------------------------------------ #
    report(0.98, "Сохраняю результат")
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
                "Движок": result.engine_label or transcript.engine,
                "Устройство": f"{transcript.device} / {transcript.compute_type}",
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
        "app_version": "1.0.0",
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

    result.outputs = written
    result.ok = True
    result.segments = len(segments)
    result.words = sum(len(segment.text.split()) for segment in segments)
    result.chars = sum(len(segment.text) for segment in segments)
    result.elapsed_seconds = time.time() - started
    result.preview = join_plain(segments[:12], paragraphs=False)[:400]
    report(1.0, f"Готово: {written[0].name if written else stem}")
    log.info(
        "%s -> %s | %d сегментов, %d слов, %.1f× за %.1f с",
        source.name,
        written[0].name if written else "—",
        result.segments,
        result.words,
        result.speed_x,
        result.elapsed_seconds,
    )
    return result
