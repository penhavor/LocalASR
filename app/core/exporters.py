"""Result export: .txt, .txt with timecodes, .srt, .vtt, .json, .md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .. import paths
from ..i18n import _
from .postprocess import (
    Segment,
    format_timestamp,
    join_plain,
    join_with_timestamps,
    split_for_subtitles,
)


def _meta_dict(meta: Dict) -> Dict:
    return {key: value for key, value in meta.items() if value is not None}


def render(
    segments: Sequence[Segment],
    fmt: str,
    meta: Optional[Dict] = None,
    paragraphs: bool = True,
) -> str:
    """Render the file content for the requested format."""
    meta = meta or {}
    segments = list(segments)

    if fmt == "txt":
        return join_plain(segments, paragraphs=paragraphs)

    if fmt == "txt-time":
        header = ""
        if meta.get("header"):
            header = meta["header"] + "\n"
        return header + join_with_timestamps(segments)

    if fmt == "srt":
        return render_srt(segments)

    if fmt == "vtt":
        return render_vtt(segments)

    if fmt == "md":
        header = meta.get("header") or ""
        body = join_plain(segments, paragraphs=paragraphs)
        return (header + "\n" + body) if header else body

    if fmt == "json":
        return render_json(segments, meta)

    raise ValueError(_("Unknown export format: {}").format(fmt))


def render_srt(segments: Sequence[Segment], max_chars: int = 84, max_seconds: float = 6.0) -> str:
    blocks: List[str] = []
    for index, segment in enumerate(split_for_subtitles(segments, max_chars, max_seconds), start=1):
        start = format_timestamp(segment.start, milliseconds=True, comma=True)
        end = format_timestamp(max(segment.end, segment.start + 0.2), milliseconds=True, comma=True)
        blocks.append(f"{index}\n{start} --> {end}\n{segment.text.strip()}\n")
    return "\n".join(blocks)


def render_vtt(segments: Sequence[Segment], max_chars: int = 84, max_seconds: float = 6.0) -> str:
    blocks: List[str] = ["WEBVTT\n"]
    for segment in split_for_subtitles(segments, max_chars, max_seconds):
        start = format_timestamp(segment.start, milliseconds=True)
        end = format_timestamp(max(segment.end, segment.start + 0.2), milliseconds=True)
        blocks.append(f"{start} --> {end}\n{segment.text.strip()}\n")
    return "\n".join(blocks)


def render_json(segments: Sequence[Segment], meta: Optional[Dict] = None) -> str:
    payload = {
        "meta": _meta_dict(meta or {}),
        "segments": [
            {
                "id": index,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "language": segment.language or None,
                "avg_logprob": round(segment.avg_logprob, 4) if segment.avg_logprob else None,
                "no_speech_prob": round(segment.no_speech_prob, 4) if segment.no_speech_prob else None,
                "words": (
                    [
                        {
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                            "word": word.text,
                            "probability": round(word.probability, 4) if word.probability else None,
                        }
                        for word in segment.words
                    ]
                    if segment.words
                    else None
                ),
            }
            for index, segment in enumerate(segments, start=1)
        ],
        "text": join_plain(segments),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export(
    segments: Sequence[Segment],
    out_path: Path | str,
    fmt: str,
    meta: Optional[Dict] = None,
    paragraphs: bool = True,
) -> Path:
    """Write the result to a file (atomic, UTF-8). Returns the path."""
    out_path = Path(out_path)
    text = render(segments, fmt, meta=meta, paragraphs=paragraphs)
    paths.atomic_write_text(out_path, text, encoding="utf-8")
    return out_path


def export_all_formats(
    segments: Sequence[Segment],
    directory: Path | str,
    stem: str,
    formats: Sequence[str],
    meta: Optional[Dict] = None,
    paragraphs: bool = True,
    overwrite: bool = False,
    labels: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Export to several formats at once.

    Naming rule: "<original name> <YYYY-MM-DD HH-MM-SS>.<ext>" — the stamp
    makes every run unique; formats of one run share one stamp; formats that
    collide by extension (txt/txt-time) get a `labels` tag; a rare same-second
    collision gets a "(2)" counter; overwrite=True writes the plain name.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels = labels or {}
    unique_formats = list(dict.fromkeys(formats))
    written: List[Path] = []
    taken: set[str] = set()
    stamp = paths.datetime_stamp()

    for fmt in unique_formats:
        suffix = paths.output_suffix(fmt)
        same_suffix = [f for f in unique_formats if paths.output_suffix(f) == suffix]
        stem_variant = stem
        if len(same_suffix) > 1 and labels.get(fmt):
            stem_variant = f"{stem} ({labels[fmt]})"

        if overwrite:
            name = f"{stem_variant}{suffix}"
        else:
            name = f"{stem_variant} {stamp}{suffix}"
            counter = 2
            while name.lower() in taken or (directory / name).exists():
                name = f"{stem_variant} {stamp} ({counter}){suffix}"
                counter += 1
                if counter > 999:
                    break

        target = directory / name
        taken.add(name.lower())
        export(segments, target, fmt, meta=meta, paragraphs=paragraphs)
        written.append(target)
    return written
