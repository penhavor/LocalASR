"""Тесты экспорта: txt, srt, vtt, json, md."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core import exporters
from app.core.postprocess import Segment, Word


@pytest.fixture()
def segments() -> list[Segment]:
    return [
        Segment(
            start=0.0,
            end=2.5,
            text="Сегодня мы разберём градиентный спуск",
            words=[
                Word(0.0, 0.6, "Сегодня", 0.98),
                Word(0.6, 0.9, "мы", 0.99),
                Word(0.9, 1.6, "разберём", 0.95),
                Word(1.6, 2.5, "градиентный спуск", 0.9),
            ],
        ),
        Segment(start=4.0, end=6.0, text="Начнём с определения.", avg_logprob=-0.3),
        Segment(start=6.0, end=8.0, text="Потом посмотрим на примеры."),
    ]


def test_txt_plain(segments) -> None:
    text = exporters.render(segments, "txt")
    assert "Сегодня мы разберём градиентный спуск" in text
    assert "[" not in text, "в обычном txt не должно быть таймкодов"


def test_txt_with_time(segments) -> None:
    text = exporters.render(segments, "txt-time", meta={"header": "# Лекция\n"})
    assert text.startswith("# Лекция")
    assert "[00:00:00]" in text
    assert "[00:00:04]" in text


def test_srt_format(segments) -> None:
    text = exporters.render(segments, "srt")
    lines = text.splitlines()
    assert lines[0] == "1"
    assert "-->" in lines[1]
    assert "," in lines[1], "SRT использует запятую как десятичный разделитель"
    assert "Сегодня мы разберём градиентный спуск" in text


def test_vtt_format(segments) -> None:
    text = exporters.render(segments, "vtt")
    assert text.startswith("WEBVTT")
    assert "-->" in text
    assert "," not in text.split("-->")[1].split("\n")[0]


def test_json_format(segments) -> None:
    text = exporters.render(segments, "json", meta={"model": "test", "language": "ru"})
    data = json.loads(text)
    assert data["meta"]["model"] == "test"
    assert len(data["segments"]) == 3
    assert data["segments"][0]["words"][0]["word"] == "Сегодня"
    assert data["text"].startswith("Сегодня")


def test_md_format(segments) -> None:
    text = exporters.render(segments, "md", meta={"header": "# Лекция 1\n"})
    assert text.startswith("# Лекция 1")
    assert "градиентный" in text


def test_unknown_format_raises(segments) -> None:
    with pytest.raises(ValueError):
        exporters.render(segments, "pdf")


def test_export_writes_file(tmp_path: Path, segments) -> None:
    target = tmp_path / "лекция 1.txt"
    written = exporters.export(segments, target, "txt")
    assert written == target
    assert target.read_text(encoding="utf-8").strip()
    assert not list(tmp_path.glob("*.part"))


def test_export_all_formats(tmp_path: Path, segments) -> None:
    written = exporters.export_all_formats(
        segments, tmp_path, "лекция 1", ["txt", "srt", "vtt", "json"], overwrite=True
    )
    assert {p.suffix for p in written} == {".txt", ".srt", ".vtt", ".json"}
    for path in written:
        assert path.is_file() and path.stat().st_size > 0


def test_long_segment_is_split_for_subtitles() -> None:
    long_text = " ".join(["слово"] * 80)
    segments = [Segment(start=0.0, end=40.0, text=long_text)]
    srt = exporters.render(segments, "srt")
    assert srt.count("-->") >= 3, "длинный сегмент должен быть порезан на несколько субтитров"
