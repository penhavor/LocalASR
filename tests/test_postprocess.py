"""Тесты анти-галлюцинаций и сборки текста — то, что пользователь видит глазами."""

from __future__ import annotations

import pytest

from app.core.postprocess import (
    Segment,
    capitalize_start,
    clean_text,
    ensure_sentence_end,
    format_timestamp,
    is_junk_segment,
    join_plain,
    join_with_timestamps,
    sanitize_segments,
    split_for_subtitles,
    squeeze_repeats,
)


# --------------------------------------------------------------------------- #
#  Мусор, который реально выдаёт Whisper
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        "[музыка]",
        "(музыка)",
        "[аплодисменты]",
        "[ Music ]",
        "[Applause]",
        "[неразборчиво]",
        "[смех]",
        "♪♪♪",
        "[subtitles]",
        "<|ru|>",
    ],
)
def test_noise_is_removed(raw: str) -> None:
    assert clean_text(raw) == ""


def test_useful_text_survives_cleaning() -> None:
    text = "  Сегодня мы разберём градиентный спуск и его варианты.  "
    assert clean_text(text) == "Сегодня мы разберём градиентный спуск и его варианты."


def test_mixed_noise_and_speech() -> None:
    cleaned = clean_text("[музыка] Итак, начнём лекцию [аплодисменты]")
    assert cleaned == "Итак, начнём лекцию"


@pytest.mark.parametrize(
    "text",
    [
        "Спасибо за просмотр!",
        "спасибо за просмотр. ставь лайк",
        "Субтитры сделал DimaTorzok",
        "Thanks for watching",
        "...",
        "",
        "—",
    ],
)
def test_junk_segments_detected(text: str) -> None:
    assert is_junk_segment(text)


def test_normal_segment_is_not_junk() -> None:
    assert not is_junk_segment("Спасибо за просмотр мы начнём со следующей темы и разберём детали")
    assert not is_junk_segment("Матрица Якоби нужна для обратного распространения ошибки")


def test_squeeze_repeats() -> None:
    assert squeeze_repeats("это это это важно") == "это важно"
    assert squeeze_repeats("ну ну ну ну вот и всё") == "ну вот и всё"
    assert squeeze_repeats("слово слово") == "слово слово"  # два повтора — ещё норма


def test_sanitize_removes_duplicate_loop() -> None:
    segments = [
        Segment(start=0.0, end=3.0, text="Переходим к следующей части лекции"),
        Segment(start=3.0, end=6.0, text="Переходим к следующей части лекции"),
        Segment(start=6.0, end=9.0, text="переходим к следующей части лекции"),
        Segment(start=9.0, end=12.0, text="И рассмотрим примеры"),
    ]
    cleaned = sanitize_segments(segments)
    texts = [s.text for s in cleaned]
    assert texts.count("Переходим к следующей части лекции") == 1
    assert "И рассмотрим примеры" in texts


def test_sanitize_removes_noise_and_low_confidence() -> None:
    segments = [
        Segment(start=0.0, end=1.0, text="[музыка]"),
        Segment(start=1.0, end=2.0, text="Полезная фраза", avg_logprob=-0.4, no_speech_prob=0.01),
        Segment(start=2.0, end=3.0, text="ага", avg_logprob=-3.5, no_speech_prob=0.99),
        Segment(start=3.0, end=4.0, text="Спасибо за просмотр!"),
    ]
    cleaned = sanitize_segments(segments)
    assert [s.text for s in cleaned] == ["Полезная фраза"]


def test_sanitize_without_aggressive_mode_keeps_more() -> None:
    segments = [
        Segment(start=0.0, end=2.0, text="ага", avg_logprob=-3.5, no_speech_prob=0.99),
    ]
    assert sanitize_segments(segments, aggressive=False)


# --------------------------------------------------------------------------- #
#  Сборка текста
# --------------------------------------------------------------------------- #


def test_join_plain_paragraphs() -> None:
    segments = [
        Segment(start=0.0, end=2.0, text="первая фраза"),
        Segment(start=2.0, end=4.0, text="вторая фраза"),
        Segment(start=10.0, end=12.0, text="после долгой паузы"),
    ]
    text = join_plain(segments, paragraphs=True)
    assert text.count("\n\n") == 1
    assert text.startswith("Первая фраза")
    assert "После долгой паузы" in text


def test_join_plain_single_paragraph_when_no_gaps() -> None:
    segments = [
        Segment(start=0.0, end=2.0, text="раз"),
        Segment(start=2.0, end=4.0, text="два"),
    ]
    assert "\n\n" not in join_plain(segments, paragraphs=True)


def test_join_with_timestamps() -> None:
    segments = [Segment(start=65.0, end=70.0, text="текст")]
    assert join_with_timestamps(segments) == "[00:01:05] Текст\n"


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(3661.5) == "01:01:01"
    assert format_timestamp(1.234, milliseconds=True) == "00:00:01.234"
    assert format_timestamp(1.234, milliseconds=True, comma=True) == "00:00:01,234"


def test_ensure_sentence_end_and_capitalize() -> None:
    assert ensure_sentence_end("привет") == "привет."
    assert ensure_sentence_end("привет!") == "привет!"
    assert ensure_sentence_end("") == ""
    assert capitalize_start("привет") == "Привет"
    assert capitalize_start("— привет") == "— Привет"
    assert capitalize_start("") == ""


# --------------------------------------------------------------------------- #
#  Субтитры
# --------------------------------------------------------------------------- #


def test_split_for_subtitles_limits_length() -> None:
    long_text = " ".join(f"слово{i}" for i in range(60))
    segments = [Segment(start=0.0, end=30.0, text=long_text)]
    parts = split_for_subtitles(segments, max_chars=60, max_seconds=6.0)
    assert len(parts) > 1
    assert all(len(part.text) <= 80 for part in parts)


def test_split_keeps_short_segment() -> None:
    segments = [Segment(start=0.0, end=2.0, text="Коротко.")]
    assert len(split_for_subtitles(segments)) == 1
