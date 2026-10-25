# -*- coding: utf-8 -*-
"""Chunking and post-processing invariant tests.

Covers audio chunking (gap-free tiling, length limits, pause boundaries,
short tails) and anti-hallucination cleanup — without loading real ASR
models.

Run:  cd <project root> && python tests/test_chunking.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from app.asr import torch_engines as te  # noqa: E402

SR = 16000


def make_audio(spec):
    """spec: list of (start_sec, end_sec, amplitude) -> float32 array."""
    total = int(max(e for _, e, _ in spec) * SR) + SR
    audio = np.zeros(total, dtype=np.float32)
    t = np.arange(total) / SR
    rng = np.random.default_rng(0)
    for start, end, amp in spec:
        mask = (t >= start) & (t < end)
        audio[mask] = (
            amp * 0.5 * np.sin(2 * np.pi * 220 * t[mask])
            + amp * 0.3 * np.sin(2 * np.pi * 640 * t[mask])
            + amp * 0.2 * rng.normal(0, 0.3, int(mask.sum()))
        ).astype(np.float32)
    return audio


def speech_spec(total_seconds: float, phrase: float = 3.0, pause: float = 0.8):
    spec = []
    t = 0.5
    while t + phrase < total_seconds:
        spec.append((t, t + phrase, 0.4))
        t += phrase + pause
    return spec


def check_tiling(bounds, total, label):
    problems = []
    if not bounds:
        return ["no chunks"]
    if bounds[0][0] != 0:
        problems.append(f"first chunk does not start at 0: {bounds[0][0]}")
    if bounds[-1][1] != total:
        problems.append(f"last chunk does not reach the end: {bounds[-1][1]} != {total}")
    for (s1, e1), (s2, e2) in zip(bounds, bounds[1:]):
        if e1 != s2:
            problems.append(f"gap/overlap: {e1} != {s2}")
        if e1 <= s1:
            problems.append(f"non-positive chunk length: {s1}..{e1}")
    lengths = [round((e - s) / SR, 2) for s, e in bounds]
    print(f"  [{label}] chunks: {len(bounds)}, lengths(s): {lengths[:12]}"
          f"{'...' if len(bounds) > 12 else ''}")
    return problems


def cuts_inside_speech(bounds, spec):
    """Did a chunk boundary (except first/last) land inside a "phrase"?"""
    bad = []
    for s, e in bounds[1:]:
        t = e / SR
        for start, end, _ in spec:
            if start + 0.05 < t < end - 0.05:
                bad.append((round(t, 2), start, end))
    return bad


def test_energy_chunker():
    print("== iter_silence_chunks (energy-based) ==")
    spec = speech_spec(90)
    audio = make_audio(spec)
    total = len(audio)
    bounds = list(te.iter_silence_chunks(audio, SR, 27.0, search_seconds=1.5))
    problems = check_tiling(bounds, total, "energy")
    maxlen = max((e - s) / SR for s, e in bounds)
    assert maxlen <= 27.0 + 1.5 + 1e-6, f"chunk longer than the limit: {maxlen}"
    assert not problems, problems


def test_smart_chunker_with_fake_vad():
    print("== iter_smart_chunks (VAD pauses: known points) ==")
    spec = speech_spec(90)
    audio = make_audio(spec)
    total = len(audio)
    points = [int(((a[1] + b[0]) / 2) * SR) for a, b in zip(spec, spec[1:])]
    orig = te._vad_pause_points
    te._vad_pause_points = lambda *a, **k: points
    try:
        bounds = list(te.iter_smart_chunks(audio, SR, 27.0, search_seconds=1.5, use_vad=True))
    finally:
        te._vad_pause_points = orig
    problems = check_tiling(bounds, total, "smart")
    cuts = cuts_inside_speech(bounds, spec)
    print(f"  cuts inside phrases: {len(cuts)} {cuts[:5]}")
    assert not problems, problems
    assert not cuts, f"smart chunker cuts phrases in half: {cuts}"


def _loud_quiet_audio(quiet_from: float, quiet_to: float, total: float, seed: int = 1):
    """Loud noise with a quiet insert [quiet_from, quiet_to] (energy minimum)."""
    rng = np.random.default_rng(seed)
    audio = rng.normal(0, 0.3, int(total * SR)).astype(np.float32)
    a, b = int(quiet_from * SR), int(quiet_to * SR)
    audio[a:b] *= 0.02
    return audio


def test_tiny_tail_merged():
    print("== short tail (<1 s) is merged into the previous chunk ==")
    # boundary found in the quiet insert ~26.2-26.6; tail 26.4..27.0 = ~0.6 s
    audio = _loud_quiet_audio(26.2, 26.6, 27.0)
    bounds = list(te.iter_silence_chunks(audio, SR, 27.0, search_seconds=1.5))
    lengths = [(e - s) / SR for s, e in bounds]
    print(f"  chunks: {len(bounds)}, lengths: {[round(x, 2) for x in lengths]}")
    assert len(bounds) == 1, f"tail not merged: {lengths}"
    assert bounds[-1][1] == len(audio), "audio tail lost"


def test_tiny_tail_kept_when_merge_exceeds_limit():
    print("== short tail: merge forbidden by the model limit ==")
    # quiet insert at 28.3-28.5, end 28.9: prev(~28.4)+tail(0.5) = 28.9
    # <= 27+1.5+1 = 29.5, so here the tail IS merged; it stays separate
    # only beyond the limit.
    audio = _loud_quiet_audio(28.3, 28.5, 28.9, seed=3)
    bounds = list(te.iter_silence_chunks(audio, SR, 27.0, search_seconds=1.5))
    lengths = [(e - s) / SR for s, e in bounds]
    print(f"  lengths: {[round(x, 2) for x in lengths]}")
    for s, e in bounds:
        assert (e - s) / SR <= 27.0 + 1.5 + te.MIN_CHUNK_SECONDS + 1e-6, "chunk limit exceeded"
    assert bounds[-1][1] == len(audio)
    # direct test of the merge helper: beyond the limit the tail stays
    merged = te._merge_short_tail([(0, int(29.4 * SR)), (int(29.4 * SR), int(29.9 * SR))],
                                  SR, 27.0, 1.5)
    assert len(merged) == 2, f"the merge should have been forbidden by the limit: {merged}"


def test_real_vad_available():
    print("== real Silero VAD (faster-whisper) — does not crash ==")
    audio = make_audio([(0.5, 3.5, 0.4), (4.3, 7.3, 0.4), (8.1, 11.0, 0.4)])
    pts = te._vad_pause_points(audio, SR, 300)
    if pts is None:
        print("  VAD unavailable (no faster-whisper) — energy fallback")
    else:
        print(f"  pause points: {[round(p / SR, 2) for p in pts]}")


def test_continuous_speech_no_pauses():
    print("== continuous speech without pauses (smart chunker -> energy fallback) ==")
    audio = np.random.default_rng(2).normal(0, 0.25, int(80 * SR)).astype(np.float32)
    orig = te._vad_pause_points
    te._vad_pause_points = lambda *a, **k: []
    try:
        bounds = list(te.iter_smart_chunks(audio, SR, 27.0, search_seconds=1.5, use_vad=True))
    finally:
        te._vad_pause_points = orig
    problems = check_tiling(bounds, len(audio), "no-pauses")
    assert not problems, problems


def test_postprocess():
    print("== post-processing (anti-hallucinations) ==")
    from app.core.postprocess import Segment, clean_text, sanitize_segments, squeeze_repeats

    assert clean_text("Привет <|ru|> [музыка] мир ** ").strip() == "Привет мир"
    assert squeeze_repeats("это это это важно") == "это важно"
    segs = [
        Segment(0, 3, "Первое предложение про лекцию."),
        Segment(3, 6, "Первое предложение про лекцию."),
        Segment(6, 9, "Второе предложение совсем другое."),
        Segment(9, 12, "Спасибо за просмотр!"),
        Segment(12, 15, "[аплодисменты]"),
    ]
    out = sanitize_segments([Segment(s.start, s.end, s.text) for s in segs])
    texts = [s.text for s in out]
    print(f"  left: {texts}")
    assert "Второе предложение совсем другое." in texts
    assert len([t for t in texts if "Первое предложение" in t]) == 1
    assert not any("Спасибо за просмотр" in t for t in texts)


def test_decoder_loop_cases():
    print("== typical decoder loops are collapsed ==")
    from app.core.postprocess import squeeze_repeats

    # 1) adjacent exact duplicate run
    t = ("делаем волшебный раствор добавляем усилитель Делаем волшебный раствор "
         "добавляем усилитель и это все растворяется в горячей воде")
    out = squeeze_repeats(t)
    assert out.lower().count("делаем волшебный раствор") == 1, out
    assert "и это все растворяется в горячей воде" in out

    # 2) adjacent duplicate with a degraded token (паш/пш)
    t = ("наносим кислотный ополаскиватель чтобы привести уровень паш в норму "
         "наносим кислотный ополаскиватель чтобы привести уровень пш в норму "
         "если этого не сделать то оставшаяся щелочь")
    out = squeeze_repeats(t)
    assert out.lower().count("наносим кислотный ополаскиватель") == 1, out
    assert "если этого не сделать" in out

    # 3) adjacent duplicate at the segment start
    t = ("второй раз помещаем его в скафандр теперь набираем второй раз помещаем "
         "его в скафандр теперь набираем кислотный раствор в этот скафандр")
    out = squeeze_repeats(t)
    assert out.lower().count("второй раз помещаем") == 1, out
    assert "кислотный раствор в этот скафандр" in out

    # 4) exact repeated window not at the tail
    t = ("теперь вот так теперь этот коврик выглядит в макро Вот так теперь этот "
         "коврик выглядит в макро у каждой кисоньки был свой уголок")
    out = squeeze_repeats(t)
    assert out.lower().count("этот коврик выглядит в макро") == 1, out
    assert "у каждой кисоньки" in out

    # 5) short echo in the tail
    t = ("финальное взвешивание и весит он теперь шесть килограмм сто грамм это "
         "значит он похудел на полтора килограмма финальное взвешивание")
    out = squeeze_repeats(t)
    assert out.lower().count("финальное взвешивание") == 1, out

    # NEGATIVE: legitimate text must not be mangled
    legit = ("стирка обычного ковра стоит три доллара за квадратный метр а стирка ковра "
             "в моче неважно чей кошачий собачий человечий стоит шесть долларов за "
             "квадратный метр потому что нужно выводить эту вонь из ковра")
    assert squeeze_repeats(legit) == legit, squeeze_repeats(legit)
    legit2 = "доброе утро коллеги сегодня мы поговорим о теории информации и кодировании"
    assert squeeze_repeats(legit2) == legit2
    print("  all loops collapsed, legitimate text intact")


def test_repr_tuple_unwrap():
    print("== decoder repr-tuples do not leak into the text ==")
    from app.asr.torch_engines import decode_item_text
    from app.core.postprocess import clean_text

    # recent GigaAM: (text, token_ids, token_frames)
    assert decode_item_text(("привет мир", [45, 33], [0, 1])) == "привет мир"
    # older versions: a bare string
    assert decode_item_text("привет мир") == "привет мир"
    assert decode_item_text(None) == ""

    garbage = "('Сегодня все правильно меня зовут владимир', [45, 33, 31, 42, 0, 59])"
    cleaned = clean_text(garbage)
    assert cleaned == "Сегодня все правильно меня зовут владимир", cleaned
    # regular text is untouched
    normal = "(привет) это обычная фраза со скобками [и квадратами]"
    assert clean_text(normal) == normal.strip(" \\t\\r\\n·—-") or clean_text(normal).startswith("(привет)")
    assert "[45, 33]" not in clean_text(garbage)


def test_cross_segment_echo():
    print("== loops torn at segment boundaries ==")
    from app.core.postprocess import Segment, sanitize_segments

    segs = [
        Segment(0, 20, "вес ковра семь килограмм делаем волшебный раствор добавляем усилитель"),
        Segment(20, 40, "Делаем волшебный раствор добавляем усилитель и это все растворяется в воде"),
        Segment(40, 50, "финальное взвешивание и весит он теперь шесть килограмм сто грамм"),
        Segment(50, 51, "финальное взвешивание"),
        Segment(51, 60, "Да."),
        Segment(60, 70, "а стирка ковра в моче стоит шесть долларов за метр"),
    ]
    out = sanitize_segments([Segment(s.start, s.end, s.text) for s in segs])
    texts = [s.text for s in out]
    print("  result:", texts)
    # the head of the second segment repeating the tail of the first is trimmed
    assert texts[0].endswith("добавляем усилитель"), texts
    assert texts[1] == "и это все растворяется в воде", texts
    # the echo stub is dropped
    assert not any(t.strip().lower() == "финальное взвешивание" for t in texts), texts
    # a legitimate short reply and a legitimate continuation are NOT touched
    assert "Да." in texts, texts
    assert any(t.startswith("а стирка ковра в моче") for t in texts), texts


def test_legit_text_untouched():
    print("== legitimate repeats and replies are NOT cut ==")
    from app.core.postprocess import Segment, sanitize_segments, squeeze_repeats

    # emphasis, parallelism, enumerations, same-root repeats
    legit = [
        "это очень очень важная точка и совершенно совершенно неприемлемый подход",
        "первый раздел второй раздел третий раздел вот и всё оглавление",
        "и коты и коты и собаки жили в этой квартире по очереди",
        "стирка обычного ковра стоит три доллара за квадратный метр а стирка ковра "
        "в моче неважно чей кошачий собачий человечий стоит шесть долларов за "
        "квадратный метр потому что нужно выводить эту вонь из ковра",
        "вода в реке была холодна вода в озере была теплее вода в море оказалась самой тёплой",
    ]
    for line in legit:
        out = squeeze_repeats(line)
        assert out == line, f"legitimate text changed: {line!r} -> {out!r}"

    # dialogue replies and a catchphrase far apart (different segments)
    segs = [
        Segment(0, 2, "Да."),
        Segment(2, 4, "Ну что."),
        Segment(4, 20, "доброе утро коллеги и друзья такова реальность нашей стирки"),
        Segment(20, 40, "продолжаем работу и помним друзья такова реальность"),
        Segment(40, 42, "Да."),
    ]
    out = sanitize_segments([Segment(s.start, s.end, s.text) for s in segs])
    texts = [s.text for s in out]
    assert texts == [s.text for s in segs], texts
    print("  emphasis, enumerations, \"Да.\" replies and a catchphrase in different segments intact")


if __name__ == "__main__":
    test_energy_chunker()
    test_smart_chunker_with_fake_vad()
    test_tiny_tail_merged()
    test_tiny_tail_kept_when_merge_exceeds_limit()
    test_real_vad_available()
    test_continuous_speech_no_pauses()
    test_postprocess()
    test_decoder_loop_cases()
    test_cross_segment_echo()
    test_legit_text_untouched()
    test_repr_tuple_unwrap()
    print("\nALL TESTS PASSED")
