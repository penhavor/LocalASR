# -*- coding: utf-8 -*-
"""Тесты чанкинга и постобработки LocalASR.

Запуск:  cd <корень проекта> && python tests/test_chunking.py
Проверяют ИНВАРИАНТЫ нарезки аудио (покрытие без дыр, лимиты длины,
границы по паузам, короткие хвосты) и работу анти-галлюцинаций —
без загрузки реальных ASR-моделей.
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
    """spec: список (start_sec, end_sec, amplitude) -> массив float32."""
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
        return ["нет кусков"]
    if bounds[0][0] != 0:
        problems.append(f"первый кусок начинается не с 0: {bounds[0][0]}")
    if bounds[-1][1] != total:
        problems.append(f"последний кусок не доходит до конца: {bounds[-1][1]} != {total}")
    for (s1, e1), (s2, e2) in zip(bounds, bounds[1:]):
        if e1 != s2:
            problems.append(f"дыра/наложение: {e1} != {s2}")
        if e1 <= s1:
            problems.append(f"неположительная длина куска: {s1}..{e1}")
    lengths = [round((e - s) / SR, 2) for s, e in bounds]
    print(f"  [{label}] кусков: {len(bounds)}, длины(с): {lengths[:12]}"
          f"{'...' if len(bounds) > 12 else ''}")
    return problems


def cuts_inside_speech(bounds, spec):
    """Граница куска (кроме первой/последней) попала внутрь «фразы»?"""
    bad = []
    for s, e in bounds[1:]:
        t = e / SR
        for start, end, _ in spec:
            if start + 0.05 < t < end - 0.05:
                bad.append((round(t, 2), start, end))
    return bad


def test_energy_chunker():
    print("== iter_silence_chunks (энергетический) ==")
    spec = speech_spec(90)
    audio = make_audio(spec)
    total = len(audio)
    bounds = list(te.iter_silence_chunks(audio, SR, 27.0, search_seconds=1.5))
    problems = check_tiling(bounds, total, "energy")
    maxlen = max((e - s) / SR for s, e in bounds)
    assert maxlen <= 27.0 + 1.5 + 1e-6, f"кусок длиннее лимита: {maxlen}"
    assert not problems, problems


def test_smart_chunker_with_fake_vad():
    print("== iter_smart_chunks (VAD-паузы: известные точки) ==")
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
    print(f"  резов внутри фраз: {len(cuts)} {cuts[:5]}")
    assert not problems, problems
    assert not cuts, f"умный чанкер режет фразы посередине: {cuts}"


def _loud_quiet_audio(quiet_from: float, quiet_to: float, total: float, seed: int = 1):
    """Шум: громкий, с тихой вставкой [quiet_from, quiet_to] (минимум энергии)."""
    rng = np.random.default_rng(seed)
    audio = rng.normal(0, 0.3, int(total * SR)).astype(np.float32)
    a, b = int(quiet_from * SR), int(quiet_to * SR)
    audio[a:b] *= 0.02
    return audio


def test_tiny_tail_merged():
    print("== короткий хвост (<1 с) сливается с предыдущим куском ==")
    # граница найдётся в тихой вставке ~26.2-26.6; хвост 26.4..27.0 = ~0.6 с
    audio = _loud_quiet_audio(26.2, 26.6, 27.0)
    bounds = list(te.iter_silence_chunks(audio, SR, 27.0, search_seconds=1.5))
    lengths = [(e - s) / SR for s, e in bounds]
    print(f"  кусков: {len(bounds)}, длины: {[round(x, 2) for x in lengths]}")
    assert len(bounds) == 1, f"хвост не слит: {lengths}"
    assert bounds[-1][1] == len(audio), "хвост аудио потерян"


def test_tiny_tail_kept_when_merge_exceeds_limit():
    print("== короткий хвост: слияние запрещено лимитом модели ==")
    # тихая вставка у 28.3-28.5, конец 28.9: слияние дало бы >29.5? нет —
    # проверяем честный лимит: prev(~28.4)+tail(0.5) = 28.9 <= 27+1.5+1 = 29.5
    # поэтому здесь хвост СЛИВАЕТСЯ; отдельным он остаётся только за лимитом.
    audio = _loud_quiet_audio(28.3, 28.5, 28.9, seed=3)
    bounds = list(te.iter_silence_chunks(audio, SR, 27.0, search_seconds=1.5))
    lengths = [(e - s) / SR for s, e in bounds]
    print(f"  длины: {[round(x, 2) for x in lengths]}")
    for s, e in bounds:
        assert (e - s) / SR <= 27.0 + 1.5 + te.MIN_CHUNK_SECONDS + 1e-6, "превышен лимит куска"
    assert bounds[-1][1] == len(audio)
    # прямой тест помощника слияния: за пределом лимита хвост остаётся
    merged = te._merge_short_tail([(0, int(29.4 * SR)), (int(29.4 * SR), int(29.9 * SR))],
                                  SR, 27.0, 1.5)
    assert len(merged) == 2, f"слияние должно было быть запрещено лимитом: {merged}"


def test_real_vad_available():
    print("== реальный Silero VAD (faster-whisper) — не падает ==")
    audio = make_audio([(0.5, 3.5, 0.4), (4.3, 7.3, 0.4), (8.1, 11.0, 0.4)])
    pts = te._vad_pause_points(audio, SR, 300)
    if pts is None:
        print("  VAD недоступен (нет faster-whisper) — энергетический фолбэк")
    else:
        print(f"  точки пауз: {[round(p / SR, 2) for p in pts]}")


def test_continuous_speech_no_pauses():
    print("== непрерывная речь без пауз (умный чанкер -> энерг. фолбэк) ==")
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
    print("== постобработка (анти-галлюцинации) ==")
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
    print(f"  осталось: {texts}")
    assert "Второе предложение совсем другое." in texts
    assert len([t for t in texts if "Первое предложение" in t]) == 1
    assert not any("Спасибо за просмотр" in t for t in texts)


def test_user_repeat_cases():
    print("== реальные петли из отчёта пользователя схлопываются ==")
    from app.core.postprocess import squeeze_repeats

    # 1) соседний точный дубль-прогон
    t = ("делаем волшебный раствор добавляем усилитель Делаем волшебный раствор "
         "добавляем усилитель и это все растворяется в горячей воде")
    out = squeeze_repeats(t)
    assert out.lower().count("делаем волшебный раствор") == 1, out
    assert "и это все растворяется в горячей воде" in out

    # 2) соседний дубль с деградацией токена (паш/пш)
    t = ("наносим кислотный ополаскиватель чтобы привести уровень паш в норму "
         "наносим кислотный ополаскиватель чтобы привести уровень пш в норму "
         "если этого не сделать то оставшаяся щелочь")
    out = squeeze_repeats(t)
    assert out.lower().count("наносим кислотный ополаскиватель") == 1, out
    assert "если этого не сделать" in out

    # 3) соседний дубль в начале сегмента
    t = ("второй раз помещаем его в скафандр теперь набираем второй раз помещаем "
         "его в скафандр теперь набираем кислотный раствор в этот скафандр")
    out = squeeze_repeats(t)
    assert out.lower().count("второй раз помещаем") == 1, out
    assert "кислотный раствор в этот скафандр" in out

    # 4) точное окно-повтор не в хвосте
    t = ("теперь вот так теперь этот коврик выглядит в макро Вот так теперь этот "
         "коврик выглядит в макро у каждой кисоньки был свой уголок")
    out = squeeze_repeats(t)
    assert out.lower().count("этот коврик выглядит в макро") == 1, out
    assert "у каждой кисоньки" in out

    # 5) короткое эхо в хвосте
    t = ("финальное взвешивание и весит он теперь шесть килограмм сто грамм это "
         "значит он похудел на полтора килограмма финальное взвешивание")
    out = squeeze_repeats(t)
    assert out.lower().count("финальное взвешивание") == 1, out

    # НЕГАТИВНЫЕ: легитимный текст не корежим
    legit = ("стирка обычного ковра стоит три доллара за квадратный метр а стирка ковра "
             "в моче неважно чей кошачий собачий человечий стоит шесть долларов за "
             "квадратный метр потому что нужно выводить эту вонь из ковра")
    assert squeeze_repeats(legit) == legit, squeeze_repeats(legit)
    legit2 = "доброе утро коллеги сегодня мы поговорим о теории информации и кодировании"
    assert squeeze_repeats(legit2) == legit2
    print("  все пользовательские петли схлопнуты, легитимный текст цел")


def test_repr_tuple_unwrap():
    print("== repr-кортежи декодера не проходят в текст ==")
    from app.asr.torch_engines import decode_item_text
    from app.core.postprocess import clean_text

    # новые версии GigaAM: (text, token_ids, token_frames)
    assert decode_item_text(("привет мир", [45, 33], [0, 1])) == "привет мир"
    # старые версии: голая строка
    assert decode_item_text("привет мир") == "привет мир"
    assert decode_item_text(None) == ""

    garbage = "('Сегодня все правильно меня зовут владимир', [45, 33, 31, 42, 0, 59])"
    cleaned = clean_text(garbage)
    assert cleaned == "Сегодня все правильно меня зовут владимир", cleaned
    # обычный текст не затрагивается
    normal = "(привет) это обычная фраза со скобками [и квадратами]"
    assert clean_text(normal) == normal.strip(" \\t\\r\\n·—-") or clean_text(normal).startswith("(привет)")
    assert "[45, 33]" not in clean_text(garbage)




def test_cross_segment_echo():
    print("== петли, рваные по границе сегментов ==")
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
    print("  результат:", texts)
    # голова второго сегмента, повторяющая хвост первого, срезана
    assert texts[0].endswith("добавляем усилитель"), texts
    assert texts[1] == "и это все растворяется в воде", texts
    # огрызок-эхо выброшен
    assert not any(t.strip().lower() == "финальное взвешивание" for t in texts), texts
    # легитимная короткая реплика и легитимное продолжение НЕ тронуты
    assert "Да." in texts, texts
    assert any(t.startswith("а стирка ковра в моче") for t in texts), texts




def test_legit_text_untouched():
    print("== легитимные повторы и реплики НЕ режутся ==")
    from app.core.postprocess import Segment, sanitize_segments, squeeze_repeats

    # ударения-усиления, параллелизм, перечисления, однокоренные повторы
    legit = [
        "это очень очень важная точка и совершенно совершенно неприемлемый подход",
        "первый раздел второй раздел третий раздел вот и всё оглавление",
        "и коты и коты и собаки жили в этой квартире по очереди",
        "стирка обычного ковра стоит три доллара за квадратный метр а стирка ковра "
        "в моче неважно чей кошачий собачий человечьий стоит шесть долларов за "
        "квадратный метр потому что нужно выводить эту вонь из ковра",
        "вода в реке была холодна вода в озере была теплее вода в море оказалась самой тёплой",
    ]
    for line in legit:
        out = squeeze_repeats(line)
        assert out == line, f"изменён легитимный текст: {line!r} -> {out!r}"

    # диалоговые реплики и catch-фраза вдали друг от друга (разные сегменты)
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
    print("  усиления, перечисления, реплики «Да.» и catch-фраза в разных сегментах целы")


if __name__ == "__main__":
    test_energy_chunker()
    test_smart_chunker_with_fake_vad()
    test_tiny_tail_merged()
    test_tiny_tail_kept_when_merge_exceeds_limit()
    test_real_vad_available()
    test_continuous_speech_no_pauses()
    test_postprocess()
    test_user_repeat_cases()
    test_cross_segment_echo()
    test_legit_text_untouched()
    test_repr_tuple_unwrap()
    print("\nВСЕ ТЕСТЫ ПРОЙДЕНЫ")
