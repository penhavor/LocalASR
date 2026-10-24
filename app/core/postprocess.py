"""Постобработка распознавания: чистка галлюцинаций, повторов и оформление текста.

Whisper (любой сборки) на тишине/шуме иногда выдаёт мусор:
    «[музыка]», «[аплодисменты]», «Спасибо за просмотр!», «Субтитры сделал DimaTorzok»,
    бесконечные повторы одной фразы и т.п.
Здесь всё это вычищается, а сегменты склеиваются в читаемые абзацы.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------- #
#  Структуры
# --------------------------------------------------------------------------- #


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List["Word"] = field(default_factory=list)
    language: str = ""
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0
    dropped: bool = False
    drop_reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 0.0


# --------------------------------------------------------------------------- #
#  Мусорные вставки Whisper
# --------------------------------------------------------------------------- #

# Скобки с «звуковыми эффектами» и служебные пометки
BRACKET_NOISE = re.compile(
    r"""[\[\(]\s*(?:
        музыка|музыку|мюзик|melody|music|applause|аплодисменты|хлопки|смех|laughter|
        шум|noise|тишина|silence|звук|sound|вступительная\s+музыка|финальная\s+музыка|
        играет\s+музыка|звуки?\s+\w+|beep|bloop|ding|gunshot|applauding|clapping|
        cheering|crowd|laughter\s*\w*|inaudible|неразборчиво|нрзб|неслышно|
        субтитры?|subtitles?|captions?|cc|перевод|translation|
        спасибо\s+за\s+просмотр|подпишись|подписывайтесь|лайк|like\s+and\s+subscribe|
        blur|beep\b|\*|\.\.\.|♪+|music\s*playing|\bvideo\b|\baudio\b|\bend\b|\bthe\s+end\b
    )\s*[\]\)]""",
    re.IGNORECASE | re.VERBOSE,
)

# Фразы-паразиты авто-субтитров (целиком в сегменте)
JUNK_PHRASES = [
    "спасибо за просмотр",
    "спасибо за внимание",
    "субтитры сделал",
    "субтитры подготовил",
    "субтитры предоставлены",
    "перевод субтитров",
    "подпишись на канал",
    "подписывайтесь на канал",
    "ставь лайк",
    "ставьте лайки",
    "thanks for watching",
    "thank you for watching",
    "subscribe to my channel",
    "like and subscribe",
    "subtitles by",
    "captioned by",
    "music by",
]

# Музыкальные ноты и прочий «символьный шум»
NOTES = re.compile(r"[♪♫♬🎵🎶]+")
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF\U00002190-\U000021FF\U00002B00-\U00002BFF]"
)

# «Бла-бла» повторы: одно и то же слово 3+ раза подряд
REPEATED_WORD = re.compile(r"\b(\w{2,})(?:\s+\1\b){2,}", re.IGNORECASE | re.UNICODE)

LANGUAGE_TOKEN = re.compile(r"<\|[^|>]{1,24}\|>")
ASTERISKS = re.compile(r"\*{2,}")


# --------------------------------------------------------------------------- #
#  Очистка одного сегмента
# --------------------------------------------------------------------------- #


def unwrap_repr_container(text: str) -> str:
    """«('текст', [45, 33, …])» -> «текст».

    Защита второго эшелона: если какой-то движок/версия модели пронесла в вывод
    repr кортежа/списка (текст + индексы токенов), вытаскиваем строковую часть.
    Срабатывает ТОЛЬКО когда строка целиком является литералом-контейнером,
    начинающимся с кавычки, — обычный текст не затрагивается.
    """
    stripped = text.strip()
    if not stripped.startswith(("('", '(\"', "['", '[\"')):
        return text
    if not stripped.endswith((")", "]")):
        return text
    try:
        import ast

        obj = ast.literal_eval(stripped)
    except Exception:  # noqa: BLE001 - не литерал => не трогаем
        return text
    if isinstance(obj, (tuple, list)):
        strings = [part for part in obj if isinstance(part, str)]
        if strings:
            return " ".join(strings)
    return text


def clean_text(text: str) -> str:
    """Убирает шум из строки (без анализа повторов между сегментами)."""
    if not text:
        return ""
    cleaned = unwrap_repr_container(text)
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = LANGUAGE_TOKEN.sub(" ", cleaned)
    cleaned = NOTES.sub(" ", cleaned)
    cleaned = EMOJI.sub(" ", cleaned)
    cleaned = BRACKET_NOISE.sub(" ", cleaned)
    cleaned = ASTERISKS.sub(" ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip(" \t\r\n·—-")
    return cleaned


def is_junk_segment(text: str) -> bool:
    low = text.lower().strip(" .!?,;:…")
    if not low:
        return True
    for phrase in JUNK_PHRASES:
        if phrase in low and len(low) < len(phrase) + 40:
            return True
    # почти нет букв — «сегмент» из знаков препинания
    letters = sum(ch.isalpha() for ch in low)
    if letters < 2:
        return True
    return False


def _collapse_repeated_tail(text: str, min_chunk: int = 4, max_repeats: int = 40) -> str:
    """Убирает «заевшую пластинку»: «фраза фраза фраза фраза» -> «фраза»."""
    tokens = text.split()
    total = len(tokens)
    if total < min_chunk * 2:
        return text
    max_chunk = min(total // 2, 40)
    for size in range(min_chunk, max_chunk + 1):
        chunk = tokens[-size:]
        repeats = 1
        index = total - size
        while index - size >= 0 and tokens[index - size:index] == chunk:
            repeats += 1
            index -= size
        if repeats >= 2:
            return " ".join(tokens[: index + size])
    return text


def squeeze_repeats(text: str) -> str:
    """«это это это важно» -> «это важно», «фраза фраза фраза» -> «фраза».

    Дополнительно снимает декодерные «петли» внутри сегмента (важно для
    CTC/RNNT-моделей БЕЗ пунктуации, где весь сегмент — одно «предложение»
    и межпредложенческая дедупликация бессильна):
      * соседние дубли-прогоны («делаем раствор добавляем усилитель Делаем
        раствор добавляем усилитель и это все…»);
      * точные повторы окна в другом месте сегмента;
      * короткое эхо в хвосте («…финальное взвешивание … финальное взвешивание»).
    """
    def repl(match: "re.Match") -> str:
        return match.group(1)

    previous = None
    current = text
    for _ in range(4):
        current = REPEATED_WORD.sub(repl, current)
        if current == previous:
            break
        previous = current
    current = squeeze_adjacent_spans(current)
    current = drop_repeat_windows(current)
    collapsed = _collapse_repeated_tail(current)
    collapsed = drop_tail_echo(collapsed)
    return re.sub(r"\s{2,}", " ", collapsed).strip()


def _norm_token(token: str) -> str:
    return re.sub(r"\W+", "", token.lower())


def _seq_ratio(a: List[str], b: List[str]) -> float:
    import difflib

    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def squeeze_adjacent_spans(
    text: str, min_span: int = 5, max_span: int = 40, threshold: float = 0.85
) -> str:
    """Схлопывает соседние дубль-прогоны: «X X …» -> «X …».

    X — подряд идущий прогон из >= min_span слов, повторенный сразу же
    (допускается деградация пары токенов: «…уровень паш в норму» /
    «…уровень пш в норму» — это петля декодера, а не речь). Легитимные
    параллельные конструкции под это не попадают: они НЕ смежные и/или
    отличаются больше, чем на 15% токенов.
    """
    tokens = text.split()
    norms = [_norm_token(t) for t in tokens]
    changed = True
    while changed:
        changed = False
        n = len(tokens)
        for size in range(min(max_span, n // 2), min_span - 1, -1):
            hit = False
            for i in range(n - 2 * size + 1):
                a = norms[i:i + size]
                b = norms[i + size:i + 2 * size]
                # граничные токены прогонов должны совпадать: иначе «петля»
                # на самом деле съехавшее окно и удаление склеит чужие слова
                if a[0] != b[0] or a[-1] != b[-1]:
                    continue
                if _seq_ratio(a, b) >= threshold:
                    del tokens[i + size:i + 2 * size]
                    del norms[i + size:i + 2 * size]
                    changed = hit = True
                    n = len(tokens)
                    break
            if hit:
                break
    return " ".join(tokens)


def drop_repeat_windows(text: str, min_span: int = 4, max_span: int = 40) -> str:
    """Убирает ТОЧНЫЙ повтор окна слов в другом месте сегмента (второе вхождение).

    «…выглядит в макро Вот так теперь этот коврик выглядит в макро у каждой…»
    Легитимные повторы коротких фраз (< min_span слов) сохраняются.
    """
    tokens = text.split()
    norms = [_norm_token(t) for t in tokens]
    removed = [False] * len(tokens)
    for size in range(max_span, min_span - 1, -1):
        seen: Dict[tuple, int] = {}
        for i in range(len(tokens) - size + 1):
            if any(removed[i:i + size]):
                continue
            key = tuple(norms[i:i + size])
            if not any(key):
                continue
            first = seen.get(key)
            if first is None:
                seen[key] = i
            elif i >= first + size:
                for k in range(i, i + size):
                    removed[k] = True
    if not any(removed):
        return text
    return " ".join(t for t, r in zip(tokens, removed) if not r)


def drop_tail_echo(text: str, min_span: int = 1, max_span: int = 5, min_gap: int = 3) -> str:
    """Убирает точное эхо в хвосте: «финальное взвешивание … финальное взвешивание».

    Однословное эхо («…съёмки этого видео Клиентка») снимается только при
    большом отступе, чтобы не резать легитимные короткие концовки.
    """
    tokens = text.split()
    norms = [_norm_token(t) for t in tokens]
    n = len(tokens)
    for size in range(max_span, min_span - 1, -1):
        gap_needed = min_gap + 1 if size == 1 else min_gap
        if n - 2 * size < gap_needed:
            continue
        tail = tuple(norms[n - size:])
        if not any(tail):
            continue
        for i in range(0, n - size - gap_needed + 1):
            # однословное эхо снимаем только когда слово открывало сегмент
            # («Клиентка … видео Клиентка»); иное — легитимное последнее слово
            if size == 1 and i != 0:
                continue
            if tuple(norms[i:i + size]) == tail and (n - size) - (i + size) >= gap_needed:
                return " ".join(tokens[:n - size])
    return text


def trim_echo_prefix(text: str, prev_norms: List[str], min_span: int = 3, max_span: int = 40) -> str:
    """Срезает начало сегмента, повторяющее конец предыдущего.

    Петля декодера часто рвётся ровно по границе куска: хвост сегмента A и
    голова сегмента B — один и тот же прогон («…добавляем усилитель» |
    «Делаем волшебный раствор добавляем усилитель и это все…»). Внутри
    каждого сегмента повтора при этом нет, поэтому ловим только на стыке.
    """
    if not prev_norms:
        return text
    tokens = text.split()
    norms = [_norm_token(t) for t in tokens]
    n, m = len(norms), len(prev_norms)
    for size in range(min(max_span, n - 1, m), min_span - 1, -1):
        a = prev_norms[m - size:]
        b = norms[:size]
        if a[0] != b[0] or a[-1] != b[-1]:
            continue
        ratio = 1.0 if a == b else _seq_ratio(a, b)
        threshold = 0.9 if size < 5 else 0.85
        if ratio >= threshold:
            return " ".join(tokens[size:])
    return text


def is_echo_fragment(text: str, prev_norms: List[str], max_words: int = 4) -> bool:
    """Короткий огрызок (<= max_words слов), целиком повторяющий начало или
    конец предыдущего сегмента («…финальное взвешивание» | «финальное
    взвешивание»). Раньше такие проходили защитой «предложения короче 3 слов
    не трогать» — но та защита про реплики диалога («Да.»), а здесь точное
    совпадение с соседом, т.е. петля."""
    norms = [_norm_token(t) for t in text.split()]
    if not (1 <= len(norms) <= max_words) or not prev_norms:
        return False
    m = len(prev_norms)
    if m < len(norms):
        return False
    return prev_norms[:len(norms)] == norms or prev_norms[m - len(norms):] == norms


def _sentence_norms(text: str) -> List[str]:
    """Нормализованные предложения текста (для дедупликации)."""
    return [
        re.sub(r"\W+", " ", part.lower()).strip()
        for part in re.split(r"(?<=[.!?…])\s+", text)
        if part.strip()
    ]


def _is_echo_sentence(candidate: str, previous: str, threshold: float = 0.80) -> bool:
    """candidate — эхо previous?

    Правила (консервативные, чтобы не съесть легитимный контент):
      * Dice-схожесть биграмм ≥ threshold  → почти дословный повтор;
      * candidate КОРОЧЕ и его биграммы ≥90% содержатся в previous →
        «обломок» предыдущей фразы (хвост петли); если candidate ДЛИННЕЕ и
        продолжает previous — он несёт новую информацию, оставляем;
      * уникальные слова candidate ≥90% покрыты словами previous И порядок
        похож (Dice ≥ 0.55), и слов ≥5 → деградировавший повтор петли.
    """
    dice = _similarity(candidate, previous)
    if dice >= threshold:
        return True
    cand_tokens = candidate.split()
    prev_tokens = previous.split()
    if not cand_tokens or not prev_tokens:
        return False
    cand_bigrams = {f"{a}|{b}" for a, b in zip(cand_tokens, cand_tokens[1:])}
    prev_bigrams = {f"{a}|{b}" for a, b in zip(prev_tokens, prev_tokens[1:])}
    if (
        cand_bigrams
        and len(cand_tokens) <= len(prev_tokens)
        and len(cand_bigrams & prev_bigrams) / len(cand_bigrams) >= 0.90
    ):
        return True
    cand_words = set(cand_tokens)
    prev_words = set(prev_tokens)
    if (
        len(cand_words) >= 5
        and len(cand_words & prev_words) / len(cand_words) >= 0.90
        and dice >= 0.55
    ):
        return True
    return False


def squeeze_duplicate_sentences(
    text: str,
    threshold: float = 0.80,
    lookback: int = 4,
    context: Optional[Sequence[str]] = None,
    min_words: int = 3,
) -> str:
    """Убирает дубли/эхо-предложения — внутри сегмента И через границу сегментов.

    RNNT/CTC-модели на шуме и музыке «зацикливаются»: фраза повторяется 2–3
    раза с постепенной деградацией, в том числе «перетекая» через границу
    куска. `context` — нормализованные предложения предыдущих сегментов
    (скользящее окно), чтобы ловить повтор «конец сегмента A = начало B».
    Предложение короче min_words слов не дедуплицируется («Да.» в диалоге).
    Из дублей оставляем самое длинное (в нём меньше «съеденных» слов).
    """
    if not text:
        return text
    sentences = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", text) if part.strip()]
    if not sentences:
        return text

    # история последних предложений (включая уже отброшенные варианты —
    # деградировавший повтор может быть ближе к ОТБРОШЕННОМУ, чем к оставленному)
    history: List[str] = list(context or [])[-lookback:]

    kept: List[str] = []
    for sentence in sentences:
        candidate = re.sub(r"\W+", " ", sentence.lower()).strip()
        if len(candidate.split()) < min_words:
            # «Да.» / «Ну что.» — диалогные мелочи не дедуплицируем
            kept.append(sentence)
            continue
        if any(_is_echo_sentence(candidate, previous, threshold) for previous in history[-lookback:]):
            continue    # эхо — отбрасываем, первое вхождение уже оставлено
        kept.append(sentence)
        history.append(candidate)
        if len(history) > lookback * 3:
            history = history[-lookback * 2:]
    return " ".join(kept)


# --------------------------------------------------------------------------- #
#  Очистка потока сегментов
# --------------------------------------------------------------------------- #


def _similarity(a: str, b: str) -> float:
    """Быстрая схожесть строк (Дайс по биграммам) — без тяжёлых зависимостей."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    def bigrams(text: str) -> set:
        tokens = text.lower().split()
        if len(tokens) < 2:
            return {tokens[0]} if tokens else set()
        return {f"{tokens[i]}|{tokens[i + 1]}" for i in range(len(tokens) - 1)}

    set_a, set_b = bigrams(a), bigrams(b)
    if not set_a or not set_b:
        return 0.0
    return 2 * len(set_a & set_b) / (len(set_a) + len(set_b))


def sanitize_segments(
    segments: Sequence[Segment],
    *,
    aggressive: bool = True,
    min_avg_logprob: float = -1.6,
    max_no_speech_prob: float = 0.85,
    max_compression_ratio: float = 3.2,
    duplicate_similarity: float = 0.94,
    window: int = 6,
) -> List[Segment]:
    """Чистит список сегментов: галлюцинации, повторы, низкокачественный вывод.

    Возвращает НОВЫЙ список (исходные сегменты помечаются флагом dropped).
    """
    result: List[Segment] = []
    recent_texts: List[tuple] = []   # (нормализованный текст, конец сегмента)
    recent_sentences: List[str] = []   # скользящее окно для межсегментных эхо-повторов
    prev_norms: List[str] = []         # токены предыдущего ПРИНЯТОГО сегмента (стыковые петли)

    for segment in segments:
        original = segment.text or ""
        text = clean_text(original)

        if aggressive:
            text = squeeze_repeats(text)
            text = squeeze_duplicate_sentences(text, context=recent_sentences)
            # петля, рваная по границе кусков: голова сегмента повторяет хвост
            # предыдущего — срезаем это общее «кольцо»
            text = trim_echo_prefix(text, prev_norms)
            for sentence_norm in _sentence_norms(text):
                if len(sentence_norm.split()) >= 3:
                    recent_sentences.append(sentence_norm)
            recent_sentences = recent_sentences[-8:]

        segment.text = text

        if not text or is_junk_segment(text):
            segment.dropped = True
            segment.drop_reason = "мусор/пусто"
            continue

        if aggressive and is_echo_fragment(text, prev_norms):
            segment.dropped = True
            segment.drop_reason = "огрызок-эхо предыдущего сегмента"
            continue

        # технические пороги качества (есть не у всех движков)
        if aggressive:
            if segment.avg_logprob and segment.avg_logprob < min_avg_logprob and segment.duration > 1.0:
                segment.dropped = True
                segment.drop_reason = f"низкая уверенность (logprob {segment.avg_logprob:.2f})"
                continue
            if segment.no_speech_prob >= max_no_speech_prob and len(text.split()) <= 2:
                segment.dropped = True
                segment.drop_reason = "похоже на тишину"
                continue
            if segment.compression_ratio and segment.compression_ratio > max_compression_ratio:
                segment.dropped = True
                segment.drop_reason = f"дегенеративный текст (cr {segment.compression_ratio:.2f})"
                continue

        # повторы между сегментами (типичная петля Whisper/GigaAM на шуме):
        # либо высокая схожесть, либо один текст почти целиком содержится
        # в другом (короткий «эхо-сегмент» после длинного).
        # ВАЖНО: петля идёт стык-в-стык (gap ~ 0), а легитимные короткие
        # реплики диалога («Да.» … «Да.») разделены секундами/минутами —
        # поэтому для сегментов в 1-2 слова требуем временной близости.
        normalized = re.sub(r"\W+", " ", text.lower()).strip()
        short_segment = len(text.split()) <= 2
        duplicate = False
        for previous, prev_end in recent_texts[-window:]:
            if short_segment and (segment.start - prev_end) > SHORT_ECHO_GAP_SECONDS:
                continue
            if _similarity(normalized, previous) >= duplicate_similarity:
                duplicate = True
                break
            shorter, longer = sorted((normalized, previous), key=len)
            if shorter and len(longer) > len(shorter) * 1.3:
                set_s = set(shorter.split())
                set_l = set(longer.split())
                if set_s and len(set_s & set_l) / len(set_s) >= 0.92:
                    # биграммный containment строже словесного — проверяем им
                    big_s = {f"{a}|{b}" for a, b in zip(shorter.split(), shorter.split()[1:])}
                    big_l = {f"{a}|{b}" for a, b in zip(longer.split(), longer.split()[1:])}
                    if big_s and len(big_s & big_l) / len(big_s) >= 0.90:
                        duplicate = True
                        break
        if duplicate:
            segment.dropped = True
            segment.drop_reason = "повтор предыдущего сегмента"
            continue

        recent_texts.append((normalized, segment.end))
        if len(recent_texts) > window * 3:
            recent_texts = recent_texts[-window * 2:]
        # запоминаем токены принятого сегмента для стыкового эхо-детектора
        prev_norms = [_norm_token(t) for t in text.split()][-80:]
        result.append(segment)

    return result


# --------------------------------------------------------------------------- #
#  Сборка текста
# --------------------------------------------------------------------------- #

SHORT_ECHO_GAP_SECONDS = 2.0    # короткая реплика («Да.») считается петлёй,
                                # только если прозвучала стык-в-стык с такой же
PARAGRAPH_GAP_SECONDS = 1.6     # пауза длиннее — новый абзац
PARAGRAPH_MAX_CHARS = 900       # и по длине тоже режем
SENTENCE_END = ".!?…"


def ensure_sentence_end(text: str) -> str:
    text = text.rstrip()
    if not text:
        return text
    if text[-1] not in SENTENCE_END and text[-1] not in ",;:—-\"'»”":
        return text + "."
    return text


def capitalize_start(text: str) -> str:
    """Первая БУКВА становится заглавной (вводные тире/кавычки не мешают)."""
    if not text:
        return text
    for index, char in enumerate(text[:12]):
        if char.isalpha():
            upper = char.upper()
            if upper == char:
                return text
            return text[:index] + upper + text[index + 1:]
    return text


def join_plain(segments: Iterable[Segment], paragraphs: bool = True) -> str:
    """Обычный текст (для .txt / .md)."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    last_end: Optional[float] = None

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        text = capitalize_start(text)
        gap = 0.0 if last_end is None else max(0.0, segment.start - last_end)

        start_new = False
        if paragraphs:
            if gap >= PARAGRAPH_GAP_SECONDS:
                start_new = True
            elif current_len + len(text) > PARAGRAPH_MAX_CHARS and text[-1:] in SENTENCE_END:
                start_new = True

        if start_new and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0

        current.append(text)
        current_len += len(text) + 1
        last_end = segment.end

    if current:
        chunks.append(" ".join(current))

    body = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    return re.sub(r"[ \t]{2,}", " ", body).strip() + ("\n" if body else "")


def join_with_timestamps(segments: Iterable[Segment], milliseconds: bool = False) -> str:
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        lines.append(f"[{format_timestamp(segment.start, milliseconds)}] {capitalize_start(text)}")
    return "\n".join(lines) + ("\n" if lines else "")


def format_timestamp(seconds: float, milliseconds: bool = False, comma: bool = False) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    sep = "," if comma else "."
    if milliseconds:
        msec = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{msec:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def split_for_subtitles(segments: Sequence[Segment], max_chars: int = 84, max_seconds: float = 6.0) -> List[Segment]:
    """Режет длинные сегменты на субтитрные куски (для SRT/VTT)."""
    out: List[Segment] = []
    for segment in segments:
        words = segment.text.split()
        if not words:
            continue
        if len(segment.text) <= max_chars and segment.duration <= max_seconds:
            out.append(segment)
            continue

        # режем по словам, стараясь попадать в границы знаков препинания
        if segment.words and len(segment.words) == len(words):
            buffer_words: List[Word] = []
            buffer_text: List[str] = []
            for word in segment.words:
                buffer_words.append(word)
                buffer_text.append(word.text.strip())
                joined = " ".join(t for t in buffer_text if t)
                ends_sentence = joined and joined[-1] in SENTENCE_END
                long_enough = len(joined) >= max_chars * 0.6
                too_long = len(joined) > max_chars
                time_long = (buffer_words[-1].end - buffer_words[0].start) >= max_seconds
                if (ends_sentence and long_enough) or too_long or time_long:
                    out.append(
                        Segment(
                            start=buffer_words[0].start,
                            end=buffer_words[-1].end,
                            text=joined,
                            words=list(buffer_words),
                            language=segment.language,
                        )
                    )
                    buffer_words, buffer_text = [], []
            if buffer_words:
                out.append(
                    Segment(
                        start=buffer_words[0].start,
                        end=buffer_words[-1].end,
                        text=" ".join(t for t in (w.text.strip() for w in buffer_words) if t),
                        words=list(buffer_words),
                        language=segment.language,
                    )
                )
            continue

        # слов с таймингами нет — режем «на глаз» пропорционально длине
        total_chars = max(1, sum(len(w) + 1 for w in words))
        per_char = segment.duration / total_chars
        cursor = segment.start
        buffer = []
        length = 0
        for word in words:
            buffer.append(word)
            length += len(word) + 1
            if length >= max_chars or word[-1:] in SENTENCE_END and length >= max_chars * 0.6:
                span = per_char * length
                out.append(
                    Segment(start=cursor, end=min(segment.end, cursor + span), text=" ".join(buffer),
                            language=segment.language)
                )
                cursor = min(segment.end, cursor + span)
                buffer, length = [], 0
        if buffer:
            out.append(Segment(start=cursor, end=segment.end, text=" ".join(buffer), language=segment.language))
    return out


def build_header(
    *,
    source_name: str,
    model_name: str,
    language: str,
    duration_seconds: float,
    processing_seconds: float,
    extra: Optional[dict] = None,
) -> str:
    """Шапка для .md/.txt-time (по желанию пользователя)."""
    lines = [
        f"# {source_name}",
        "",
        f"- Модель: {model_name}",
        f"- Язык: {language or 'определён автоматически'}",
        f"- Длительность: {format_timestamp(duration_seconds)}",
        f"- Обработано за: {processing_seconds:.1f} с "
        f"({(duration_seconds / processing_seconds):.1f}× реального времени)"
        if processing_seconds > 0 else f"- Обработано за: {processing_seconds:.1f} с",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)
