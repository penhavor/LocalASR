"""Post-processing: hallucination/repeat cleanup and text assembly.

Whisper-family models emit junk on silence/noise ("[music]", "Thanks for
watching!", endless phrase repeats). This module strips it and joins
segments into readable paragraphs.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from ..i18n import _



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
#  Whisper junk inserts
# --------------------------------------------------------------------------- #

# Bracketed "sound effects" and service marks
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

# Auto-subtitle parasite phrases (whole segment)
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

# Music notes and other symbolic noise
NOTES = re.compile(r"[♪♫♬🎵🎶]+")
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF\U00002190-\U000021FF\U00002B00-\U00002BFF]"
)

# The same word 3+ times in a row
REPEATED_WORD = re.compile(r"\b(\w{2,})(?:\s+\1\b){2,}", re.IGNORECASE | re.UNICODE)

LANGUAGE_TOKEN = re.compile(r"<\|[^|>]{1,24}\|>")
ASTERISKS = re.compile(r"\*{2,}")


# --------------------------------------------------------------------------- #
#  Single segment cleanup
# --------------------------------------------------------------------------- #


def unwrap_repr_container(text: str) -> str:
    """ "('text', [45, 33, …])" -> "text".

    Second-line defense: if an engine leaked a tuple/list repr into the
    output, extract the string part. Fires ONLY when the whole string is a
    container literal — regular text is untouched.
    """
    stripped = text.strip()
    if not stripped.startswith(("('", '(\"', "['", '[\"')):
        return text
    if not stripped.endswith((")", "]")):
        return text
    try:
        import ast

        obj = ast.literal_eval(stripped)
    except Exception:  # noqa: BLE001 - not a literal => leave as is
        return text
    if isinstance(obj, (tuple, list)):
        strings = [part for part in obj if isinstance(part, str)]
        if strings:
            return " ".join(strings)
    return text


def clean_text(text: str) -> str:
    """Remove noise from a string (no cross-segment repeat analysis)."""
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
    # almost no letters — a "segment" of punctuation
    letters = sum(ch.isalpha() for ch in low)
    if letters < 2:
        return True
    return False


def _collapse_repeated_tail(text: str, min_chunk: int = 4, max_repeats: int = 40) -> str:
    """Collapse a stuck tail: "phrase phrase phrase phrase" -> "phrase"."""
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
    """Collapse in-segment decoder loops ("this this this matters" -> "this matters").

    Handles loops inside one segment (important for CTC/RNNT models without
    punctuation): adjacent duplicate runs, exact repeated windows, tail echo.
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
    """Collapse adjacent duplicate runs: "X X …" -> "X …".

    X is a run of >= min_span words repeated immediately (a couple of
    degraded tokens are tolerated — that is a decoder loop, not speech).
    Legitimate parallel constructions are not adjacent and/or differ by
    more than 15% of tokens.
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
                # boundary tokens must match: otherwise the "loop" is a
                # shifted window and removal would glue unrelated words
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
    """Remove the SECOND exact occurrence of a word window within a segment.

    Legitimate repeats of short phrases (< min_span words) are kept.
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
    """Remove an exact echo in the tail ("final weighting … final weighting").

    A single-word echo is removed only at a large gap so legitimate short
    endings are not cut.
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
            # a single-word echo is removed only when the word opened the
            # segment; otherwise it is a legitimate last word
            if size == 1 and i != 0:
                continue
            if tuple(norms[i:i + size]) == tail and (n - size) - (i + size) >= gap_needed:
                return " ".join(tokens[:n - size])
    return text


def trim_echo_prefix(text: str, prev_norms: List[str], min_span: int = 3, max_span: int = 40) -> str:
    """Trim a segment start that repeats the previous segment's end.

    A decoder loop often breaks exactly at the chunk boundary: tail of A and
    head of B are the same run. Neither segment contains the repeat alone,
    so it is caught only at the seam.
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
    """A short stub (<= max_words) fully repeating the start or the end of
    the previous segment — an exact match with the neighbor, i.e. a loop."""
    norms = [_norm_token(t) for t in text.split()]
    if not (1 <= len(norms) <= max_words) or not prev_norms:
        return False
    m = len(prev_norms)
    if m < len(norms):
        return False
    return prev_norms[:len(norms)] == norms or prev_norms[m - len(norms):] == norms


def _sentence_norms(text: str) -> List[str]:
    """Normalized sentences of the text (for deduplication)."""
    return [
        re.sub(r"\W+", " ", part.lower()).strip()
        for part in re.split(r"(?<=[.!?…])\s+", text)
        if part.strip()
    ]


def _is_echo_sentence(candidate: str, previous: str, threshold: float = 0.80) -> bool:
    """Is candidate an echo of previous?

    Conservative rules (never eat legitimate content):
      * bigram Dice >= threshold -> near-verbatim repeat;
      * candidate is SHORTER and >=90% of its bigrams occur in previous ->
        a fragment of the previous phrase (loop tail); a longer candidate
        that continues previous carries new information and is kept;
      * >=90% of candidate words occur in previous, order is similar
        (Dice >= 0.55) and there are >=5 words -> degraded loop repeat.
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
    """Remove duplicate/echo sentences — within a segment AND across boundaries.

    RNNT/CTC models loop on noise/music: a phrase repeats with gradual
    degradation, sometimes flowing across the chunk boundary. `context` is a
    sliding window of normalized sentences of previous segments. Sentences
    shorter than min_words are never deduplicated ("Yes." in a dialogue).
    """
    if not text:
        return text
    sentences = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", text) if part.strip()]
    if not sentences:
        return text

    # history of recent sentences (including dropped variants — a degraded
    # repeat may be closer to the DROPPED one than to the kept one)
    history: List[str] = list(context or [])[-lookback:]

    kept: List[str] = []
    for sentence in sentences:
        candidate = re.sub(r"\W+", " ", sentence.lower()).strip()
        if len(candidate.split()) < min_words:
            # short dialogue lines ("Yes.") are not deduplicated
            kept.append(sentence)
            continue
        if any(_is_echo_sentence(candidate, previous, threshold) for previous in history[-lookback:]):
            continue    # echo — drop; the first occurrence is already kept
        kept.append(sentence)
        history.append(candidate)
        if len(history) > lookback * 3:
            history = history[-lookback * 2:]
    return " ".join(kept)


# --------------------------------------------------------------------------- #
#  Segment stream cleanup
# --------------------------------------------------------------------------- #


def _similarity(a: str, b: str) -> float:
    """Fast string similarity (bigram Dice) — no heavy dependencies."""
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
    """Clean a segment list: hallucinations, repeats, low-quality output.

    Returns a NEW list (dropped segments are flagged in place).
    """
    result: List[Segment] = []
    recent_texts: List[tuple] = []     # (normalized text, segment end)
    recent_sentences: List[str] = []   # sliding window for cross-segment echoes
    prev_norms: List[str] = []         # tokens of the previous KEPT segment

    for segment in segments:
        original = segment.text or ""
        text = clean_text(original)

        if aggressive:
            text = squeeze_repeats(text)
            text = squeeze_duplicate_sentences(text, context=recent_sentences)
            # loop torn at the chunk boundary: the head repeats the previous
            # tail — trim the shared "ring"
            text = trim_echo_prefix(text, prev_norms)
            for sentence_norm in _sentence_norms(text):
                if len(sentence_norm.split()) >= 3:
                    recent_sentences.append(sentence_norm)
            recent_sentences = recent_sentences[-8:]

        segment.text = text

        if not text or is_junk_segment(text):
            segment.dropped = True
            segment.drop_reason = "junk/empty"
            continue

        if aggressive and is_echo_fragment(text, prev_norms):
            segment.dropped = True
            segment.drop_reason = "echo stub of the previous segment"
            continue

        # technical quality thresholds (not all engines provide them)
        if aggressive:
            if segment.avg_logprob and segment.avg_logprob < min_avg_logprob and segment.duration > 1.0:
                segment.dropped = True
                segment.drop_reason = f"low confidence (logprob {segment.avg_logprob:.2f})"
                continue
            if segment.no_speech_prob >= max_no_speech_prob and len(text.split()) <= 2:
                segment.dropped = True
                segment.drop_reason = "looks like silence"
                continue
            if segment.compression_ratio and segment.compression_ratio > max_compression_ratio:
                segment.dropped = True
                segment.drop_reason = f"degenerate text (cr {segment.compression_ratio:.2f})"
                continue

        # cross-segment repeats (a typical Whisper/GigaAM loop on noise):
        # either high similarity or one text almost fully contained in the
        # other (a short echo segment after a long one). Loops are back to
        # back (gap ~ 0) while legitimate short replies are seconds apart —
        # so 1-2 word segments must also be close in time.
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
                    # bigram containment is stricter than word-level — use it
                    big_s = {f"{a}|{b}" for a, b in zip(shorter.split(), shorter.split()[1:])}
                    big_l = {f"{a}|{b}" for a, b in zip(longer.split(), longer.split()[1:])}
                    if big_s and len(big_s & big_l) / len(big_s) >= 0.90:
                        duplicate = True
                        break
        if duplicate:
            segment.dropped = True
            segment.drop_reason = "repeat of the previous segment"
            continue

        recent_texts.append((normalized, segment.end))
        if len(recent_texts) > window * 3:
            recent_texts = recent_texts[-window * 2:]
        # remember tokens of the kept segment for the seam echo detector
        prev_norms = [_norm_token(t) for t in text.split()][-80:]
        result.append(segment)

    return result


# --------------------------------------------------------------------------- #
#  Text assembly
# --------------------------------------------------------------------------- #

SHORT_ECHO_GAP_SECONDS = 2.0    # a short reply ("Yes.") counts as a loop only
                                # when it is back to back with an identical one
PARAGRAPH_GAP_SECONDS = 1.6     # a longer pause starts a new paragraph
PARAGRAPH_MAX_CHARS = 900       # paragraphs are also split by length
SENTENCE_END = ".!?…"


def ensure_sentence_end(text: str) -> str:
    text = text.rstrip()
    if not text:
        return text
    if text[-1] not in SENTENCE_END and text[-1] not in ",;:—-\"'»”":
        return text + "."
    return text


def capitalize_start(text: str) -> str:
    """Capitalize the first LETTER (leading dashes/quotes do not interfere)."""
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
    """Plain text (for .txt / .md)."""
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
    """Split long segments into subtitle chunks (for SRT/VTT)."""
    out: List[Segment] = []
    for segment in segments:
        words = segment.text.split()
        if not words:
            continue
        if len(segment.text) <= max_chars and segment.duration <= max_seconds:
            out.append(segment)
            continue

        # split by words, preferring punctuation boundaries
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

        # no word timings — split proportionally to length
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
    """Header for .md/.txt-time (optional, user-enabled)."""
    lines = [
        f"# {source_name}",
        "",
        _("- Model: {}").format(model_name),
        _("- Language: {}").format(language or _("detected automatically")),
        _("- Duration: {}").format(format_timestamp(duration_seconds)),
        _("- Processed in: {:.1f} s ({:.1f}× realtime)").format(
            processing_seconds, duration_seconds / processing_seconds
        )
        if processing_seconds > 0 else _("- Processed in: {:.1f} s").format(processing_seconds),
    ]
    for key, value in (extra or {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)
