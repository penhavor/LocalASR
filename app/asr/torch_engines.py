"""Движки на PyTorch/transformers: Qwen3-ASR (Alibaba) и GigaAM (Sber).

Обе модели скачиваются из Hugging Face в папку models/ как обычные репозитории
transformers, а инференс идёт локально через torch:

* Qwen3-ASR — мультимодальная модель (аудио-энкодер + Qwen3-LM): 30 языков,
  автоопределение языка, punctuation из коробки. Работает через
  ``AutoProcessor.apply_transcription_request`` + ``generate``.
* GigaAM — conformer-модель Сбера (CTC/RNNT, best-in-class на русском).
  Грузится через ``AutoModel.from_pretrained(..., trust_remote_code=True)``;
  родной ``transcribe`` умеет только ≤25 секунд и на каждом вызове гоняет
  ffmpeg через subprocess (на Windows это мигающие консоли), поэтому длинные
  записи мы режем сами и считаем БЫСТРЫМ путём:
    1. куски по ~20 секунд собираются в батчи (энкодер грузит GPU целиком,
       а не по одному куску);
    2. RNNT-декодирование выполняется на CPU в несколько потоков — без
       синхронизации GPU↔CPU на каждый токен (родной цикл .item() тормозит);
    3. временные WAV не пишутся вовсе — тензоры идут в модель напрямую.
  Если структура модели нестандартная (другое поколение GigaAM и т.п.) —
  автоматический откат на «родной» покадровый путь с пропатченным загрузчиком
  звука (без subprocess/ffmpeg).

Длинное аудио оба движка получают кусками (у тишины), чтобы:
  * не съедать гигабайты памяти на трёхчасовой лекции;
  * показывать живой прогресс и поддерживать отмену.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import tempfile
import time
import types
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..constants import ENGINE_GIGAAM, ENGINE_QWEN_ASR, SAMPLE_RATE
from ..core.postprocess import Segment
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

log = get_logger("asr.torch")

# Максимальная длина куска для каждого движка (секунды)
QWEN_CHUNK_SECONDS = 120.0     # Qwen3-ASR умеет длинные окна, но 120 с —
#                                безопасный потолок для потребительских GPU
#                                (6 ГБ VRAM): при нехватке памяти кусок
#                                автоматически делится пополам (см. _infer_qwen)
QWEN_SPLIT_FLOOR_SECONDS = 5.0 # предел OOM-деления: куски короче не режем
GIGAAM_CHUNK_SECONDS = 20.0    # родной transcribe GigaAM ограничен ~25 секундами

# Куски короче этого на модели лучше не подавать: на обрывках <1 с
# seq2seq-модели галлюцинируют («Спасибо за просмотр»), а CTC/RNNT выдают
# мусорные токены. Короткий хвост вместо отдельного куска сливается с
# предыдущим (см. _merge_short_tail) — содержимое при этом не теряется.
MIN_CHUNK_SECONDS = 1.0

# Батчинг GigaAM: сколько ~20-секундных кусков уходит в один forward энкодера
GIGAAM_DEFAULT_BATCH = 8
GIGAAM_MAX_BATCH = 16


# --------------------------------------------------------------------------- #
#  Нарезка аудио по тишине
# ---------------------------------------------------------------------------


def _moving_energy_min_index(segment: np.ndarray, win: int) -> int:
    """Индекс окна наименьшей энергии (RMS) — O(n) через кумулятивные суммы.

    Замена np.convolve(segment², ones(win)/win, 'valid'): «наивная» свёртка
    стоит O(n·win) и на 20-секундных кусках съедала секунды чистого CPU
    на каждый кусок (заметная доля всего распознавания). argmin суммы окна
    равен argmin RMS, поэтому результат побайтово совпадает со старым.
    """
    if win <= 0 or segment.size <= win:
        return -1
    square = np.square(segment, dtype=np.float32)
    cumulative = np.cumsum(square)
    # sums[i] = сумма square[i:i+win] — ровно как «valid»-свёртка с ones(win)
    sums = cumulative[win - 1:]
    sums[1:] -= cumulative[:-win]
    return int(np.argmin(sums))


def _merge_short_tail(
    bounds: List[Tuple[int, int]],
    sample_rate: int,
    chunk_seconds: float,
    search_seconds: float,
) -> List[Tuple[int, int]]:
    """Сливает короткий хвостовой кусок (<1 с) с предыдущим.

    Последний кусок может оказаться обрывком в пару сотен миллисекунд —
    ровно тем входом, на котором модели придумывают «спасибо за просмотр»
    и мусорные токены. Слияние разрешено, пока объединённый кусок не выйдет
    за безопасный лимит `chunk + search + MIN_CHUNK`:
      * HF (27 с + 1.5 с)   -> максимум 29.5 с — строго короче 30-секундного
                               окна Whisper, т.е. short-form режим сохраняется;
      * GigaAM (20 с + 1 с) -> максимум 22 с при родном лимите ~25 с;
      * Qwen (300 с + 2 с)  -> жёсткого лимита нет.
    Если слияние всё же вывело бы за лимит — хвост остаётся отдельным куском
    (потерять контент хуже, чем рискнуть огрехом на коротком фрагменте).
    """
    if len(bounds) < 2:
        return bounds
    min_samples = int(MIN_CHUNK_SECONDS * sample_rate)
    limit_samples = int((chunk_seconds + search_seconds + MIN_CHUNK_SECONDS) * sample_rate)
    start, end = bounds[-1]
    if end - start >= min_samples:
        return bounds
    prev_start, _prev_end = bounds[-2]
    if end - prev_start <= limit_samples:
        return [*bounds[:-2], (prev_start, end)]
    return bounds


def iter_silence_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    search_seconds: float = 2.0,
    window_seconds: float = 0.4,
) -> Iterator[Tuple[int, int]]:
    """Генератор границ (start, end) в сэмплах: куски до chunk_seconds,
    граница ищется в окрестности ±search_seconds по минимуму громкости."""
    total = len(audio)
    if total == 0:
        return
    win = max(1, int(window_seconds * sample_rate))
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        target = start + int(chunk_seconds * sample_rate)
        if target >= total:
            end = total
        else:
            lo = max(start + win, start + int((chunk_seconds - search_seconds) * sample_rate))
            hi = min(total, start + int((chunk_seconds + search_seconds) * sample_rate))
            end = target
            if hi - lo > win:
                idx = _moving_energy_min_index(audio[lo:hi], win)
                if idx >= 0:
                    end = lo + idx + win // 2
            if end <= start:
                end = target
        bounds.append((start, min(end, total)))
        start = min(end, total)
    yield from _merge_short_tail(bounds, sample_rate, chunk_seconds, search_seconds)


def _vad_pause_points(
    audio: np.ndarray,
    sample_rate: int,
    min_pause_ms: int = 300,
) -> Optional[List[int]]:
    """Центры речевых пауз (в сэмплах) от Silero VAD.

    None — VAD недоступен (нет faster-whisper/onnxruntime, ошибка модели):
    вызывающий код откатывается на энергетическую нарезку.
    Модель silero_vad_v6.onnx лежит внутри пакета faster-whisper — интернет
    для неё НЕ нужен.
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception:  # noqa: BLE001
        return None
    try:
        options = VadOptions(
            min_speech_duration_ms=100,
            min_silence_duration_ms=min_pause_ms,
            speech_pad_ms=0,          # нужна честная граница, без расширения
        )
        speeches = get_speech_timestamps(audio, vad_options=options, sampling_rate=sample_rate)
    except Exception as exc:  # noqa: BLE001
        log.debug("Silero VAD не отработал — буду резать по минимуму громкости: %s", exc)
        return None
    points: List[int] = []
    for previous, following in zip(speeches, speeches[1:]):
        gap_start = int(previous.get("end") or 0)
        gap_end = int(following.get("start") or 0)
        if gap_end > gap_start:
            points.append((gap_start + gap_end) // 2)
    return points


def iter_smart_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    search_seconds: float = 1.0,
    use_vad: bool = True,
    min_pause_ms: int = 200,
) -> Iterator[Tuple[int, int]]:
    """Границы кусков, которые НЕ режут слова посередине.

    Приоритет: центры пауз, найденных Silero VAD (позднейшая пауза в окне
    [половина куска, chunk+search]). Если VAD недоступен или в окне нет ни
    одной паузы (непрерывная речь/музыка) — минимум громкости, как раньше.

    Это сильно улучшает текст на «трудных» записях (видео с музыкой, шумом,
    быстрой речью без пауз): ровный энергетический рез режет слова, и модель
    выдаёт обрывки («уси.» + «Ный раствор» вместо «усилительный раствор»).
    """
    total = len(audio)
    if total == 0:
        return
    points = _vad_pause_points(audio, sample_rate, min_pause_ms) if use_vad else None
    if not points:
        yield from iter_silence_chunks(audio, sample_rate, chunk_seconds, search_seconds=search_seconds)
        return

    import bisect

    max_len = int(chunk_seconds * sample_rate)
    search = int(search_seconds * sample_rate)
    win = max(1, int(0.4 * sample_rate))
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        target = start + max_len
        if target >= total:
            bounds.append((start, total))
            break
        lo = start + max_len // 2
        hi = min(total, target + search)
        index_lo = bisect.bisect_left(points, lo)
        index_hi = bisect.bisect_right(points, hi)
        end = 0
        if index_hi > index_lo:
            end = points[index_hi - 1]        # самая поздняя пауза окна
        else:
            # непрерывная речь: энергетический минимум в узком окне у цели
            seg_lo = max(start + win, target - search)
            seg_hi = min(total, target + search)
            if seg_hi - seg_lo > win:
                idx = _moving_energy_min_index(audio[seg_lo:seg_hi], win)
                if idx >= 0:
                    end = seg_lo + idx + win // 2
            if end <= start:
                end = target
        end = max(end, start + 1)
        bounds.append((start, min(end, total)))
        start = min(end, total)
    yield from _merge_short_tail(bounds, sample_rate, chunk_seconds, search_seconds)


def _build_gigaam_manual(mod: Any, config: Any, folder: Path) -> Any:
    """Собирает GigaAM вручную: init на cpu + явная загрузка весов.

    Обход meta-инициализации transformers v5, с которой hydra-внутренности
    (mel-фильтры, регистры RNNT) создаются на cpu и ловят конфликт устройств.
    """
    import torch

    config.name_or_path = str(folder)
    model = mod.GigaAMModel(config)
    weights = folder / "pytorch_model.bin"
    safetensors = folder / "model.safetensors"
    if safetensors.is_file():
        from safetensors.torch import load_file

        state = load_file(str(safetensors))
    elif weights.is_file():
        try:
            # mmap заметно снижает пик памяти на больших моделях
            state = torch.load(str(weights), map_location="cpu", weights_only=True, mmap=True)
        except (RuntimeError, OSError, ValueError):
            state = torch.load(str(weights), map_location="cpu", weights_only=True)
    else:
        raise EngineError(f"В {folder} нет ни pytorch_model.bin, ни model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # префикс base_model_prefix может отличаться — пробуем со снятием «model.»
    if unexpected and all(key.startswith("model.") for key in unexpected):
        state = {key[len("model."):]: value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise EngineError(
            f"В весах GigaAM не хватает {len(missing)} тензоров (например {missing[:3]})"
        )
    return model


def decode_item_text(item: Any) -> str:
    """Достаёт текст из результата декодера.

    Новые версии modeling_gigaam.py возвращают из CTC/RNNT-декода кортеж
    ``(text, token_ids, token_frames)``; старые — голую строку. Naive
    ``str(tuple)`` раньше писал в файл целиком repr кортежа вместе со
    списками индексов («('текст', [45, 33, …])») — отсюда «мусорный» вывод.
    Здесь берём первый строковой элемент кортежа/списка, строку — как есть.
    """
    if isinstance(item, (tuple, list)):
        for part in item:
            if isinstance(part, str):
                return part
        return ""
    return str(item or "")


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Пишет mono float32 → WAV s16le (стандартная библиотека, без зависимостей)."""
    pcm = np.clip(audio, -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(data.tobytes())


def missing_files_hint(exc: BaseException) -> str:
    """Подсказка к ошибкам «в папке модели нет такого-то файла».

    Типичная причина — прерванное скачивание (сеть моргнула, установку отменили).
    Вместо голого стектрейса объясняем пользователю, что делать.
    """
    low = str(exc).lower()
    markers = (
        "preprocessor_config.json", "config.json", "tokenizer", "does not appear to have a file named",
        "can't load", "cannot load", "is not a local folder", "no such file or directory",
        "feature extractor", "processor_config.json",
    )
    if any(marker in low for marker in markers):
        return (
            "\n\nПохоже, файлы модели скачаны не полностью или папка повреждена "
            "(такое случается при обрыве сети во время установки).\n"
            "Что делать: удали модель (карточка модели → «Удалить») и установи заново — "
            "целые файлы будут пропущены, докачается только недостающее."
        )
    return ""


def transformers_version_hint(exc: BaseException, model_kind: str, min_version: str) -> str:
    """Подсказка, если установленный transformers слишком старый для модели."""
    low = str(exc).lower()
    if any(marker in low for marker in (
        "unrecognized", "not recognized", "unknown model", "keyerror", "does not support",
        "has no attribute", "invalid model type",
    )):
        return (
            f"\n\nУстановленный transformers не знает «{model_kind}» — нужен transformers>={min_version}.\n"
            "Обнови его: программа сама предложит установку при следующем запуске модели "
            "(или: Сервис → Диагностика и восстановление).\n"
            "ВАЖНО: если новая версия transformers была установлена только что, ПЕРЕЗАПУСТИ "
            "программу — уже импортированная старая версия остаётся в памяти процесса до перезапуска."
        )
    return ""


def reset_transformers_caches() -> int:
    """Сбрасывает lru-кеши проверок «пакет доступен?» внутри transformers.

    transformers кэширует результаты is_accelerate_available()/is_torch_available()
    и подобных проверок (lru_cache на всю жизнь процесса). Если пакет установили
    ПРЯМО ВО ВРЕМЯ работы программы (штатный сценарий: движок ставится по
    требованию), кэш продолжает отвечать «недоступно» — загрузка Qwen3-ASR падала
    с «requires `accelerate`» на свежеустановленный accelerate. Сброс кэшей во
    всех загруженных модулях transformers возвращает проверки в актуальное
    состояние БЕЗ перезапуска программы. Возвращает число сброшенных кэшей.
    """
    cleared = 0
    seen: set = set()
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if not (name == "transformers" or name.startswith("transformers.")):
            continue
        try:
            attributes = list(vars(module).values())
        except TypeError:
            continue
        for attr in attributes:
            # Классы и модули не трогаем: ленивые объекты transformers при
            # ЛЮБОМ getattr пытаются импортировать бэкенды и строят сообщения
            # requires_backends (секунды на атрибут). Все нужные lru_cache —
            # функции; обход без них занимает ~0.1 с.
            if isinstance(attr, (type, types.ModuleType)):
                continue
            try:
                if id(attr) in seen:
                    continue
                cache_clear = getattr(attr, "cache_clear", None)
                if cache_clear is None or not callable(getattr(attr, "cache_info", None)):
                    continue    # это не lru_cache-обёртка
            except Exception:  # noqa: BLE001
                continue
            seen.add(id(attr))
            try:
                cache_clear()
                cleared += 1
            except Exception:  # noqa: BLE001
                continue
    if cleared:
        log.debug("transformers: сброшено lru-кэшей доступности: %d", cleared)
    return cleared


def read_wav_mono(path: Path, sample_rate: int):
    """Читает WAV s16le в torch-тензор float32 (без ffmpeg и subprocess)."""
    import torch

    with wave.open(str(path), "rb") as fh:
        params_ok = fh.getnchannels() == 1 and fh.getsampwidth() == 2
        raw = fh.readframes(fh.getnframes())
    if not params_ok:
        raise ValueError(f"Ожидался моно s16le WAV: {path}")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


def safe_load_audio(path: Any, sample_rate: int = SAMPLE_RATE):
    """Безопасная замена gigaam.load_audio: НИКАКИХ subprocess/ffmpeg-консолей.

    Родной load_audio модели запускает ``ffmpeg`` через subprocess.run без
    CREATE_NO_WINDOW — из pythonw.exe это показывает чёрное окно консоли на
    каждый кусок аудио. Здесь: WAV читаем стандартным wave, всё остальное —
    нашим декодером (он ffmpeg запускает невидимо).
    """
    path = Path(str(path))
    if path.suffix.lower() == ".wav":
        try:
            return read_wav_mono(path, sample_rate)
        except Exception:  # noqa: BLE001 - не наш WAV (другой формат семплов)
            pass
    from ..core.ffmpeg import decode_audio

    import torch

    audio = decode_audio(path, sample_rate)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


# --------------------------------------------------------------------------- #
#  Общий базовый класс для torch-движков
# --------------------------------------------------------------------------- #


class TorchASREngine(ASREngine):
    """Общее: проверка доступности, устройство, разгрузка памяти."""

    chunk_seconds: float = 60.0
    #: все модули, без которых движок не заведётся (GUI предложит доустановку)
    import_names: Tuple[str, ...] = ("torch", "transformers")

    @classmethod
    def is_available(cls) -> bool:
        return all(importlib.util.find_spec(name) is not None for name in cls.import_names)

    # -- вспомогательное -------------------------------------------------- #
    def _pick_device(self, hardware: Dict[str, Any]) -> Tuple[str, Any]:
        import torch

        wanted = str(hardware.get("device") or "auto")
        # torch носит CUDA-рантайм с собой (pip-wheel), поэтому ему НЕ нужна
        # проверка ctranslate2 (hardware["cuda_ok"]) — она про другой движок.
        # Достаточно torch.cuda.is_available().
        if wanted == "cuda":
            use_cuda = torch.cuda.is_available()
        elif wanted == "cpu":
            use_cuda = False
        else:  # auto
            use_cuda = bool(torch.cuda.is_available())
        self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32
        self.compute_type = "float16" if use_cuda else "float32"
        return self.device, dtype

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self.loaded_model_id = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _ensure_model(self) -> Any:
        if self._model is None:
            raise EngineError(f"Модель движка {self.name} не загружена")
        return self._model

    def __init__(self) -> None:
        super().__init__()
        self._processor: Any = None


# --------------------------------------------------------------------------- #
#  Qwen3-ASR
# --------------------------------------------------------------------------- #


class QwenASREngine(TorchASREngine):
    name = ENGINE_QWEN_ASR
    # qwen3_asr появился в transformers 5.13.0 (2026-07-03) — более ранние
    # 5.x модель НЕ знают, поэтому пин строгий
    packages = ["torch>=2.6", "transformers>=5.13", "accelerate>=1.0", "soundfile>=0.12"]
    import_name = "transformers"
    import_names = ("torch", "transformers", "accelerate")
    chunk_seconds = QWEN_CHUNK_SECONDS
    MIN_TRANSFORMERS = (5, 13)

    # ------------------------------------------------------------------ #
    @classmethod
    def is_available(cls) -> bool:
        """Доступность = пакеты на месте И transformers знает qwen3_asr.

        Версию проверяем через метаданные (importlib.metadata) — БЕЗ импорта
        transformers: импорт тяжёлый, а is_available() дёргается из GUI
        (вкладка «Движки», кнопка «РАБОТАТЬ») и не должна подвешивать окно.
        """
        if not super().is_available():
            return False
        try:
            from importlib.metadata import version as pkg_version

            raw = pkg_version("transformers")
        except Exception:  # noqa: BLE001
            return False
        parts = []
        for chunk in raw.split(".")[:2]:
            digits = "".join(ch for ch in chunk if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        while len(parts) < 2:
            parts.append(0)
        return tuple(parts) >= cls.MIN_TRANSFORMERS

    # ------------------------------------------------------------------ #
    @staticmethod
    def _from_pretrained_dtype(auto_cls: Any, path: str, dtype: Any, **kwargs: Any) -> Any:
        """from_pretrained с учётом переименования torch_dtype -> dtype в 5.10+."""
        try:
            return auto_cls.from_pretrained(path, dtype=dtype, **kwargs)
        except TypeError:
            # transformers < 5.10: аргумент назывался torch_dtype
            return auto_cls.from_pretrained(path, torch_dtype=dtype, **kwargs)

    def _load_qwen_model(self, auto_cls: Any, path: str, dtype: Any, device: str) -> Any:
        """Загрузка весов с тройной страховкой от «accelerate-казусов».

        Каскад:
          1. device_map=device (быстрый путь, нужен accelerate);
          2. при «requires accelerate» — сброс lru-кэшей transformers
             (пакет могли установить только что, прямо во время работы
             программы, а кэш помнит старое «нет») и повтор;
          3. если accelerate реально нет/невидим — загрузка БЕЗ device_map
             и обычный .to(device): для одной видеокарты результат тот же.
        """
        if importlib.util.find_spec("accelerate") is not None:
            try:
                return self._from_pretrained_dtype(auto_cls, path, dtype, device_map=device)
            except ValueError as exc:
                if "accelerate" not in str(exc).lower():
                    raise
                cleared = reset_transformers_caches()
                log.warning(
                    "transformers не увидел установленный accelerate (устаревший lru_cache, "
                    "сброшено кэшей: %d) — повторяю загрузку", cleared,
                )
                try:
                    return self._from_pretrained_dtype(auto_cls, path, dtype, device_map=device)
                except ValueError as exc2:
                    if "accelerate" not in str(exc2).lower():
                        raise
                    log.warning("accelerate по-прежнему недоступен — гружу без device_map (.to(%s))", device)
        else:
            log.info("accelerate не установлен — гружу Qwen3-ASR без device_map (.to(%s))", device)
        # запасной путь БЕЗ accelerate: загрузка на CPU и перенос на устройство
        model = self._from_pretrained_dtype(auto_cls, path, dtype)
        return model.to(device)

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Не удалось импортировать torch/transformers: {exc}\n"
                "Недостающие компоненты программа предложит установить автоматически "
                "при запуске модели (или: Сервис → Диагностика и восстановление)."
            ) from exc

        path = str(model.resolved_path)
        device, dtype = self._pick_device(hardware)
        log.debug("Qwen3-ASR: загружаю %s (device=%s, dtype=%s)", path, device, dtype)
        started = time.time()
        try:
            self._processor = AutoProcessor.from_pretrained(path)
            self._model = self._load_qwen_model(AutoModelForMultimodalLM, path, dtype, device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            hint = missing_files_hint(exc) or transformers_version_hint(exc, "Qwen3-ASR", "5.13")
            log.error("Qwen3-ASR не загрузился из %s: %s", path, exc, exc_info=True)
            raise EngineError(f"Не удалось загрузить Qwen3-ASR из {path}: {exc}{hint}") from exc
        self.loaded_model_id = model.model_id
        log.info(
            "Qwen3-ASR загружен: %s (%s, %s) за %.1f с",
            model.model_id, device, self.compute_type, time.time() - started,
        )

    # ------------------------------------------------------------------ #
    def _qwen_inputs(self, chunk: np.ndarray, language: Optional[str], prompt: Optional[str]) -> Any:
        """Готовит входы модели через apply_transcription_request.

        ВАЖНО: аудио передаётся МАССИВОМ 16 кГц (именно это принимает
        transformers 5.x — make_list_of_audio ждёт numpy/torch-массив, путь
        или URL). Вариант «кортеж (массив, sample_rate)» из старых примеров
        приводит к «Invalid input type. Must be a single audio…» — этот баг
        здесь исправлен.
        """
        apply = getattr(self._processor, "apply_transcription_request", None)
        if apply is None:
            raise EngineError(
                "Установленный transformers не поддерживает apply_transcription_request "
                "для Qwen3-ASR — обнови transformers (нужен >= 5.13)."
            )
        try:
            return apply(audio=chunk, language=language, prompt=prompt)
        except TypeError as exc:
            # страховка от смены сигнатуры между версиями transformers
            text = str(exc)
            if "prompt" in text:
                log.debug("apply_transcription_request без prompt: %s", exc)
                return apply(audio=chunk, language=language)
            if "language" in text:
                log.debug("apply_transcription_request без language: %s", exc)
                return apply(audio=chunk)
            raise

    def _infer_qwen_once(
        self, model: Any, chunk: np.ndarray, language: Optional[str], prompt: Optional[str],
    ) -> Tuple[str, str]:
        """Один forward+generate на кусок. Возвращает (текст, язык)."""
        import torch

        inputs = self._qwen_inputs(chunk, language, prompt)
        inputs = inputs.to(model.device, model.dtype)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        decode = getattr(self._processor, "decode", None)
        if decode is None:
            raise EngineError("Процессор Qwen3-ASR без decode() — обнови transformers (>= 5.13)")
        try:
            parsed = decode(generated, return_format="parsed")[0]
        except (TypeError, ValueError):
            # совсем старые/новые варианты API: декодируем «сыро» и чистим маркеры
            raw = decode(generated, skip_special_tokens=True)
            text = raw[0] if isinstance(raw, (list, tuple)) else str(raw)
            parsed = {"transcription": text, "language": ""}
        if isinstance(parsed, (list, tuple)):
            parsed = parsed[0] if parsed else {}
        if not isinstance(parsed, dict):
            parsed = {"transcription": str(parsed), "language": ""}
        return str(parsed.get("transcription") or "").strip(), str(parsed.get("language") or "")

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        try:
            import torch

            oom_class = getattr(torch.cuda, "OutOfMemoryError", None)
            if oom_class is not None and isinstance(exc, oom_class):
                return True
        except Exception:  # noqa: BLE001
            pass
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _infer_qwen(
        self,
        model: Any,
        chunk: np.ndarray,
        language: Optional[str],
        prompt: Optional[str],
        offset_seconds: float,
    ) -> Tuple[str, str]:
        """Распознаёт кусок; при нехватке VRAM делит его пополам (рекурсивно).

        Потребительские GPU (6 ГБ VRAM) не всегда тянут 120-секундное окно
        энкодера Qwen3-ASR — вместо падения «CUDA out of memory» кусок
        режется пополам и половинки считаются по отдельности (тексты
        склеиваются). Предел деления — QWEN_SPLIT_FLOOR_SECONDS.
        """
        import torch

        try:
            return self._infer_qwen_once(model, chunk, language, prompt)
        except Exception as exc:  # noqa: BLE001
            if not self._is_cuda_oom(exc):
                raise
            floor_samples = int(QWEN_SPLIT_FLOOR_SECONDS * SAMPLE_RATE)
            if len(chunk) < floor_samples * 2:
                raise
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            mid = len(chunk) // 2
            log.warning(
                "Qwen3-ASR: не хватило VRAM на кусок %.1f с (около %.1f с) — делю пополам и повторяю",
                len(chunk) / SAMPLE_RATE, offset_seconds,
            )
            left = self._infer_qwen(model, np.ascontiguousarray(chunk[:mid]), language, prompt, offset_seconds)
            right = self._infer_qwen(
                model, np.ascontiguousarray(chunk[mid:]), language, prompt,
                offset_seconds + mid / SAMPLE_RATE,
            )
            text = (left[0] + (" " if left[0] and right[0] else "") + right[0]).strip()
            detected = left[1] or right[1]
            return text, detected

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
        model = self._ensure_model()
        started = time.time()
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        total = len(audio) or 1

        language = (options.get("language") or "").strip() or None
        prompt_parts = [options.get("initial_prompt") or "", options.get("hotwords") or ""]
        prompt = " ".join(part.strip() for part in prompt_parts if part.strip()) or None

        # нарезка по паузам (VAD, как у остальных движков): куски не режут
        # слова посередине, а короткие хвосты сливаются с предыдущим куском
        bounds = list(
            iter_smart_chunks(
                audio, SAMPLE_RATE, self.chunk_seconds, search_seconds=2.0,
                use_vad=bool(options.get("vad", True)),
            )
        )
        log.debug(
            "Qwen3-ASR: %d кусков (язык=%s, prompt=%s, аудио %.1f с)",
            len(bounds), language or "авто", bool(prompt), duration,
        )

        segments: List[Segment] = []
        language_detected = ""
        warnings: List[str] = []

        for start, end in bounds:
            if cancel and cancel():
                warnings.append("Распознавание остановлено пользователем")
                break
            chunk = audio[start:end]
            chunk_started = time.time()
            try:
                text, detected = self._infer_qwen(model, chunk, language, prompt, start / SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001
                hint = ""
                if self._is_cuda_oom(exc):
                    hint = (
                        "\n\nНе хватило видеопамяти даже на минимальный кусок. "
                        "Попробуй модель Qwen3-ASR 0.6B или другое устройство "
                        "(Сервис → Настройки → «Скорость и железо»)."
                    )
                log.error(
                    "Qwen3-ASR сбой на куске %.1f–%.1f с: %s",
                    start / SAMPLE_RATE, end / SAMPLE_RATE, exc, exc_info=True,
                )
                raise EngineError(f"Qwen3-ASR не смог распознать фрагмент: {exc}{hint}") from exc

            language_detected = detected or language_detected
            log.debug(
                "Qwen3-ASR: кусок %.1f–%.1f с -> %d симв. за %.1f с",
                start / SAMPLE_RATE, end / SAMPLE_RATE, len(text), time.time() - chunk_started,
            )
            if text:
                segment = Segment(start=start / SAMPLE_RATE, end=end / SAMPLE_RATE, text=text)
                segments.append(segment)
                if on_segment:
                    on_segment(segment)

            if progress:
                progress(end / total, f"Распознаю: {end / SAMPLE_RATE:.0f} с из {duration:.0f} с")

        if options.get("word_timestamps"):
            warnings.append("Qwen3-ASR не даёт таймкоды слов — сегменты по кускам аудио")

        return TranscriptResult(
            segments=segments,
            language=language_detected,
            audio_duration=duration,
            processed_duration=duration,
            elapsed_seconds=time.time() - started,
            engine=self.name,
            device=self.device,
            compute_type=self.compute_type,
            warnings=warnings,
        )


# --------------------------------------------------------------------------- #
#  GigaAM (Sber)
# --------------------------------------------------------------------------- #


def _rnnt_greedy_one(
    enc_frames: Any,
    length: int,
    decoder: Any,
    pred: Any,
    joint_net: Any,
    blank_id: int,
    max_symbols: int,
    tokenizer: Any,
) -> str:
    """RNNT greedy-декодирование ОДНОЙ последовательности на CPU.

    Математически повторяет GigaAM RNNTGreedyDecoding._greedy_decode, но:
      * проекция энкодера (joint.enc) уже посчитана батчем на GPU;
      * LSTM/joint работают на CPU в float32 — нет синхронизации GPU↔CPU
        на каждый токен (главный тормоз родного пути);
      * функцию можно запускать параллельно в нескольких потоках.
    """
    import torch

    hyp: List[int] = []
    state: Any = None
    last_label: Any = None
    with torch.inference_mode():
        for t in range(length):
            frame = enc_frames[t].view(1, 1, -1)
            not_blank = True
            emitted = 0
            while not_blank and emitted < max_symbols:
                g, hidden = decoder.predict(last_label, state)
                logits = joint_net(frame + pred(g))
                token = int(logits.argmax(dim=-1).item())
                if token == blank_id:
                    not_blank = False
                    # ВАЖНО: на blank состояние LSTM НЕ обновляется — ровно как
                    # в родном _greedy_decode (dec_state = hidden только при
                    # эмиссии токена). Иначе декод «съедает» слоги.
                else:
                    hyp.append(token)
                    state = hidden
                    last_label = torch.tensor([[token]])
                    emitted += 1
    return str(tokenizer.decode(hyp) or "")


class GigaAMEngine(TorchASREngine):
    name = ENGINE_GIGAAM
    packages = [
        "torch>=2.6",
        "torchaudio>=2.6",
        "transformers>=4.52",
        "hydra-core>=1.3,<2",
        "omegaconf>=2.3,<3",
        "sentencepiece>=0.2.2",
    ]
    import_name = "transformers"
    # soundfile проверяем явно: его импортирует сам код модели
    # (modeling_gigaam.py), без него загрузка падает ModuleNotFoundError
    import_names = ("torch", "transformers", "torchaudio", "hydra", "omegaconf", "sentencepiece", "soundfile")
    chunk_seconds = GIGAAM_CHUNK_SECONDS

    def __init__(self) -> None:
        super().__init__()
        self._rnnt_kit: Optional[Tuple[Any, ...]] = None
        self._fast_batch = GIGAAM_DEFAULT_BATCH

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        try:
            from transformers import AutoModel
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"Не удалось импортировать transformers: {exc}\n"
                "Недостающие компоненты программа предложит установить автоматически "
                "при запуске модели (или: Сервис → Диагностика и восстановление)."
            ) from exc

        # Код GigaAM внутри вызывает ffmpeg из PATH — подскажем ему наш бинарник
        # (пригодится только запасному пути; быстрый путь ffmpeg не использует)
        try:
            from ..core.ffmpeg import find_ffmpeg

            ffmpeg = find_ffmpeg()
            if ffmpeg:
                extra = str(Path(ffmpeg).parent)
                current = os.environ.get("PATH", "")
                if extra not in current:
                    os.environ["PATH"] = extra + os.pathsep + current
        except Exception:  # noqa: BLE001
            pass

        path = str(model.resolved_path)
        device, dtype = self._pick_device(hardware)
        # ВАЖНО: dtype при from_pretrained не передаём — внутренности модели
        # (mel-фильтры и т.п.) создаются hydra на cpu и конфликтуют с meta-
        # инициализацией transformers. Точность применяем после загрузки.
        try:
            loaded = AutoModel.from_pretrained(
                path, trust_remote_code=True, low_cpu_mem_usage=False
            )
        except ImportError as exc:
            # modeling_gigaam.py лениво импортирует pyannote (только для ихнего
            # longform с HF-токеном). Нам longform не нужен: длинные записи мы
            # режем сами, поэтому грузим модуль напрямую, без проверки transformers.
            log.info("AutoModel требует %s — гружу modeling_gigaam.py напрямую", exc)
            loaded = self._load_direct(path, dtype)
        except Exception as exc:  # noqa: BLE001
            try:
                loaded = self._load_direct(path, dtype)
            except Exception as direct_exc:  # noqa: BLE001
                log.error("GigaAM не загрузился из %s: %s / %s", path, exc, direct_exc, exc_info=True)
                raise EngineError(
                    f"Не удалось загрузить GigaAM из {path}: {exc} / {direct_exc}\n"
                    "Нужны пакеты: torch, torchaudio, transformers, hydra-core, omegaconf, sentencepiece."
                    + missing_files_hint(direct_exc)
                ) from exc
        try:
            loaded = loaded.to(device)
            # ВАЖНО: веса на GPU НЕ кастим в fp16. Родной GigaAM.forward сам
            # включает torch.autocast(fp16) вокруг энкодера, а его rotary-PE
            # (pe-буфер) всегда создаётся в fp32: на fp16-весах произведение
            # q(fp16)×cos(fp32) даёт fp32 и linear_q падает с dtype-mismatch.
            # fp32-веса + autocast = схема вендора: быстро и без крашей.
            if device == "cuda":
                self.compute_type = "float32+autocast(fp16)"
            loaded.eval()
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"Не удалось перевести GigaAM на {device}: {exc}") from exc
        self._model = loaded
        self._rnnt_kit = None
        self._fast_batch = GIGAAM_DEFAULT_BATCH
        self._patch_audio_loader()
        self.loaded_model_id = model.model_id
        log.info("GigaAM загружен: %s (%s, %s)", model.model_id, device, self.compute_type)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_direct(path: str, dtype: Any = None) -> Any:
        """Загрузка modeling_gigaam.py напрямую (без dynamic-module проверки)."""
        import importlib.util as iu

        module_path = Path(path) / "modeling_gigaam.py"
        if not module_path.is_file():
            raise EngineError(f"В {path} нет modeling_gigaam.py")
        spec = iu.spec_from_file_location("localasr_gigaam_modeling", str(module_path))
        if spec is None or spec.loader is None:
            raise EngineError(f"Не удалось прочитать {module_path}")
        mod = iu.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
            config = mod.GigaAMConfig.from_pretrained(path)
            try:
                # low_cpu_mem_usage=False: иначе transformers создаёт модель на
                # meta-устройстве, а hydra-внутренности (mel-фильтры) — на cpu,
                # и загрузка падает с «Tensor on device cpu is not on meta»
                return mod.GigaAMModel.from_pretrained(
                    path, config=config, low_cpu_mem_usage=False
                )
            except Exception as exc:  # noqa: BLE001
                # transformers v5 всё равно может увести init на meta-устройство:
                # собираем модель вручную и накладываем веса сами
                log.info("from_pretrained не смог (%s) — собираю GigaAM вручную", exc)
                return _build_gigaam_manual(mod, config, Path(path))
        except Exception:
            sys.modules.pop(spec.name, None)
            raise

    # ------------------------------------------------------------------ #
    def _patch_audio_loader(self) -> None:
        """Подменяет load_audio ВО ВСЕХ копиях модуля модели (без ffmpeg-консолей).

        modeling_gigaam.py делает ``sys.path.append(DIR_NAME)``, и hydra затем
        импортирует ``modeling_gigaam`` ЗАНОВО — получается вторая копия модуля,
        чей load_audio (subprocess+ffmpeg) оставался непропатченным и показывал
        чёрное окно консоли на каждый кусок аудио. Патчим все найденные копии:
        внешний класс, внутренний (hydra) класс и все модули с таким именем.
        """
        patched = 0
        seen: set = set()
        candidates: List[Any] = []
        inner = getattr(self._model, "model", None)
        for obj in (self._model, inner):
            if obj is None:
                continue
            module = sys.modules.get(type(obj).__module__ or "")
            if module is not None:
                candidates.append(module)
        for name, module in list(sys.modules.items()):
            if name == "modeling_gigaam" or name.endswith(".modeling_gigaam"):
                candidates.append(module)
        for module in candidates:
            if id(module) in seen:
                continue
            seen.add(id(module))
            if hasattr(module, "load_audio"):
                module.load_audio = safe_load_audio
                patched += 1
        log.debug("load_audio пропатчен в %d модулях GigaAM", patched)

    def unload(self) -> None:
        self._rnnt_kit = None
        super().unload()

    # ------------------------------------------------------------------ #
    #  Быстрый путь: батчи + CPU-декодирование RNNT
    # ------------------------------------------------------------------ #
    def _inner_asr(self) -> Any:
        """Внутренняя GigaAMASR (self._model.model) или None."""
        inner = getattr(self._model, "model", None)
        if inner is None or not hasattr(inner, "forward"):
            return None
        return inner

    def _fast_kind(self, inner: Any) -> str:
        """'rnnt' | 'ctc' | '' — поддерживает ли структура быстрый путь."""
        decoding = getattr(inner, "decoding", None)
        head = getattr(inner, "head", None)
        if decoding is None or head is None:
            return ""
        kind = type(decoding).__name__.upper()
        if "RNNT" in kind:
            joint = getattr(head, "joint", None)
            decoder = getattr(head, "decoder", None)
            tokenizer = getattr(decoding, "tokenizer", None)
            if (
                decoder is not None
                and hasattr(decoder, "predict")
                and tokenizer is not None
                and all(hasattr(joint, attr) for attr in ("enc", "pred", "joint_net"))
            ):
                return "rnnt"
            return ""
        if "CTC" in kind:
            if callable(head) and hasattr(decoding, "decode"):
                return "ctc"
        return ""

    def _rnnt_cpu_kit(self, inner: Any) -> Tuple[Any, ...]:
        """Копии мелких RNNT-сетей (decoder/pred/joint_net) на CPU в float32."""
        if self._rnnt_kit is not None:
            return self._rnnt_kit
        import torch

        head = inner.head
        with torch.no_grad():
            decoder = copy.deepcopy(head.decoder).cpu().float().eval()
            pred = copy.deepcopy(head.joint.pred).cpu().float().eval()
            joint_net = copy.deepcopy(head.joint.joint_net).cpu().float().eval()

        # Родной predict() для первого шага создаёт torch.zeros(...) с ГЛОБАЛЬНЫМ
        # default dtype — если окружение сменило его (например, на fp16), нули
        # разойдутся с fp32-весами копии. Подменяем на явный dtype весов.
        kit_dtype = next(decoder.parameters()).dtype
        original_predict = decoder.predict

        def predict(x: Any, state: Any, _dec=decoder, _orig=original_predict, _dt=kit_dtype):
            if x is not None:
                return _orig(x, state)
            import torch as _torch

            emb = _torch.zeros((1, 1, _dec.pred_hidden), dtype=_dt)
            g, hid = _dec.lstm(emb.transpose(0, 1), state)
            return g.transpose(0, 1), hid

        decoder.predict = predict
        self._rnnt_kit = (decoder, pred, joint_net)
        return self._rnnt_kit

    def _forward_single(self, inner: Any, piece: np.ndarray) -> Tuple[Any, Any]:
        """Forward энкодера на ОДИН кусок: (encoded (1,C,T), length (1,)).

        ВАЖНО: именно поштучно, без батч-паддинга. Энкодер GigaAM не полностью
        маскирует паддинг внутри conv-модулей, поэтому в батче короткие куски
        «отравляются» нулевым хвостом соседа (замер: до 70% относительной
        ошибки против поштучного forward) — текст плывёт. Поштучный forward
        побайтово совпадает с родным prepare_wav+forward модели.
        """
        import torch

        parameters = next(inner.parameters())
        wav = torch.from_numpy(np.ascontiguousarray(piece)).unsqueeze(0)
        wav = wav.to(device=parameters.device, dtype=parameters.dtype)
        length = torch.tensor([len(piece)], dtype=torch.int64, device=parameters.device)
        return inner.forward(wav, length)

    def _decode_group(self, inner: Any, kind: str, encoded_items: List[Tuple[Any, Any]]) -> List[str]:
        """Декодирует список (encoded, length) — по одному на кусок.

        RNNT: проекция joint.enc считается на устройстве модели, дальше greedy-
        цикл идёт на CPU в несколько потоков (без GPU-синхронизаций на токен).
        CTC: родной векторизованный декод GigaAM по каждому куску.
        """
        import torch

        if kind == "ctc":
            texts: List[str] = []
            with torch.inference_mode():
                for encoded, encoded_len in encoded_items:
                    for item in inner.decoding.decode(inner.head, encoded, encoded_len):
                        texts.append(decode_item_text(item))
            return texts

        decoding = inner.decoding
        head = inner.head
        blank_id = int(getattr(decoding, "blank_id", len(decoding.tokenizer)))
        max_symbols = int(getattr(decoding, "max_symbols", 10))
        decoder, pred, joint_net = self._rnnt_cpu_kit(inner)

        tasks = []
        with torch.inference_mode():
            for encoded, encoded_len in encoded_items:
                frames = encoded.transpose(1, 2)                  # (1, T, C)
                enc_proj = head.joint.enc(frames).float().cpu()[0]  # (T, joint_hidden)
                tasks.append(
                    (enc_proj, int(encoded_len[0]), decoder, pred, joint_net,
                     blank_id, max_symbols, decoding.tokenizer)
                )

        workers = min(len(tasks), max(1, os.cpu_count() or 1))
        if workers <= 1:
            return [_rnnt_greedy_one(*task) for task in tasks]

        # LSTM-шаг крошечный: внутриоперационный параллелизм torch только
        # мешает — выключаем его на время декодирования и распараллеливаем
        # по последовательностям (torch отпускает GIL в своих операциях).
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rnnt") as pool:
                return list(pool.map(lambda task: _rnnt_greedy_one(*task), tasks))
        finally:
            torch.set_num_threads(previous_threads)

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        import torch

        oom_class = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_class is not None and isinstance(exc, oom_class):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

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
        import torch

        self._ensure_model()
        started = time.time()
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        total = len(audio) or 1

        bounds = list(
            iter_smart_chunks(
                audio, SAMPLE_RATE, self.chunk_seconds, search_seconds=1.0,
                use_vad=bool(options.get("vad", True)),
            )
        )
        segments: List[Segment] = []
        warnings: List[str] = []

        inner = self._inner_asr()
        kind = self._fast_kind(inner) if inner is not None else ""
        if options.get("batch_size"):
            self._fast_batch = max(1, min(GIGAAM_MAX_BATCH, int(options["batch_size"])))
        if options.get("batched") is False:
            self._fast_batch = 1

        used_fast = False
        if kind:
            if kind == "rnnt":
                # копии маленьких RNNT-сетей для CPU-декодирования готовим
                # ЗАРАНЕЕ (вне inference_mode), чтобы тензоры были «обычными»
                self._rnnt_cpu_kit(inner)
            try:
                self._transcribe_fast(
                    inner, kind, audio, bounds, total, duration,
                    segments, warnings, progress, on_segment, cancel,
                )
                used_fast = True
            except Exception as exc:  # noqa: BLE001
                if segments:
                    # уже есть результат — падать на весь файл из-за быстрого
                    # пути не стоит, но и молчать нельзя
                    raise EngineError(f"GigaAM: сбой быстрого пути: {exc}") from exc
                log.warning("Быстрый путь GigaAM недоступен (%s) — иду последовательным", exc)
                if self._is_oom(exc):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                segments.clear()
                warnings.clear()

        if not used_fast:
            self._transcribe_stock(
                audio, bounds, total, duration, segments, warnings, progress, on_segment, cancel
            )

        if options.get("word_timestamps"):
            warnings.append("GigaAM использован без таймкодов слов (режим chunks)")

        return TranscriptResult(
            segments=segments,
            language="ru",
            audio_duration=duration,
            processed_duration=duration,
            elapsed_seconds=time.time() - started,
            engine=self.name,
            device=self.device,
            compute_type=self.compute_type,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    def _emit(self, segments, start, end, text, on_segment) -> None:
        text = str(text or "").strip()
        if text:
            segment = Segment(start=start / SAMPLE_RATE, end=end / SAMPLE_RATE, text=text)
            segments.append(segment)
            if on_segment:
                on_segment(segment)

    def _report(self, progress, end_sample, total, duration) -> None:
        if progress:
            progress(
                min(1.0, end_sample / max(total, 1)),
                f"Распознаю: {end_sample / SAMPLE_RATE:.0f} с из {duration:.0f} с",
            )

    def _transcribe_fast(
        self, inner, kind, audio, bounds, total, duration,
        segments, warnings, progress, on_segment, cancel,
    ) -> None:
        """Группами: энкодер ПОКУЧНО (без паддинга!), декод — параллельно.

        Группа из N кусков: сначала N поштучных forward'ов энкодера (точно как
        в родном пути), затем все N декодируются разом — RNNT в пуле потоков
        на CPU. Скорость даёт именно параллельный декод без GPU-синхронизаций;
        батчинг энкодера отключён из-за неполной маскировки паддинга в модели.
        """
        import torch

        index = 0
        count = len(bounds)
        while index < count:
            if cancel and cancel():
                warnings.append("Распознавание остановлено пользователем")
                return
            take = max(1, min(self._fast_batch, count - index))
            group_bounds = bounds[index: index + take]

            encoded_items: List[Tuple[Any, Any]] = []
            for start_sample, end_sample in group_bounds:
                if cancel and cancel():
                    warnings.append("Распознавание остановлено пользователем")
                    return
                while True:  # OOM: чистим кэш и пробуем ещё раз
                    try:
                        with torch.inference_mode():
                            encoded_items.append(
                                self._forward_single(inner, audio[start_sample:end_sample])
                            )
                        break
                    except Exception as exc:  # noqa: BLE001
                        if not self._is_oom(exc):
                            raise
                        log.warning("GPU не хватило памяти на кусок — очищаю кэш и повторяю")
                        try:
                            torch.cuda.empty_cache()
                        except Exception:  # noqa: BLE001
                            pass

            with torch.inference_mode():
                texts = self._decode_group(inner, kind, encoded_items)
            del encoded_items

            for (start_sample, end_sample), text in zip(group_bounds, texts):
                self._emit(segments, start_sample, end_sample, text, on_segment)
            index += len(group_bounds)
            self._report(progress, group_bounds[-1][1], total, duration)

    def _transcribe_stock(
        self, audio, bounds, total, duration, segments, warnings, progress, on_segment, cancel,
    ) -> None:
        """Запасной путь: родной model.transcribe() по одному куску через WAV.

        load_audio в модулях модели пропатчен на safe_load_audio, поэтому
        subprocess/ffmpeg здесь не запускается и консоль не мигает.
        """
        model = self._ensure_model()
        tmp_path: Optional[Path] = None
        try:
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = Path(handle.name)
            handle.close()

            for start, end in bounds:
                if cancel and cancel():
                    warnings.append("Распознавание остановлено пользователем")
                    return
                write_wav(tmp_path, audio[start:end], SAMPLE_RATE)
                try:
                    text = model.transcribe(str(tmp_path))
                except Exception as exc:  # noqa: BLE001
                    raise EngineError(f"GigaAM не смог распознать фрагмент: {exc}") from exc
                self._emit(segments, start, end, text, on_segment)
                self._report(progress, end, total, duration)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
