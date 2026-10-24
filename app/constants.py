"""Константы приложения: имена папок, пороговые значения, списки форматов."""

from __future__ import annotations

APP_NAME = "LocalASR"
APP_VERSION = "1.3.0"
ORG_NAME = "LocalASR"

# --- папки внутри проекта -------------------------------------------------- #
VENV_DIRNAME = ".venv"
MODELS_DIRNAME = "models"
OUTPUTS_DIRNAME = "outputs"
LOGS_DIRNAME = "logs"
RESOURCES_DIRNAME = "resources"
TEMP_DIRNAME = ".cache"

CATALOG_FILENAME = "models_catalog.json"
# Локальный каталог (resources/models_catalog.json) — основной и всегда рабочий.
# Отсюда приложение попробует подтянуть свежую версию (если есть сеть и ты
# опубликовал проект на GitHub) — иначе тихо продолжит с локальной.
CATALOG_REMOTE_URL = (
    "https://raw.githubusercontent.com/local-asr/LocalASR/main/resources/models_catalog.json"
)
CATALOG_CACHE_TTL_HOURS = 24

SETTINGS_FILENAME = "settings.json"
ENV_STAMP_FILENAME = ".localasr-env.json"

# --- медиа ----------------------------------------------------------------- #
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma",
    ".aiff", ".aif", ".amr", ".alac", ".caf", ".mka", ".ac3", ".eac3", ".mp2",
    ".m4b", ".ape", ".wv", ".tta", ".spx", ".3gp",
}
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv", ".m4v", ".mpg",
    ".mpeg", ".ts", ".m2ts", ".mts", ".vob", ".3g2", ".ogv", ".rm", ".rmvb",
    ".divx", ".f4v", ".asf", ".mxf",
}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Расширения файлов, которые уже являются текстом/субтитрами (можно игнорировать)
IGNORED_EXTENSIONS = {".txt", ".srt", ".vtt", ".nfo", ".jpg", ".png"}

SAMPLE_RATE = 16_000  # Whisper/Vosk работают с 16 кГц моно

# --- форматы результата ---------------------------------------------------- #
OUTPUT_FORMATS = {
    "txt": "Текст (.txt)",
    "txt-time": "Текст с таймкодами (.txt)",
    "srt": "Субтитры (.srt)",
    "vtt": "Субтитры (.vtt)",
    "json": "JSON (сегменты + слова)",
    "md": "Markdown (.md)",
}
DEFAULT_OUTPUT_FORMAT = "txt"

# --- движки ---------------------------------------------------------------- #
ENGINE_FASTER_WHISPER = "faster-whisper"
ENGINE_WHISPERCPP = "whispercpp"
ENGINE_VOSK = "vosk"
ENGINE_QWEN_ASR = "qwen-asr"          # Qwen3-ASR (Alibaba) через transformers + torch
ENGINE_GIGAAM = "gigaam"              # GigaAM (Sber) через transformers + torch
ENGINE_HF = "hf-transformers"         # ЛЮБАЯ ASR-модель Hugging Face через transformers.pipeline
ENGINES = (
    ENGINE_FASTER_WHISPER,
    ENGINE_WHISPERCPP,
    ENGINE_VOSK,
    ENGINE_QWEN_ASR,
    ENGINE_GIGAAM,
    ENGINE_HF,
)

#: движки на PyTorch — для них имеет смысл ставить CUDA-сборку torch
TORCH_ENGINE_NAMES = (ENGINE_QWEN_ASR, ENGINE_GIGAAM, ENGINE_HF)

#: индекс CUDA-сборок PyTorch (Windows-pip ставит только CPU-вариант).
#: cu126 работает с драйверами NVIDIA >= 525/527 (minor version compatibility)
TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu126"
TORCH_CUDA_MIN_DRIVER_MAJOR = 527

ENGINE_LABELS = {
    ENGINE_FASTER_WHISPER: "faster-whisper · CTranslate2",
    ENGINE_WHISPERCPP: "whisper.cpp · GGML",
    ENGINE_VOSK: "Vosk · Kaldi",
    ENGINE_QWEN_ASR: "Qwen3-ASR · transformers",
    ENGINE_GIGAAM: "GigaAM (Sber) · transformers",
    ENGINE_HF: "Transformers · любая ASR с Hugging Face",
}

# --- язык ------------------------------------------------------------------ #
DEFAULT_LANGUAGE = "ru"
LANGUAGE_AUTODETECT = ""  # пустая строка = «определить автоматически»

# Популярные языки для выпадающего списка (код, название по-русски)
LANGUAGES = [
    ("", "Определить автоматически"),
    ("ru", "Русский"),
    ("uk", "Украинский"),
    ("be", "Белорусский"),
    ("kk", "Казахский"),
    ("en", "Английский"),
    ("de", "Немецкий"),
    ("fr", "Французский"),
    ("es", "Испанский"),
    ("it", "Итальянский"),
    ("pl", "Польский"),
    ("cs", "Чешский"),
    ("tr", "Турецкий"),
    ("zh", "Китайский"),
    ("ja", "Японский"),
    ("ko", "Корейский"),
    ("ar", "Арабский"),
    ("pt", "Португальский"),
    ("nl", "Нидерландский"),
    ("sv", "Шведский"),
    ("fi", "Финский"),
    ("he", "Иврит"),
    ("hi", "Хинди"),
    ("hy", "Армянский"),
    ("ka", "Грузинский"),
    ("az", "Азербайджанский"),
    ("uz", "Узбекский"),
    ("ky", "Киргизский"),
    ("tg", "Таджикский"),
    ("mn", "Монгольский"),
    ("lv", "Латышский"),
    ("lt", "Литовский"),
    ("et", "Эстонский"),
    ("ro", "Румынский"),
    ("bg", "Болгарский"),
    ("sr", "Сербский"),
    ("hr", "Хорватский"),
    ("sl", "Словенский"),
    ("sk", "Словацкий"),
    ("hu", "Венгерский"),
    ("el", "Греческий"),
    ("no", "Норвежский"),
    ("da", "Датский"),
    ("is", "Исландский"),
    ("id", "Индонезийский"),
    ("vi", "Вьетнамский"),
    ("th", "Тайский"),
    ("fa", "Персидский"),
]

# --- производительность ---------------------------------------------------- #
DEFAULT_CPU_THREADS = 0          # 0 = «сколько нужно» (решает CTranslate2/система)
DEFAULT_BATCH_SIZE = 8           # для batched-режима на GPU
MAX_PREFETCH = 2                 # сколько файлов декодируем в фоне заранее
DECODER_TIMEOUT_PER_HOUR = 600   # сек на каждый час аудио (защита от зависшего ffmpeg)

# --- HTTP ------------------------------------------------------------------ #
HTTP_TIMEOUT = 30
HTTP_CHUNK = 1024 * 512
USER_AGENT = f"{APP_NAME}/{APP_VERSION} (+https://github.com/local-asr/LocalASR)"
HF_API_BASE = "https://huggingface.co/api/models"
HF_RESOLVE_BASE = "https://huggingface.co"
