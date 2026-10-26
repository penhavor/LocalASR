"""Application constants: folder names, thresholds, format lists."""

from __future__ import annotations

APP_NAME = "LocalASR"
APP_VERSION = "1.1.0"

# --- project folders --------------------------------------------------------
MODELS_DIRNAME = "models"
OUTPUTS_DIRNAME = "outputs"
LOGS_DIRNAME = "logs"
RESOURCES_DIRNAME = "resources"
RUNTIME_DIRNAME = "runtime"  # portable Python, downloaded on first run (Windows)

CATALOG_FILENAME = "models_catalog.json"
# The local catalog is primary; a remote refresh is attempted when online.
REPO_URL = "https://github.com/penhavor/LocalASR"
REPO_RAW_BASE = "https://raw.githubusercontent.com/penhavor/LocalASR/main"
CATALOG_REMOTE_URL = f"{REPO_RAW_BASE}/resources/models_catalog.json"
CATALOG_CACHE_TTL_HOURS = 24

# --- self-update (app/core/updater.py) --------------------------------------
GITHUB_OWNER = "penhavor"
GITHUB_REPO = "LocalASR"
GITHUB_API_BASE = "https://api.github.com"
#: short timeout: the startup check must not hang on a flaky connection
UPDATE_HTTP_TIMEOUT = 10
CHANGELOG_FILENAME = "CHANGELOG.md"

SETTINGS_FILENAME = "settings.json"

# --- media ------------------------------------------------------------------
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

SAMPLE_RATE = 16_000  # Whisper/Vosk expect 16 kHz mono

# --- output -----------------------------------------------------------------
OUTPUT_FORMATS = ("txt", "txt-time", "srt", "vtt", "json", "md")
DEFAULT_OUTPUT_FORMAT = "txt"

# --- engines ----------------------------------------------------------------
ENGINE_FASTER_WHISPER = "faster-whisper"
ENGINE_WHISPERCPP = "whispercpp"
ENGINE_VOSK = "vosk"
ENGINE_QWEN_ASR = "qwen-asr"
ENGINE_GIGAAM = "gigaam"
ENGINE_HF = "hf-transformers"
ENGINES = (
    ENGINE_FASTER_WHISPER,
    ENGINE_WHISPERCPP,
    ENGINE_VOSK,
    ENGINE_QWEN_ASR,
    ENGINE_GIGAAM,
    ENGINE_HF,
)

#: PyTorch-based engines — CUDA builds of torch make sense for these
TORCH_ENGINE_NAMES = (ENGINE_QWEN_ASR, ENGINE_GIGAAM, ENGINE_HF)

#: PyTorch CUDA wheel index (Windows pip installs CPU builds by default).
#: cu126 works with NVIDIA drivers >= 525/527 (minor version compatibility).
TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu126"
TORCH_CUDA_MIN_DRIVER_MAJOR = 527

#: Display strings; translate with i18n._() at the point of use.
ENGINE_LABELS = {
    ENGINE_FASTER_WHISPER: "faster-whisper · CTranslate2",
    ENGINE_WHISPERCPP: "whisper.cpp · GGML",
    ENGINE_VOSK: "Vosk · Kaldi",
    ENGINE_QWEN_ASR: "Qwen3-ASR · transformers",
    ENGINE_GIGAAM: "GigaAM (Sber) · transformers",
    ENGINE_HF: "Transformers · any ASR model from Hugging Face",
}

# --- speech language ----------------------------------------------------------
DEFAULT_LANGUAGE = "ru"
LANGUAGE_AUTODETECT = ""  # empty string = detect automatically

#: Popular speech languages for the dropdown (code, English display name).
LANGUAGES = [
    ("", "Detect automatically"),
    ("ru", "Russian"),
    ("uk", "Ukrainian"),
    ("be", "Belarusian"),
    ("kk", "Kazakh"),
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("ar", "Arabic"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("fi", "Finnish"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hy", "Armenian"),
    ("ka", "Georgian"),
    ("az", "Azerbaijani"),
    ("uz", "Uzbek"),
    ("ky", "Kyrgyz"),
    ("tg", "Tajik"),
    ("mn", "Mongolian"),
    ("lv", "Latvian"),
    ("lt", "Lithuanian"),
    ("et", "Estonian"),
    ("ro", "Romanian"),
    ("bg", "Bulgarian"),
    ("sr", "Serbian"),
    ("hr", "Croatian"),
    ("sl", "Slovenian"),
    ("sk", "Slovak"),
    ("hu", "Hungarian"),
    ("el", "Greek"),
    ("no", "Norwegian"),
    ("da", "Danish"),
    ("is", "Icelandic"),
    ("id", "Indonesian"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
    ("fa", "Persian"),
]

# --- performance --------------------------------------------------------------
DEFAULT_BATCH_SIZE = 8           # batched mode on GPU
MAX_PREFETCH = 2                 # files decoded ahead of time in background
DECODER_TIMEOUT_PER_HOUR = 600   # seconds per hour of audio (stuck ffmpeg guard)

# --- HTTP -------------------------------------------------------------------
HTTP_TIMEOUT = 30
HTTP_CHUNK = 1024 * 512
USER_AGENT = f"{APP_NAME}/{APP_VERSION} (+{REPO_URL})"
HF_API_BASE = "https://huggingface.co/api/models"
HF_RESOLVE_BASE = "https://huggingface.co"
