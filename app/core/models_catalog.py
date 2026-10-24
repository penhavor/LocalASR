"""Каталог моделей: что можно скачать, сколько весит, какие языки, что советовать.

Каталог лежит в `resources/models_catalog.json` и работает офлайн. При наличии
сети приложение раз в сутки пытается подтянуть свежую версию, но это не
обязательное условие работы — всё самое важное есть в локальном файле.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .. import paths
from ..constants import (
    CATALOG_CACHE_TTL_HOURS,
    CATALOG_FILENAME,
    CATALOG_REMOTE_URL,
    ENGINE_FASTER_WHISPER,
    ENGINE_GIGAAM,
    ENGINE_HF,
    ENGINE_LABELS,
    ENGINE_QWEN_ASR,
    ENGINE_VOSK,
    ENGINE_WHISPERCPP,
    ENGINES,
    MODELS_DIRNAME,
)
from ..logutils import get_logger

log = get_logger("catalog")

META_FILENAME = "installed.json"


# --------------------------------------------------------------------------- #
#  Запись каталога
# --------------------------------------------------------------------------- #


@dataclass
class ModelEntry:
    id: str                                   # уникальный идентификатор = имя папки в models/
    name: str                                 # человекочитаемое название
    engine: str = ENGINE_FASTER_WHISPER
    source: str = ""                          # HF-репозиторий, прямой URL или локальный путь
    subfolder: str = ""                       # подпапка внутри HF-репозитория (ct2_int8_float16 …)
    revision: str = "main"                    # ветка/тег/коммит HF
    file: str = ""                            # если модель — один файл (ggml-*.bin, *.zip)
    size_bytes: Optional[int] = None          # размер при скачивании
    ram_mb: int = 0                           # потребление памяти в рантайме
    vram_mb: int = 0                          # потребление VRAM (0 = только CPU)
    languages: List[str] = field(default_factory=list)
    languages_note: str = ""
    quality: int = 3                          # 1..5 — точность
    speed: int = 3                            # 1..5 — скорость (5 = быстрее всех)
    tags: List[str] = field(default_factory=list)
    license: str = ""
    description: str = ""
    vendor: str = ""                          # кто сделал модель (OpenAI, Qwen, Sber…)
    recommended: bool = False
    recommended_for: str = ""
    defaults: Dict[str, Any] = field(default_factory=dict)
    url: str = ""                             # страница модели (кнопка «Открыть»)
    released: str = ""

    # --- состояние (не относится к каталогу, заполняется при сканировании) -- #
    installed: bool = False
    install_dir: Optional[Path] = None
    installed_size_bytes: int = 0
    custom: bool = False                      # добавлена пользователем по ссылке
    local_only: bool = False                  # просто папка в models/, в каталоге её нет

    # ------------------------------------------------------------------ #
    @property
    def engine_label(self) -> str:
        return ENGINE_LABELS.get(self.engine, self.engine)

    @property
    def size_text(self) -> str:
        return paths.human_size(self.size_bytes if self.size_bytes else self.installed_size_bytes)

    @property
    def languages_text(self) -> str:
        if self.languages_note:
            return self.languages_note
        if not self.languages:
            return "—"
        codes = [str(lang).upper() for lang in self.languages]
        if len(codes) > 6:
            return ", ".join(codes[:5]) + f" + ещё {len(codes) - 5}"
        return ", ".join(codes)

    @property
    def supports_russian(self) -> bool:
        return "ru" in [str(lang).lower() for lang in self.languages]

    @property
    def memory_text(self) -> str:
        parts = []
        if self.ram_mb:
            parts.append(f"RAM ≈ {self.ram_mb / 1024:.1f} ГБ".replace(".0 ГБ", " ГБ"))
        if self.vram_mb:
            parts.append(f"VRAM ≈ {self.vram_mb / 1024:.1f} ГБ".replace(".0 ГБ", " ГБ"))
        return " · ".join(parts) or "—"

    def fits_ram(self, available_mb: int) -> bool:
        if not self.ram_mb:
            return True
        return available_mb == 0 or self.ram_mb <= available_mb

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["install_dir"] = str(self.install_dir) if self.install_dir else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        entry = cls(**payload)
        if entry.install_dir:
            entry.install_dir = Path(str(entry.install_dir))
        return entry


# --------------------------------------------------------------------------- #
#  Определение движка по содержимому
# --------------------------------------------------------------------------- #

#: model_type из config.json -> движок LocalASR
_CONFIG_TYPE_TO_ENGINE = {
    "gigaam": ENGINE_GIGAAM,
    "qwen3_asr": ENGINE_QWEN_ASR,
    # всё остальное ASR-семейство Hugging Face уходит в универсальный движок
    "whisper": ENGINE_HF,
    "wav2vec2": ENGINE_HF,
    "wav2vec2-bert": ENGINE_HF,
    "wav2vec2_conformer": ENGINE_HF,
    "hubert": ENGINE_HF,
    "unispeech": ENGINE_HF,
    "unispeech-sat": ENGINE_HF,
    "data2vec-audio": ENGINE_HF,
    "speecht5": ENGINE_HF,
    "seamless-m4t": ENGINE_HF,
    "seamless-m4t-v2": ENGINE_HF,
    "mms": ENGINE_HF,
    "vits": ENGINE_HF,
    "parakeet": ENGINE_HF,
    "canary": ENGINE_HF,
    "distil-whisper": ENGINE_HF,
}

#: признаки аудио-модели в architectures (для config.json без известного model_type)
_AUDIO_ARCH_MARKER = (
    "speech", "asr", "wav2vec", "whisper", "hubert", "conformer", "parakeet",
    "canary", "seamless", "mms", "audio", "ctc", "rnnt", "transducer",
)


def engine_from_config(cfg: Any) -> str:
    """Движок по словарю config.json (transformers-модели). '' — не опознано."""
    if not isinstance(cfg, dict):
        return ""
    model_type = str(cfg.get("model_type") or "").lower()
    if model_type in _CONFIG_TYPE_TO_ENGINE:
        return _CONFIG_TYPE_TO_ENGINE[model_type]
    architectures = cfg.get("architectures") or []
    joined = " ".join(str(a) for a in architectures).lower()
    if any(marker in joined for marker in _AUDIO_ARCH_MARKER):
        return ENGINE_HF
    if isinstance(cfg.get("audio_config"), dict):
        # мультимодальные модели со звуковым входом (qwen2-audio и т.п.)
        return ENGINE_HF
    return ""


def detect_engine(folder: Path) -> str:
    """Определяет движок по файлам в папке модели (или по одному файлу).

    Порядок важен: CT2-сборки Whisper тоже содержат config.json, поэтому
    сначала ищем их признак — `model.bin`.
    """
    folder = Path(folder)
    if not folder.exists():
        return ""

    if folder.is_file():
        name = folder.name.lower()
        if name.endswith(".zip"):
            return ENGINE_VOSK
        if name.startswith("ggml"):
            return ENGINE_WHISPERCPP
        return ""

    try:
        names = [p.name.lower() for p in folder.iterdir()]
    except OSError:
        return ""

    # 1) CTranslate2 (faster-whisper): model.bin может лежать и в подпапке
    if "model.bin" in names:
        return ENGINE_FASTER_WHISPER
    for child in folder.iterdir():
        if child.is_dir() and (child / "model.bin").is_file():
            return ENGINE_FASTER_WHISPER

    # 2) transformers-модели: GigaAM, Qwen3-ASR и любые прочие ASR с HF
    config = folder / "config.json"
    has_weights = any(
        n.endswith((".safetensors", ".bin", ".pt", ".pth")) and n != "model.bin"
        for n in names
    )
    if config.is_file() and has_weights:
        try:
            with open(config, "r", encoding="utf-8") as fh:
                engine = engine_from_config(json.load(fh))
            if engine:
                return engine
        except (OSError, ValueError):
            pass

    # 3) Vosk (Kaldi)
    if "am" in names or any(n == "final.mdl" or n.endswith(".mdl") for n in names):
        return ENGINE_VOSK

    # 4) whisper.cpp (GGML)
    if any(n.startswith("ggml") for n in names):
        return ENGINE_WHISPERCPP

    return ""


def ct2_subfolder(folder: Path) -> str:
    """Если CT2-модель лежит в подпапке — вернуть её имя."""
    folder = Path(folder)
    if (folder / "model.bin").is_file():
        return ""
    try:
        for child in sorted(folder.iterdir()):
            if child.is_dir() and (child / "model.bin").is_file():
                return child.name
    except OSError:
        pass
    return ""


# --------------------------------------------------------------------------- #
#  Каталог
# --------------------------------------------------------------------------- #


class ModelCatalog:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else paths.ROOT_DIR
        self.models_root = self.root / MODELS_DIRNAME
        self.entries: Dict[str, ModelEntry] = {}
        self.loaded_from = "built-in"
        self.load()

    # -- загрузка -------------------------------------------------------- #
    def catalog_file(self) -> Optional[Path]:
        return paths.resource_path(CATALOG_FILENAME, self.root)

    def cache_file(self) -> Path:
        return self.root / "settings" / "catalog_cache.json"

    def load(self) -> None:
        data: Optional[Dict[str, Any]] = None

        cache = self.cache_file()
        if cache.is_file():
            try:
                age_hours = (time.time() - cache.stat().st_mtime) / 3600
                if age_hours < CATALOG_CACHE_TTL_HOURS:
                    with open(cache, "r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    if isinstance(payload, dict) and payload.get("models"):
                        data = payload
                        self.loaded_from = f"кэш сети ({age_hours:.1f} ч назад)"
            except (OSError, ValueError) as exc:
                log.debug("Кэш каталога не читается: %s", exc)

        if data is None:
            local = self.catalog_file()
            if local and local.is_file():
                try:
                    with open(local, "r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    if isinstance(payload, dict):
                        data = payload
                        self.loaded_from = str(local)
                except (OSError, ValueError) as exc:
                    log.warning("Не удалось прочитать каталог %s: %s", local, exc)

        if data is None:
            data = {"models": []}
            self.loaded_from = "каталог не найден (пустой список)"

        self.entries = {}
        for raw in data.get("models", []) or []:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            try:
                entry = ModelEntry.from_dict(raw)
            except TypeError as exc:
                log.warning("Пропускаю битую запись каталога %s: %s", raw.get("id"), exc)
                continue
            entry.installed = False
            entry.install_dir = None
            self.entries[entry.id] = entry

        self.refresh_installed()
        log.info("Каталог: %d моделей (%s)", len(self.entries), self.loaded_from)

    def refresh_remote(self, force: bool = False) -> bool:
        """Пробует обновить каталог из сети. True — если получилось."""
        try:
            from . import net  # локальный импорт: requests нужен не всегда
        except Exception:  # noqa: BLE001
            return False

        cache = self.cache_file()
        if not force and cache.is_file():
            try:
                age_hours = (time.time() - cache.stat().st_mtime) / 3600
                if age_hours < CATALOG_CACHE_TTL_HOURS:
                    return False
            except OSError:
                pass
        try:
            text = net.fetch_text(CATALOG_REMOTE_URL, timeout=10)
            payload = json.loads(text)
            if not isinstance(payload, dict) or not payload.get("models"):
                return False
            cache.parent.mkdir(parents=True, exist_ok=True)
            paths.atomic_write_text(cache, json.dumps(payload, ensure_ascii=False))
            self.load()
            log.info("Каталог обновлён из сети: %d моделей", len(self.entries))
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("Обновление каталога из сети недоступно: %s", exc)
            return False

    # -- сканирование models/ -------------------------------------------- #
    def refresh_installed(self) -> None:
        """Помечает установленные модели и подхватывает папки без метаданных."""
        # чистим состояние: catalog-записи сбрасываем, «локальные» удаляем
        for entry in self.entries.values():
            entry.installed = False
            entry.install_dir = None
            entry.installed_size_bytes = 0
        self.entries = {k: v for k, v in self.entries.items() if not v.local_only}

        if not self.models_root.is_dir():
            return

        for child in sorted(self.models_root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            meta_file = child / META_FILENAME
            meta: Dict[str, Any] = {}
            if meta_file.is_file():
                try:
                    with open(meta_file, "r", encoding="utf-8") as fh:
                        loaded = json.load(fh)
                    if isinstance(loaded, dict):
                        meta = loaded
                except (OSError, ValueError) as exc:
                    log.warning("Битый %s в %s: %s", META_FILENAME, child.name, exc)

            engine = str(meta.get("engine") or detect_engine(child))
            if not engine:
                continue  # папка не похожа на модель — игнорируем

            model_id = str(meta.get("id") or child.name)
            entry = self.entries.get(model_id)
            if entry is None:
                entry = ModelEntry(
                    id=model_id,
                    name=str(meta.get("name") or child.name),
                    engine=engine,
                    source=str(meta.get("source") or str(child)),
                    size_bytes=meta.get("size_bytes"),
                    ram_mb=int(meta.get("ram_mb") or 0),
                    languages=list(meta.get("languages") or []),
                    description=str(meta.get("description") or ""),
                    url=str(meta.get("url") or ""),
                    custom=bool(meta.get("custom", not meta)),
                    local_only=not meta,
                    tags=list(meta.get("tags") or (["локальная"] if not meta else [])),
                )
                self.entries[model_id] = entry

            entry.installed = True
            entry.install_dir = child
            entry.engine = engine
            entry.installed_size_bytes = int(
                meta.get("installed_size_bytes") or paths.dir_size(child)
            )
            if not entry.size_bytes:
                entry.size_bytes = entry.installed_size_bytes

    # -- выборки ---------------------------------------------------------- #
    def all(self) -> List[ModelEntry]:
        return list(self.entries.values())

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self.entries.get(model_id)

    def installed(self) -> List[ModelEntry]:
        return [e for e in self.entries.values() if e.installed]

    def not_installed(self) -> List[ModelEntry]:
        return [e for e in self.entries.values() if not e.installed]

    def filter(
        self,
        engine: str = "",
        language: str = "",
        installed_only: bool = False,
        available_only: bool = False,
        max_size_bytes: Optional[int] = None,
        max_ram_mb: Optional[int] = None,
        russian_only: bool = False,
    ) -> List[ModelEntry]:
        result: List[ModelEntry] = []
        for entry in self.entries.values():
            if engine and entry.engine != engine:
                continue
            if language and language.lower() not in [str(l).lower() for l in entry.languages]:
                continue
            if russian_only and not entry.supports_russian:
                continue
            if installed_only and not entry.installed:
                continue
            if available_only and entry.installed:
                continue
            if max_size_bytes and entry.size_bytes and entry.size_bytes > max_size_bytes:
                continue
            if max_ram_mb and entry.ram_mb and entry.ram_mb > max_ram_mb:
                continue
            result.append(entry)
        return result

    def sorted_for_display(self, entries: Optional[Iterable[ModelEntry]] = None) -> List[ModelEntry]:
        items = list(entries if entries is not None else self.entries.values())

        def key(entry: ModelEntry) -> tuple:
            return (
                0 if entry.installed else 1,
                0 if entry.recommended else 1,
                0 if entry.supports_russian else 1,
                -entry.quality,
                entry.ram_mb or 99_999,
                entry.name.lower(),
            )

        return sorted(items, key=key)

    def recommendation(self, hardware: Any = None) -> Optional[ModelEntry]:
        """Главная рекомендация под железо (приоритет — русский язык)."""
        candidates = [e for e in self.entries.values() if not e.installed] or self.all()
        if not candidates:
            return None

        limit_bytes: Optional[int] = None
        if hardware is not None:
            try:
                from .sysinfo import max_model_size_mb

                limit_bytes = max_model_size_mb(hardware) * 1024 * 1024
            except Exception as exc:  # noqa: BLE001
                # молча «проглотить» ошибку здесь = дать невыполнимый совет,
                # поэтому обязательно пишем в лог
                log.warning("Не удалось оценить лимит памяти под модель: %s", exc)
                limit_bytes = None

        def score(entry: ModelEntry) -> tuple:
            size = entry.size_bytes or 0
            fits = True if limit_bytes is None else size <= limit_bytes
            return (
                0 if fits else 1,
                0 if entry.recommended else 1,
                0 if entry.supports_russian else 1,
                -(entry.quality * 2 + entry.speed),
                size,
            )

        ranked = sorted(candidates, key=score)
        best = ranked[0]

        # если под доступную память не влезает ничего — берём самое лёгкое
        # (и в первую очередь русскоязычное), иначе совет будет невыполним
        if limit_bytes is not None and (best.size_bytes or 0) > limit_bytes:
            light = [e for e in ranked if e.supports_russian] or ranked
            best = min(light, key=lambda e: (e.size_bytes or 0, -e.quality))
        return best

    def add_custom_entry(self, entry: ModelEntry) -> None:
        self.entries[entry.id] = entry

    def export_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "models": [e.to_dict() for e in self.sorted_for_display()],
            },
            ensure_ascii=False,
            indent=2,
        )


def available_engines() -> List[str]:
    return list(ENGINES)
