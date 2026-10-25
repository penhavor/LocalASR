"""Model catalog: what can be downloaded, sizes, languages, recommendations.

The catalog lives in resources/models_catalog.json and works offline. When
online, a fresh copy is fetched at most once a day — optional, everything
essential is in the local file.
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
    MODELS_DIRNAME,
)
from ..i18n import _
from ..logutils import get_logger

log = get_logger("catalog")

META_FILENAME = "installed.json"


# --------------------------------------------------------------------------- #
#  Catalog entry
# --------------------------------------------------------------------------- #


@dataclass
class ModelEntry:
    id: str                                   # unique id = folder name in models/
    name: str                                 # human-readable name
    engine: str = ENGINE_FASTER_WHISPER
    source: str = ""                          # HF repo, direct URL or local path
    subfolder: str = ""                       # subfolder inside the HF repo (ct2_int8_float16 …)
    revision: str = "main"                    # HF branch/tag/commit
    file: str = ""                            # single-file models (ggml-*.bin, *.zip)
    size_bytes: Optional[int] = None          # download size
    ram_mb: int = 0                           # runtime RAM usage
    vram_mb: int = 0                          # VRAM usage (0 = CPU only)
    languages: List[str] = field(default_factory=list)
    languages_note: str = ""
    quality: int = 3                          # 1..5 accuracy
    speed: int = 3                            # 1..5 speed (5 = fastest)
    tags: List[str] = field(default_factory=list)
    license: str = ""
    description: str = ""
    vendor: str = ""                          # model author (OpenAI, Qwen, Sber…)
    params_b: Optional[float] = None          # parameter count in billions (None = unknown)
    quantization: str = ""                    # weight format: int8 / float16 / q5_0 / …
    released_year: int = 0                    # weight release year (0 = unknown)
    recommended_for: str = ""                 # short "best for" note
    defaults: Dict[str, Any] = field(default_factory=dict)
    url: str = ""                             # model page ("Open" button)

    # --- runtime state (filled by scanning, not part of the catalog) -------- #
    installed: bool = False
    install_dir: Optional[Path] = None
    installed_size_bytes: int = 0
    custom: bool = False                      # added by the user via link
    local_only: bool = False                  # a folder in models/ absent from the catalog

    # ------------------------------------------------------------------ #
    @property
    def display_name(self) -> str:
        """Localized name for display (catalog names are English msgids)."""
        return _(self.name)

    @property
    def engine_label(self) -> str:
        return ENGINE_LABELS.get(self.engine, self.engine)

    @property
    def size_text(self) -> str:
        return paths.human_size(self.size_bytes if self.size_bytes else self.installed_size_bytes)

    @property
    def languages_text(self) -> str:
        if self.languages_note:
            return _(self.languages_note)
        if not self.languages:
            return "—"
        codes = [str(lang).upper() for lang in self.languages]
        if len(codes) > 6:
            return ", ".join(codes[:5]) + _(" + {} more").format(len(codes) - 5)
        return ", ".join(codes)

    @property
    def supports_russian(self) -> bool:
        return "ru" in [str(lang).lower() for lang in self.languages]

    @property
    def memory_text(self) -> str:
        parts = []
        if self.ram_mb:
            parts.append(_("RAM ≈ {gb:g} {unit}").format(
                gb=round(self.ram_mb / 1024, 1), unit=_("GB")))
        if self.vram_mb:
            parts.append(_("VRAM ≈ {gb:g} {unit}").format(
                gb=round(self.vram_mb / 1024, 1), unit=_("GB")))
        return " · ".join(parts) or "—"

    @property
    def params_text(self) -> str:
        """Human-readable parameter count: "809 M", "1.55 B"."""
        if not self.params_b:
            return "—"
        if self.params_b >= 1:
            return _("{:g} B").format(self.params_b)
        return _("{} M").format(int(round(self.params_b * 1000)))

    @property
    def quantization_text(self) -> str:
        return self.quantization or "—"

    @property
    def released_year_text(self) -> str:
        return str(self.released_year) if self.released_year else "—"

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
        try:
            payload["released_year"] = int(payload.get("released_year") or 0)
        except (TypeError, ValueError):
            payload["released_year"] = 0
        entry = cls(**payload)
        if entry.install_dir:
            entry.install_dir = Path(str(entry.install_dir))
        return entry


# --------------------------------------------------------------------------- #
#  Engine detection by content
# --------------------------------------------------------------------------- #

#: model_type from config.json -> LocalASR engine
_CONFIG_TYPE_TO_ENGINE = {
    "gigaam": ENGINE_GIGAAM,
    "qwen3_asr": ENGINE_QWEN_ASR,
    # the rest of the Hugging Face ASR family goes to the generic engine
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

#: audio-model markers in architectures (config.json without a known model_type)
_AUDIO_ARCH_MARKER = (
    "speech", "asr", "wav2vec", "whisper", "hubert", "conformer", "parakeet",
    "canary", "seamless", "mms", "audio", "ctc", "rnnt", "transducer",
)


def engine_from_config(cfg: Any) -> str:
    """Engine from a config.json dict (transformers models). "" = not recognized."""
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
        # multimodal models with an audio input (qwen2-audio etc.)
        return ENGINE_HF
    return ""


def detect_engine(folder: Path) -> str:
    """Detect the engine by the files in a model folder (or a single file).

    Order matters: CT2 Whisper builds also contain config.json, so their
    marker (`model.bin`) is checked first.
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

    # 1) CTranslate2 (faster-whisper): model.bin may live in a subfolder
    if "model.bin" in names:
        return ENGINE_FASTER_WHISPER
    for child in folder.iterdir():
        if child.is_dir() and (child / "model.bin").is_file():
            return ENGINE_FASTER_WHISPER

    # 2) transformers models: GigaAM, Qwen3-ASR and any other HF ASR
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
    """If the CT2 model lives in a subfolder — return its name."""
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
#  Catalog
# --------------------------------------------------------------------------- #


class ModelCatalog:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else paths.ROOT_DIR
        self.models_root = self.root / MODELS_DIRNAME
        self.entries: Dict[str, ModelEntry] = {}
        self.loaded_from = "built-in"
        self.load()

    # -- loading ---------------------------------------------------------- #
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
                        self.loaded_from = f"network cache ({age_hours:.1f} h old)"
            except (OSError, ValueError) as exc:
                log.debug("Catalog cache unreadable: %s", exc)

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
                    log.warning("Failed to read the catalog %s: %s", local, exc)

        if data is None:
            data = {"models": []}
            self.loaded_from = "catalog not found (empty list)"

        self.entries = {}
        for raw in data.get("models", []) or []:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            try:
                entry = ModelEntry.from_dict(raw)
            except TypeError as exc:
                log.warning("Skipping a broken catalog entry %s: %s", raw.get("id"), exc)
                continue
            entry.installed = False
            entry.install_dir = None
            self.entries[entry.id] = entry

        self.refresh_installed()
        log.info(_("Catalog: %d models (%s)"), len(self.entries), self.loaded_from)

    def refresh_remote(self, force: bool = False) -> bool:
        """Try to refresh the catalog from the network. True on success."""
        try:
            from . import net  # local import: requests is not always needed
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
            log.info(_("Catalog refreshed from the network: %d models"), len(self.entries))
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("Remote catalog refresh unavailable: %s", exc)
            return False

    # -- scanning models/ --------------------------------------------------- #
    def refresh_installed(self) -> None:
        """Mark installed models and pick up folders without metadata."""
        # reset state: catalog entries are cleared, "local" ones removed
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
                    log.warning("Broken %s in %s: %s", META_FILENAME, child.name, exc)
                # an install finished but the marker removal failed — clean it
                # up so neither the catalog nor the user gets confused
                try:
                    (child / ".localasr-partial.json").unlink(missing_ok=True)
                except OSError:
                    pass

            # incomplete install (network drop/cancel): the marker exists but
            # installed.json does not — such a folder is NOT installed,
            # otherwise the GUI would offer a "half model" with missing files
            if not meta and (child / ".localasr-partial.json").is_file():
                log.warning(
                    _("Skipping incompletely downloaded model %s: the install did not "
                      "finish (reinstalling will resume the missing files)"),
                    child.name,
                )
                continue

            engine = str(meta.get("engine") or detect_engine(child))
            if not engine:
                log.debug("Folder %s does not look like a model — skipping", child.name)
                continue  # the folder does not look like a model — ignore

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
                    tags=list(meta.get("tags") or (["local model"] if not meta else [])),
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

    # -- selections --------------------------------------------------------- #
    def all(self) -> List[ModelEntry]:
        return list(self.entries.values())

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self.entries.get(model_id)

    def installed(self) -> List[ModelEntry]:
        return [e for e in self.entries.values() if e.installed]

    def sorted_for_display(self, entries: Optional[Iterable[ModelEntry]] = None) -> List[ModelEntry]:
        items = list(entries if entries is not None else self.entries.values())

        def key(entry: ModelEntry) -> tuple:
            return (
                0 if entry.installed else 1,
                0 if entry.supports_russian else 1,
                -entry.quality,
                entry.ram_mb or 99_999,
                entry.name.lower(),
            )

        return sorted(items, key=key)

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

