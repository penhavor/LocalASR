"""Общие типы движков распознавания.

Движок = «чем считаем»: faster-whisper (CTranslate2), whisper.cpp, Vosk.
Все они приводят результат к единому формату `Segment` из postprocess, поэтому
GUI, экспорт и постобработка от движка не зависят.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..constants import (
    ENGINE_FASTER_WHISPER,
    ENGINE_GIGAAM,
    ENGINE_HF,
    ENGINE_QWEN_ASR,
    ENGINE_VOSK,
    ENGINE_WHISPERCPP,
)
from ..core.postprocess import Segment

CancelToken = Callable[[], bool]
ProgressCallback = Callable[[float, str], None]      # (0..1, текст)
SegmentCallback = Callable[[Segment], None]


class EngineError(RuntimeError):
    """Ошибка внутри движка (не удалось загрузить модель, сбой инференса…)."""


class EngineNotInstalled(RuntimeError):
    """Движок не установлен — GUI предложит доустановить пакеты в .venv."""

    def __init__(self, engine: str, packages: List[str], note: str = "") -> None:
        self.engine = engine
        self.packages = packages
        self.note = note
        super().__init__(
            f"Движок «{engine}» не установлен. Нужно поставить: {', '.join(packages) or '—'}."
        )


@dataclass
class ModelRef:
    """Ссылка на установленную модель (всё, что нужно движку)."""

    model_id: str
    name: str
    engine: str
    path: Path                      # папка или файл модели
    subfolder: str = ""             # для CT2-моделей внутри подпапки
    languages: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_path(self) -> Path:
        if self.subfolder:
            candidate = self.path / self.subfolder
            if candidate.is_dir():
                return candidate
        if self.path.is_dir() and not (self.path / "model.bin").is_file():
            for child in sorted(self.path.iterdir()):
                if child.is_dir() and (child / "model.bin").is_file():
                    return child
        return self.path


@dataclass
class TranscriptResult:
    segments: List[Segment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    audio_duration: float = 0.0
    processed_duration: float = 0.0
    elapsed_seconds: float = 0.0
    engine: str = ""
    device: str = ""
    compute_type: str = ""
    model_name: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def speed_x(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.processed_duration / self.elapsed_seconds

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments if segment.text)


# --------------------------------------------------------------------------- #
#  Базовый класс движка
# --------------------------------------------------------------------------- #


class ASREngine:
    name: str = "base"
    packages: List[str] = []
    import_name: str = ""

    def __init__(self) -> None:
        self.loaded_model_id: Optional[str] = None
        self._model: Any = None
        self.device: str = ""
        self.compute_type: str = ""

    # -- обязательные методы --------------------------------------------- #
    @classmethod
    def is_available(cls) -> bool:
        if not cls.import_name:
            return True
        return importlib.util.find_spec(cls.import_name) is not None

    @classmethod
    def ensure_available(cls) -> None:
        if not cls.is_available():
            raise EngineNotInstalled(cls.name, list(cls.packages))

    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        raise NotImplementedError

    def transcribe(
        self,
        audio: "Any",
        *,
        options: Dict[str, Any],
        duration: float,
        progress: Optional[ProgressCallback] = None,
        on_segment: Optional[SegmentCallback] = None,
        cancel: Optional[CancelToken] = None,
    ) -> TranscriptResult:
        raise NotImplementedError

    def unload(self) -> None:
        self._model = None
        self.loaded_model_id = None

    # -- вспомогательное -------------------------------------------------- #
    @property
    def model(self) -> Any:
        if self._model is None:
            raise EngineError(f"Модель движка {self.name} не загружена")
        return self._model

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} model={self.loaded_model_id} device={self.device}>"


def engine_packages(engine: str) -> List[str]:
    """pip-пакеты, которые нужны движку (читаем requirements/engines.json)."""
    from .. import paths
    import json

    config_path = paths.resource_path("engines.json")
    if config_path is None:
        candidate = paths.ROOT_DIR / "requirements" / "engines.json"
        config_path = candidate if candidate.is_file() else None
    if config_path is not None:
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return list((data.get(engine) or {}).get("pip") or [])
        except (OSError, ValueError):
            pass
    return {
        ENGINE_FASTER_WHISPER: [],
        ENGINE_WHISPERCPP: ["pywhispercpp>=1.4.0"],
        ENGINE_VOSK: ["vosk>=0.3.45"],
        ENGINE_QWEN_ASR: ["torch>=2.6", "transformers>=5.0", "accelerate>=1.0", "soundfile"],
        ENGINE_GIGAAM: [
            "torch>=2.6", "torchaudio>=2.6", "transformers>=4.52",
            "hydra-core>=1.3,<2", "omegaconf>=2.3,<3", "sentencepiece>=0.2.2",
        ],
        ENGINE_HF: ["torch>=2.6", "transformers>=4.49", "accelerate>=1.0", "soundfile>=0.12"],
    }.get(engine, [])


def engine_note(engine: str) -> str:
    """Человеческое пояснение про движок из requirements/engines.json."""
    import json

    from .. import paths

    config_path = paths.resource_path("engines.json")
    if config_path is None:
        candidate = paths.ROOT_DIR / "requirements" / "engines.json"
        config_path = candidate if candidate.is_file() else None
    if config_path is not None:
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return str((data.get(engine) or {}).get("note") or "")
        except (OSError, ValueError):
            pass
    return ""


