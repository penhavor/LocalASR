"""Настройки приложения: JSON в папке проекта, устойчив к битым файлам."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths
from .constants import DEFAULT_LANGUAGE, DEFAULT_OUTPUT_FORMAT, SETTINGS_FILENAME


@dataclass
class Settings:
    # --- файлы и результат ------------------------------------------------- #
    output_dir: str = ""                      # пусто = <проект>/outputs
    output_format: str = DEFAULT_OUTPUT_FORMAT
    overwrite_existing: bool = False
    open_output_folder_when_done: bool = True
    last_open_dir: str = ""

    # --- язык и качество --------------------------------------------------- #
    language: str = DEFAULT_LANGUAGE
    translate_to_english: bool = False
    initial_prompt: str = ""
    hotwords: str = ""                        # слова-подсказки (термины лекции)
    beam_size: int = 5
    use_vad: bool = True
    suppress_hallucinations: bool = True      # анти-«[музыка]»/«Спасибо за просмотр»
    add_timestamps_to_txt: bool = False
    word_timestamps: bool = False
    paragraphs: bool = True                   # склеивать сегменты в абзацы

    # --- движок и железо --------------------------------------------------- #
    device: str = "auto"                      # auto | cpu | cuda
    compute_type: str = "auto"                # auto | int8 | int8_float16 | float16 | default
    cpu_threads: int = 0                      # 0 = авто
    batched: bool = True                      # BatchedInferencePipeline (ускорение на GPU)
    batch_size: int = 8
    parallel_files: int = 1                   # >1 — несколько файлов одновременно (требует RAM)
    gpu_index: int = 0

    # --- модели ------------------------------------------------------------ #
    selected_model_id: str = ""
    last_used_model_id: str = ""
    catalog_filters: Dict[str, Any] = field(default_factory=lambda: {"engine": "", "hide_installed": False})

    # --- интерфейс --------------------------------------------------------- #
    window_geometry: str = ""
    window_state: str = ""
    splitter_sizes: List[int] = field(default_factory=list)
    show_log_panel: bool = True
    theme: str = "dark"
    first_run_done: bool = False
    gpu_boost_offered: bool = False       # предложение «ускорить GPU» показано один раз

    # ------------------------------------------------------------------ #
    @staticmethod
    def file_path(root: Optional[Path] = None) -> Path:
        base = Path(root) if root else paths.ROOT_DIR
        return base / "settings" / SETTINGS_FILENAME

    @classmethod
    def load(cls, root: Optional[Path] = None) -> "Settings":
        path = cls.file_path(root)
        data: Dict[str, Any] = {}
        try:
            if path.is_file():
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    data = loaded.get("settings", loaded)
        except (OSError, ValueError):
            # битый settings.json не должен ронять приложение
            try:
                path.replace(path.with_suffix(".json.broken"))
            except OSError:
                pass
            data = {}

        settings = cls()
        known = {f.name: f for f in fields(cls)}
        for key, value in data.items():
            if key not in known:
                continue
            try:
                current = getattr(settings, key)
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                elif isinstance(current, str):
                    value = str(value)
                elif isinstance(current, list) and not isinstance(value, list):
                    continue
                elif isinstance(current, dict) and not isinstance(value, dict):
                    continue
            except (TypeError, ValueError):
                continue
            object.__setattr__(settings, key, value)

        # нормализация
        settings.beam_size = max(1, min(12, int(settings.beam_size or 1)))
        settings.batch_size = max(1, min(64, int(settings.batch_size or 1)))
        settings.parallel_files = max(1, min(8, int(settings.parallel_files or 1)))
        settings.cpu_threads = max(0, min(128, int(settings.cpu_threads or 0)))
        if settings.device not in ("auto", "cpu", "cuda"):
            settings.device = "auto"
        if settings.output_format not in ("txt", "txt-time", "srt", "vtt", "json", "md"):
            settings.output_format = DEFAULT_OUTPUT_FORMAT
        return settings

    def save(self, root: Optional[Path] = None) -> None:
        path = self.file_path(root)
        payload = {"version": 1, "settings": asdict(self)}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError:
            pass  # не смогли сохранить — не смертельно

    # ------------------------------------------------------------------ #
    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            path = Path(self.output_dir).expanduser()
        else:
            path = paths.outputs_dir()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            path = paths.outputs_dir()
        return path

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(asdict(self))

    def update(self, values: Dict[str, Any]) -> None:
        known = {f.name for f in fields(self)}
        for key, value in values.items():
            if key in known:
                setattr(self, key, value)
