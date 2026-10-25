"""Application settings: JSON in the project folder, robust to broken files."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths
from .constants import DEFAULT_LANGUAGE, DEFAULT_OUTPUT_FORMAT, SETTINGS_FILENAME
from .i18n import _
from .logutils import get_logger

log = get_logger("settings")


@dataclass
class Settings:
    # --- files and output ---------------------------------------------------
    output_dir: str = ""                      # empty = <project>/outputs
    output_format: str = DEFAULT_OUTPUT_FORMAT
    overwrite_existing: bool = False
    open_output_folder_when_done: bool = True
    last_open_dir: str = ""

    # --- speech language and quality -----------------------------------------
    language: str = DEFAULT_LANGUAGE
    translate_to_english: bool = False
    initial_prompt: str = ""
    hotwords: str = ""                        # hint words (lecture terms)
    beam_size: int = 5
    use_vad: bool = True
    suppress_hallucinations: bool = True      # anti "[music]"/"Thanks for watching"
    add_timestamps_to_txt: bool = False
    word_timestamps: bool = False
    paragraphs: bool = True                   # merge segments into paragraphs

    # --- engine and hardware --------------------------------------------------
    device: str = "auto"                      # auto | cpu | cuda
    compute_type: str = "auto"                # auto | int8 | int8_float16 | float16 | default
    cpu_threads: int = 0                      # 0 = auto
    batched: bool = True                      # BatchedInferencePipeline (GPU speedup)
    batch_size: int = 8
    parallel_files: int = 1                   # >1 = several files at once (needs RAM)
    gpu_index: int = 0

    # --- models ---------------------------------------------------------------
    selected_model_id: str = ""
    last_used_model_id: str = ""
    #: models panel filter/sort state (numeric ranges etc.)
    catalog_filters: Dict[str, Any] = field(default_factory=lambda: {
        "engine": "",
        "vendor": "",
        "sort": "default",
        "only_installed": False,
        "only_russian": True,
        "min_quality": 0,
        "size_mb_lo": 0,
        "size_mb_hi": 0,
        "ram_mb_lo": 0,
        "ram_mb_hi": 0,
        "year_lo": 0,
        "year_hi": 0,
        "params_lo": 0.0,
        "params_hi": 0.0,
        "collapsed": False,
    })

    # --- interface -------------------------------------------------------------
    ui_language: str = ""                 # "" = follow the system language
    window_geometry: str = ""
    window_state: str = ""
    splitter_sizes: List[int] = field(default_factory=list)
    show_log_panel: bool = True
    advanced_open: bool = False           # advanced options panel expanded
    theme: str = "dark"
    gpu_boost_offered: bool = False       # "boost with GPU" prompt shown once

    # --- diagnostics ------------------------------------------------------------
    #: verbose file log in logs/ (new file per launch). OFF by default; toggled
    #: in Settings -> Logging & diagnostics (applies immediately).
    file_logging: bool = False

    # Root the settings were loaded from (None = project root). No type
    # annotation on purpose: a plain class attribute, not a dataclass field.
    _root = None

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
            # a broken settings.json must not crash the app
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

        # normalization
        settings.beam_size = max(1, min(12, int(settings.beam_size or 1)))
        settings.batch_size = max(1, min(64, int(settings.batch_size or 1)))
        settings.parallel_files = max(1, min(8, int(settings.parallel_files or 1)))
        settings.cpu_threads = max(0, min(128, int(settings.cpu_threads or 0)))
        if settings.device not in ("auto", "cpu", "cuda"):
            settings.device = "auto"
        if settings.output_format not in ("txt", "txt-time", "srt", "vtt", "json", "md"):
            settings.output_format = DEFAULT_OUTPUT_FORMAT
        settings._root = Path(root) if root else None
        log.debug(
            _("Settings loaded from %s: output_dir=%r, format=%s, language=%s, model=%s, "
              "device=%s, file_logging=%s"),
            path, settings.output_dir, settings.output_format, settings.language or _("auto"),
            settings.selected_model_id or "—", settings.device, settings.file_logging,
        )
        return settings

    def save(self, root: Optional[Path] = None) -> None:
        path = self.file_path(root if root is not None else getattr(self, "_root", None))
        payload = {"version": 1, "settings": asdict(self)}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            log.debug(_("Settings saved: %s"), path)
        except OSError as exc:
            log.warning(_("Failed to save settings %s: %s"), path, exc)

    def resolved_output_dir(self) -> Path:
        """Where results are saved. Empty output_dir = project outputs/.

        If the saved folder is unavailable (disk gone, no permissions) fall
        back to the project outputs/ instead of writing to random places.
        """
        if self.output_dir:
            path = Path(self.output_dir).expanduser()
            try:
                path.mkdir(parents=True, exist_ok=True)
                return path
            except OSError as exc:
                fallback = paths.outputs_dir(self._root)
                log.warning(
                    _("Output folder %s is unavailable (%s) — using %s"), path, exc, fallback
                )
                return fallback
        return paths.outputs_dir(self._root)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(asdict(self))

    def update(self, values: Dict[str, Any]) -> None:
        known = {f.name for f in fields(self)}
        for key, value in values.items():
            if key in known:
                setattr(self, key, value)

    def reset_to_defaults(self) -> None:
        """Reset ALL settings to defaults (including window geometry)."""
        defaults = Settings()
        for f in fields(self):
            object.__setattr__(self, f.name, getattr(defaults, f.name))
        self.save()
