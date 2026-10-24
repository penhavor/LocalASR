"""Настройки приложения: JSON в папке проекта, устойчив к битым файлам."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths
from .constants import (
    DEFAULT_LANGUAGE,
    DEFAULT_OUTPUT_FORMAT,
    OUTPUTS_DIRNAME,
    SETTINGS_FILENAME,
)
from .logutils import get_logger

log = get_logger("settings")


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
    #: состояние фильтров/сортировки панели моделей (числовые диапазоны и пр.)
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
        "collapsed": False,       # свёрнута ли панель фильтров каталога
    })

    # --- интерфейс --------------------------------------------------------- #
    window_geometry: str = ""
    window_state: str = ""
    splitter_sizes: List[int] = field(default_factory=list)
    show_log_panel: bool = True
    advanced_open: bool = False           # раскрыта ли панель «Расширенные настройки»
    theme: str = "dark"
    gpu_boost_offered: bool = False       # предложение «ускорить GPU» показано один раз

    # --- диагностика --------------------------------------------------------- #
    #: подробный файл-лог в logs/ (новый файл на каждый запуск).
    #: По умолчанию ВЫКЛЮЧЕН, чтобы не засорять папку; включается в
    #: «Сервис → Настройки → Журнал и диагностика» (сразу, без перезапуска).
    file_logging: bool = False

    # ------------------------------------------------------------------ #
    # Корень, из которого настройки были загружены (None = корень проекта).
    # БЕЗ аннотации типа — это обычный атрибут класса, а не поле dataclass
    # (иначе попал бы в asdict()/fields() и сломал сериализацию).
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
        settings._root = Path(root) if root else None
        settings._migrate_stale_output_dir()
        log.debug(
            "Настройки загружены из %s: output_dir=%r, формат=%s, язык=%s, модель=%s, "
            "device=%s, file_logging=%s",
            path, settings.output_dir, settings.output_format, settings.language or "авто",
            settings.selected_model_id or "—", settings.device, settings.file_logging,
        )
        return settings

    # ------------------------------------------------------------------ #
    def _migrate_stale_output_dir(self) -> None:
        """Лечит «переехавший» проект: outputs/ от СТАРОГО расположения.

        Раньше значение по умолчанию («<проект>/outputs») сохранялось в
        settings.json абсолютным путём. Если папку проекта переносили
        (например, с рабочего стола в C:\\Projects), результаты продолжали
        уезжать по старому пути — да ещё и воссоздавали старую папку.
        Правило миграции: путь, который заканчивается именем штатной папки
        результатов (outputs) и НЕ лежит внутри текущего проекта, — это
        устаревший дефолт. Сбрасываем его на «по умолчанию» (outputs/
        текущего проекта). Намеренно выбранная пользователем папка с другим
        именем не затрагивается.
        """
        raw = (self.output_dir or "").strip()
        if not raw:
            return
        base = self._root or paths.ROOT_DIR
        try:
            target = Path(raw).expanduser()
            if target.name.lower() != OUTPUTS_DIRNAME.lower():
                return
            resolved_target = target.resolve()
            resolved_base = Path(base).resolve()
            try:
                resolved_target.relative_to(resolved_base)
                return      # это outputs текущего проекта — всё нормально
            except ValueError:
                pass
            log.warning(
                "Папка результатов %s указывает на старое расположение проекта — "
                "сбрасываю на outputs/ текущего проекта (%s)",
                raw, resolved_base / OUTPUTS_DIRNAME,
            )
            self.output_dir = ""
        except (OSError, ValueError) as exc:
            log.debug("Миграция output_dir не выполнена для %r: %s", raw, exc)

    def save(self, root: Optional[Path] = None) -> None:
        path = self.file_path(root if root is not None else getattr(self, "_root", None))
        payload = {"version": 1, "settings": asdict(self)}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            log.debug("Настройки сохранены: %s", path)
        except OSError as exc:
            log.warning("Не удалось сохранить настройки %s: %s", path, exc)

    # ------------------------------------------------------------------ #
    def resolved_output_dir(self) -> Path:
        """Куда сохранять результаты. Пустой output_dir = outputs/ проекта.

        Если сохранённая папка недоступна (диск отключён, нет прав, проект
        переехал) — НЕ создаём мусор в случайных местах, а откатываемся на
        outputs/ текущего проекта и сообщаем об этом в лог.
        """
        if self.output_dir:
            path = Path(self.output_dir).expanduser()
            try:
                path.mkdir(parents=True, exist_ok=True)
                return path
            except OSError as exc:
                fallback = paths.outputs_dir(self._root)
                log.warning(
                    "Папка результатов %s недоступна (%s) — использую %s", path, exc, fallback
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
        """Возвращает ВСЕ настройки к стандартным (включая геометрию окна)."""
        defaults = Settings()
        for f in fields(self):
            object.__setattr__(self, f.name, getattr(defaults, f.name))
        self.save()
