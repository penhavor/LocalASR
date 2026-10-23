"""Тесты настроек: сохранение/загрузка, защита от битого JSON, нормализация."""

from __future__ import annotations

import json
from pathlib import Path

from app.settings import Settings


def test_defaults_are_sane() -> None:
    settings = Settings()
    assert settings.language == "ru"
    assert settings.output_format == "txt"
    assert settings.use_vad is True
    assert settings.suppress_hallucinations is True
    assert settings.beam_size == 5
    assert settings.device in ("auto", "cpu", "cuda")


def test_save_and_load_roundtrip(temp_root: Path) -> None:
    settings = Settings()
    settings.language = "uk"
    settings.output_format = "srt"
    settings.beam_size = 8
    settings.output_dir = str(temp_root / "custom")
    settings.selected_model_id = "ru-turbo-corriolon-int8"
    settings.save(root=temp_root)

    loaded = Settings.load(root=temp_root)
    assert loaded.language == "uk"
    assert loaded.output_format == "srt"
    assert loaded.beam_size == 8
    assert loaded.output_dir == str(temp_root / "custom")
    assert loaded.selected_model_id == "ru-turbo-corriolon-int8"


def test_broken_settings_file_does_not_crash(temp_root: Path) -> None:
    path = Settings.file_path(temp_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ это не json", encoding="utf-8")

    loaded = Settings.load(root=temp_root)
    assert loaded.language == "ru"          # дефолты
    assert path.with_suffix(".json.broken").exists() or not path.exists()


def test_unknown_and_wrong_typed_keys_are_ignored(temp_root: Path) -> None:
    path = Settings.file_path(temp_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "settings": {
                    "language": "ru",
                    "beam_size": "не число",
                    "unknown_key": 42,
                    "batched": 1,
                    "splitter_sizes": "не список",
                }
            }
        ),
        encoding="utf-8",
    )
    loaded = Settings.load(root=temp_root)
    assert loaded.beam_size >= 1            # не упали, значение нормализовано
    assert isinstance(loaded.batched, bool)
    assert isinstance(loaded.splitter_sizes, list)


def test_values_are_clamped(temp_root: Path) -> None:
    path = Settings.file_path(temp_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"settings": {"beam_size": 999, "batch_size": -5, "parallel_files": 99, "device": "quantum"}}),
        encoding="utf-8",
    )
    loaded = Settings.load(root=temp_root)
    assert 1 <= loaded.beam_size <= 12
    assert 1 <= loaded.batch_size <= 64
    assert 1 <= loaded.parallel_files <= 8
    assert loaded.device == "auto"


def test_resolved_output_dir_creates_folder(temp_root: Path) -> None:
    settings = Settings()
    settings.output_dir = str(temp_root / "results" / "deep")
    directory = settings.resolved_output_dir()
    assert directory.exists()
    assert directory == temp_root / "results" / "deep"


def test_update_only_known_keys() -> None:
    settings = Settings()
    settings.update({"language": "en", "totally_unknown": 1})
    assert settings.language == "en"
    assert not hasattr(settings, "totally_unknown")
