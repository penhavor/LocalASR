"""Тесты каталога моделей и логики выбора рекомендации."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.models_catalog import ModelCatalog, ModelEntry, ct2_subfolder, detect_engine
from app.constants import ENGINE_FASTER_WHISPER, ENGINE_VOSK, ENGINE_WHISPERCPP


@pytest.fixture()
def catalog(temp_root: Path) -> ModelCatalog:
    return ModelCatalog(root=temp_root)


def test_catalog_loads_and_is_not_empty(catalog: ModelCatalog) -> None:
    assert len(catalog.all()) >= 10
    assert catalog.loaded_from != "каталог не найден (пустой список)"


def test_every_entry_has_required_fields(catalog: ModelCatalog) -> None:
    for entry in catalog.all():
        assert entry.id, "пустой id"
        assert entry.name, f"пустое имя у {entry.id}"
        assert entry.engine in (ENGINE_FASTER_WHISPER, ENGINE_WHISPERCPP, ENGINE_VOSK)
        assert entry.source, f"нет источника у {entry.id}"
        assert entry.size_bytes and entry.size_bytes > 1_000_000, f"подозрительный размер у {entry.id}"
        assert entry.ram_mb > 0, f"не указана память у {entry.id}"
        assert entry.languages, f"не указаны языки у {entry.id}"
        assert 1 <= entry.quality <= 5
        assert 1 <= entry.speed <= 5
        assert entry.description and len(entry.description) > 40


def test_ids_are_unique(catalog: ModelCatalog) -> None:
    ids = [entry.id for entry in catalog.all()]
    assert len(ids) == len(set(ids))


def test_russian_models_present_and_prioritised(catalog: ModelCatalog) -> None:
    russian = [entry for entry in catalog.all() if entry.supports_russian]
    assert len(russian) >= 4, "нужны и универсальные, и чисто русские модели"
    top = catalog.sorted_for_display(catalog.all())[0]
    assert top.supports_russian, "самая верхняя модель должна поддерживать русский"


def test_has_recommendation(catalog: ModelCatalog) -> None:
    assert any(entry.recommended for entry in catalog.all())
    assert any(entry.recommended_for for entry in catalog.all())


def test_multiple_engines_available(catalog: ModelCatalog) -> None:
    engines = {entry.engine for entry in catalog.all()}
    assert engines >= {ENGINE_FASTER_WHISPER, ENGINE_WHISPERCPP, ENGINE_VOSK}


def test_recommendation_respects_hardware(catalog: ModelCatalog) -> None:
    class WeakHardware:
        cuda_runtime_ok = False
        vram_mb = 0
        ram_available_mb = 1200
        ram_total_mb = 4096

    pick = catalog.recommendation(WeakHardware())
    assert pick is not None
    assert pick.size_bytes <= 700_000_000, "для 4 ГБ RAM не должны советовать гигабайтную модель"


def test_recommendation_prefers_russian_on_strong_hardware(catalog: ModelCatalog) -> None:
    class StrongHardware:
        cuda_runtime_ok = True
        vram_mb = 12_000
        ram_available_mb = 24_000
        ram_total_mb = 32_000

    pick = catalog.recommendation(StrongHardware())
    assert pick is not None
    assert pick.supports_russian


def test_filter_by_engine_and_language(catalog: ModelCatalog) -> None:
    vosk = catalog.filter(engine=ENGINE_VOSK)
    assert vosk and all(entry.engine == ENGINE_VOSK for entry in vosk)

    russian_english = catalog.filter(language="ru")
    assert all("ru" in [lang.lower() for lang in entry.languages] for entry in russian_english)

    only_en = catalog.filter(language="en", engine=ENGINE_FASTER_WHISPER)
    assert only_en


def test_installed_detection_by_meta_file(catalog: ModelCatalog, temp_root: Path) -> None:
    folder = temp_root / "models" / "ru-turbo-corriolon-int8"
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"\x00" * 10)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "installed.json").write_text(
        json.dumps({"id": "ru-turbo-corriolon-int8", "engine": ENGINE_FASTER_WHISPER,
                    "installed_size_bytes": 10}),
        encoding="utf-8",
    )

    catalog.refresh_installed()
    entry = catalog.get("ru-turbo-corriolon-int8")
    assert entry is not None and entry.installed
    assert entry.install_dir == folder
    assert catalog.filter(installed_only=True)
    assert len(catalog.installed()) == 1


def test_unmanaged_folder_is_picked_up(catalog: ModelCatalog, temp_root: Path) -> None:
    folder = temp_root / "models" / "my-own-model"
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"\x00" * 8)
    (folder / "config.json").write_text("{}", encoding="utf-8")

    catalog.refresh_installed()
    entries = [entry for entry in catalog.all() if entry.install_dir == folder]
    assert entries, "папка без installed.json должна попасть в список"
    assert entries[0].engine == ENGINE_FASTER_WHISPER
    assert entries[0].local_only


def test_detect_engine(tmp_path: Path) -> None:
    vosk = tmp_path / "vosk"
    (vosk / "am").mkdir(parents=True)
    assert detect_engine(vosk) == ENGINE_VOSK

    ggml = tmp_path / "ggml"
    ggml.mkdir()
    (ggml / "ggml-large-v3-turbo-q5_0.bin").write_bytes(b"\x00")
    assert detect_engine(ggml) == ENGINE_WHISPERCPP

    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (ct2 / "model.bin").write_bytes(b"\x00")
    assert detect_engine(ct2) == ENGINE_FASTER_WHISPER

    assert detect_engine(tmp_path / "missing") == ""


def test_ct2_subfolder(tmp_path: Path) -> None:
    root = tmp_path / "model"
    inner = root / "ct2_int8_float16"
    inner.mkdir(parents=True)
    (inner / "model.bin").write_bytes(b"\x00")
    assert ct2_subfolder(root) == "ct2_int8_float16"
    assert ct2_subfolder(inner) == ""


def test_entry_text_helpers() -> None:
    entry = ModelEntry(
        id="x",
        name="X",
        size_bytes=1_617_884_929,
        ram_mb=2600,
        vram_mb=3000,
        languages=["ru", "en", "de", "fr", "es", "it", "pl", "tr"],
    )
    assert "1." in entry.size_text or "ГБ" in entry.size_text
    assert "RAM" in entry.memory_text and "VRAM" in entry.memory_text
    assert entry.supports_russian
    assert "ещё 3" in entry.languages_text


def test_export_json_roundtrip(catalog: ModelCatalog) -> None:
    payload = json.loads(catalog.export_json())
    assert payload["models"]
    entry = ModelEntry.from_dict(payload["models"][0])
    assert entry.id
