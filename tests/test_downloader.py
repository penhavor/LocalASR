"""Тесты разбора источников моделей и «установки» из локальной папки."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.constants import ENGINE_FASTER_WHISPER, ENGINE_VOSK, ENGINE_WHISPERCPP
from app.core import downloader, net
from app.core.models_catalog import ModelEntry


# --------------------------------------------------------------------------- #
#  Разбор ссылок на Hugging Face
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source,expected_repo,expected_sub,expected_revision",
    [
        ("owner/name", "owner/name", "", "main"),
        ("owner/name/tree/main/ct2_int8_float16", "owner/name", "ct2_int8_float16", "main"),
        ("owner/name/tree/v1.0/sub/dir", "owner/name", "sub/dir", "v1.0"),
        ("owner/name:ct2_int8", "owner/name", "ct2_int8", "main"),
        ("https://huggingface.co/owner/name", "owner/name", "", "main"),
        ("https://huggingface.co/owner/name/tree/main/ct2_int16", "owner/name", "ct2_int16", "main"),
        ("https://huggingface.co/owner/name/tree/dev", "owner/name", "", "dev"),
        ("owner/name/", "owner/name", "", "main"),
    ],
)
def test_split_repo_and_subfolder(
    source: str, expected_repo: str, expected_sub: str, expected_revision: str
) -> None:
    repo, sub, revision = downloader.split_repo_and_subfolder(source)
    assert (repo, sub, revision) == (expected_repo, expected_sub, expected_revision)


def test_is_hf_repo_id() -> None:
    assert net.is_hf_repo_id("coriollon/whisper-large-v3-turbo-russian")
    assert net.is_hf_repo_id("Systran/faster-whisper-large-v3")
    assert not net.is_hf_repo_id("https://huggingface.co/x/y")
    assert not net.is_hf_repo_id("C:\\models\\mine")
    assert not net.is_hf_repo_id("just-a-name")
    assert not net.is_hf_repo_id("")


def test_sanitize_id() -> None:
    assert downloader.sanitize_id("owner/model name") == "owner_model_name"
    assert downloader.sanitize_id("coriollon/whisper-large-v3-turbo-russian") == "coriollon_whisper-large-v3-turbo-russian"
    assert downloader.sanitize_id("") == "model"
    assert "/" not in downloader.sanitize_id("a/b/c")


# --------------------------------------------------------------------------- #
#  Установка из локальной папки (без сети)
# --------------------------------------------------------------------------- #


def _make_ct2_model(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.bin").write_bytes(b"CT2" + b"\x00" * 64)
    (folder / "config.json").write_text('{"variables": {}}', encoding="utf-8")
    (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
    return folder


def test_install_from_local_folder(temp_root: Path, tmp_path: Path) -> None:
    source = _make_ct2_model(tmp_path / "external-model")
    entry = ModelEntry(
        id="my-local-model",
        name="Моя локальная модель",
        engine=ENGINE_FASTER_WHISPER,
        source=str(source),
        languages=["ru"],
        size_bytes=1024,
        custom=True,
    )

    progress_log = []
    result = downloader.install_model(
        entry,
        root=temp_root,
        progress=lambda fraction, status: progress_log.append((fraction, status)),
    )

    assert result.entry.installed is True
    assert result.directory == temp_root / "models" / "my-local-model"
    target = temp_root / "models" / "my-local-model"
    assert (target / "model.bin").is_file()
    assert (target / "installed.json").is_file()

    meta = json.loads((target / "installed.json").read_text(encoding="utf-8"))
    assert meta["id"] == "my-local-model"
    assert meta["engine"] == ENGINE_FASTER_WHISPER
    assert meta["languages"] == ["ru"]

    assert progress_log and progress_log[-1][0] == pytest.approx(1.0)


def test_install_detects_engine_from_files(temp_root: Path, tmp_path: Path) -> None:
    source = tmp_path / "ggml-model"
    source.mkdir()
    (source / "ggml-large-v3-turbo-q5_0.bin").write_bytes(b"\x00" * 32)

    entry = ModelEntry(
        id="ggml-test",
        name="GGML test",
        engine="",          # намеренно пусто — должен определиться сам
        source=str(source),
        size_bytes=32,
    )
    result = downloader.install_model(entry, root=temp_root)
    assert result.entry.engine == ENGINE_WHISPERCPP


def test_uninstall_removes_folder(temp_root: Path, tmp_path: Path) -> None:
    source = _make_ct2_model(tmp_path / "to-delete")
    entry = ModelEntry(id="to-delete", name="x", source=str(source), size_bytes=10)
    downloader.install_model(entry, root=temp_root)

    folder = temp_root / "models" / "to-delete"
    assert folder.is_dir()

    entry.install_dir = folder
    assert downloader.uninstall_model(entry, root=temp_root) is True
    assert not folder.exists()


def test_uninstall_refuses_to_touch_files_outside_models(temp_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "important"
    outside.mkdir()
    (outside / "data.bin").write_bytes(b"1")
    entry = ModelEntry(id="outside", name="outside", install_dir=outside)
    assert downloader.uninstall_model(entry, root=temp_root) is False
    assert outside.exists()


def test_resolve_source_rejects_garbage() -> None:
    entry = ModelEntry(id="x", name="x", source="это не источник")
    with pytest.raises(downloader.InstallError):
        downloader.resolve_source(entry)

    empty = ModelEntry(id="y", name="y", source="")
    with pytest.raises(downloader.InstallError):
        downloader.resolve_source(empty)


def test_plan_local_file(temp_root: Path, tmp_path: Path) -> None:
    ggml = tmp_path / "ggml-tiny.bin"
    ggml.write_bytes(b"\x00" * 128)
    entry = ModelEntry(id="t", name="t", engine=ENGINE_WHISPERCPP, source=str(ggml))
    plan = downloader.resolve_source(entry)
    assert plan["kind"] == "local"
    assert plan["total_bytes"] == 128


def test_plan_url_file() -> None:
    url = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
    entry = ModelEntry(id="v", name="v", engine=ENGINE_VOSK, source=url, size_bytes=45_000_000)
    plan = downloader.resolve_source(entry)
    assert plan["kind"] == "url_file"
    assert plan["files"][0][0].endswith(".zip")
    assert plan["files"][0][1] == url


def test_inspect_local_source(tmp_path: Path) -> None:
    folder = _make_ct2_model(tmp_path / "inspect-me")
    entry = downloader.inspect_custom_source(str(folder))
    assert entry.engine == ENGINE_FASTER_WHISPER
    assert entry.size_bytes > 0
    assert entry.custom is True


def test_inspect_ggml_url() -> None:
    url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
    entry = downloader.inspect_custom_source(url)
    assert entry.engine == ENGINE_WHISPERCPP
    assert entry.file == "ggml-large-v3-turbo-q5_0.bin"


def test_inspect_vosk_zip_url() -> None:
    url = "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip"
    entry = downloader.inspect_custom_source(url)
    assert entry.engine == ENGINE_VOSK


def test_inspect_rejects_garbage() -> None:
    with pytest.raises(downloader.InstallError):
        downloader.inspect_custom_source("   ")
    with pytest.raises(downloader.InstallError):
        downloader.inspect_custom_source("не-репо")


def test_ram_guess_for_custom_entry() -> None:
    entry = ModelEntry(id="x", name="x", engine=ENGINE_WHISPERCPP, source="s", size_bytes=574_000_000)
    downloader._guess_ram(entry)
    assert 500 < entry.ram_mb < 2000


def test_flatten_single_subdir(tmp_path: Path) -> None:
    root = tmp_path / "model"
    inner = root / "vosk-model-ru-0.42"
    (inner / "am").mkdir(parents=True)
    (inner / "am" / "final.mdl").write_bytes(b"x")
    downloader._flatten_single_subdir(root)
    assert (root / "am" / "final.mdl").is_file()
    assert not inner.exists()
