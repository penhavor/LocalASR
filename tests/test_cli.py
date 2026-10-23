"""Тесты CLI: парсер, list-models, doctor, headless-транскрибация без модели."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import cli


def test_parser_gui_by_default() -> None:
    args = cli.build_parser().parse_args([])
    assert args.command is None


def test_parser_transcribe() -> None:
    args = cli.build_parser().parse_args(
        ["transcribe", "лекция 1.mp4", "лекция 2.mkv", "-m", "whisper-small", "-f", "srt", "-l", "ru", "--words"]
    )
    assert args.command == "transcribe"
    assert args.inputs == ["лекция 1.mp4", "лекция 2.mkv"]
    assert args.model == "whisper-small"
    assert args.format == "srt"
    assert args.language == "ru"
    assert args.words is True


def test_parser_format_choices_guard() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["transcribe", "a.mp4", "-f", "pdf"])


def test_parser_flags() -> None:
    args = cli.build_parser().parse_args(
        ["transcribe", "a.mp4", "--no-vad", "--no-clean", "--overwrite", "--all-formats", "--header", "--no-batched"]
    )
    assert args.no_vad and args.no_clean and args.overwrite
    assert args.all_formats and args.header and args.no_batched


def test_list_models_runs(capsys) -> None:
    args = cli.build_parser().parse_args(["list-models"])
    assert cli.cmd_list_models(args) == 0
    out = capsys.readouterr().out
    assert "моделей в каталоге" in out
    assert "faster-whisper" in out


def test_list_models_json(capsys) -> None:
    import json

    args = cli.build_parser().parse_args(["list-models", "--json"])
    assert cli.cmd_list_models(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"]


def test_doctor_runs(capsys) -> None:
    args = cli.build_parser().parse_args(["doctor"])
    assert cli.cmd_doctor(args) == 0
    out = capsys.readouterr().out
    assert "doctor" in out and "ffmpeg" in out


def test_version(capsys) -> None:
    args = cli.build_parser().parse_args(["version"])
    assert cli.cmd_version(args) == 0
    assert "LocalASR" in capsys.readouterr().out


def test_resolve_model_by_id_and_fuzzy() -> None:
    from app.core.models_catalog import ModelCatalog

    catalog = ModelCatalog()
    assert cli._resolve_model(catalog, "whisper-small").id == "whisper-small"
    assert cli._resolve_model(catalog, "WHISPER-SMALL").id == "whisper-small"
    assert cli._resolve_model(catalog, "large-v3-turbo") is not None
    assert cli._resolve_model(catalog, "") is None
    assert cli._resolve_model(catalog, "несуществующая") is None


def test_transcribe_without_media_files_fails_gracefully(tmp_path: Path, capsys) -> None:
    args = cli.build_parser().parse_args(["transcribe", str(tmp_path / "нет-такого-файла.mp4")])
    code = cli.cmd_transcribe(args)
    assert code == 2
    assert "Не найдено медиафайлов" in capsys.readouterr().err


def test_transcribe_without_models_prints_hint(tmp_path: Path, capsys, monkeypatch) -> None:
    """Если моделей нет вообще — должна быть понятная подсказка, а не трейсбек."""

    monkeypatch.setattr(cli, "_resolve_model", lambda catalog, wanted: None)
    monkeypatch.setattr(
        "app.core.models_catalog.ModelCatalog.installed", lambda self: []
    )
    media = tmp_path / "лекция.mp4"
    media.write_bytes(b"\x00" * 16)
    args = cli.build_parser().parse_args(["transcribe", str(media)])
    code = cli.cmd_transcribe(args)
    assert code == 2
    assert "модель" in capsys.readouterr().err.lower()


def test_inside_venv_detection() -> None:
    assert isinstance(cli._inside_venv(), bool)
