"""Тесты путей и имён файлов (включая кириллицу и «говорящие» имена)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import paths


@pytest.mark.parametrize(
    "source,expected",
    [
        ("лекция 1.mp4", "лекция 1"),
        ("лекция 1", "лекция 1"),
        ("C:\\Users\\user\\Videos\\лекция 1.mp4", "лекция 1"),
        ("/home/user/лекция 1.mkv", "лекция 1"),
        ("a/b:c*.mp4", "b_c"),         # берётся только basename, «:*» заменяются
        ('le"c|ti?on.mp4', "le_c_ti_on"),
        ("   ", "transcript"),
        ("...", "transcript"),
        ("CON.mp4", "_CON"),
        ("very" + "long" * 60 + ".mp4", None),
    ],
)
def test_sanitize_stem(source: str, expected: str | None) -> None:
    result = paths.sanitize_stem(source)
    if expected is None:
        assert len(result) <= paths.MAX_STEM_LENGTH
        assert result
    else:
        assert result == expected


def test_sanitize_stem_keeps_cyrillic_and_spaces() -> None:
    assert paths.sanitize_stem("Лекция №3 — Машинное обучение.mp4") == "Лекция №3 — Машинное обучение"


def test_unique_path_no_overwrite(tmp_path: Path) -> None:
    first = paths.unique_path(tmp_path, "лекция 1", ".txt")
    assert first.name == "лекция 1.txt"
    first.write_text("x", encoding="utf-8")

    second = paths.unique_path(tmp_path, "лекция 1", ".txt")
    assert second.name == "лекция 1 (2).txt"

    second.write_text("y", encoding="utf-8")
    third = paths.unique_path(tmp_path, "лекция 1", ".txt")
    assert third.name == "лекция 1 (3).txt"


def test_unique_path_overwrite(tmp_path: Path) -> None:
    (tmp_path / "лекция 1.txt").write_text("старое", encoding="utf-8")
    result = paths.unique_path(tmp_path, "лекция 1", ".txt", overwrite=True)
    assert result.name == "лекция 1.txt"


def test_output_suffix_mapping() -> None:
    assert paths.output_suffix("txt") == ".txt"
    assert paths.output_suffix("txt-time") == ".txt"
    assert paths.output_suffix("srt") == ".srt"
    assert paths.output_suffix("vtt") == ".vtt"
    assert paths.output_suffix("json") == ".json"
    assert paths.output_suffix("md") == ".md"
    assert paths.output_suffix("unknown") == ".txt"


def test_expand_media_inputs(tmp_path: Path, sample_media: list[Path]) -> None:
    result = paths.expand_media_inputs([str(p) for p in sample_media])
    names = {p.name for p in result}
    assert "лекция 1.mp4" in names
    assert "lecture-02.mkv" in names
    assert "подкаст.mp3" in names
    assert "notes.txt" not in names, "не-медиа файлы не должны попадать в очередь"

    folder_result = paths.expand_media_inputs([str(tmp_path / "media")])
    assert len(folder_result) == 3

    # дубликаты не размножаются
    doubled = paths.expand_media_inputs([str(sample_media[0]), str(sample_media[0])])
    assert len(doubled) == 1


def test_is_media_file() -> None:
    assert paths.is_media_file("a.MP4")
    assert paths.is_media_file("a.opus")
    assert not paths.is_media_file("a.txt")
    assert not paths.is_media_file("a.exe")


def test_human_size() -> None:
    assert paths.human_size(None) == "—"
    assert paths.human_size(0) == "0 Б"
    assert paths.human_size(1024) == "1 КБ"
    assert "ГБ" in paths.human_size(1_617_884_929)
    assert "ГБ" in paths.human_size(3_087_284_237)


def test_short_path() -> None:
    long = "/home/user/очень/длинный/путь/до/файла/лекция 1.mp4"
    short = paths.short_path(long, 40)
    assert len(short) <= 44
    assert short.startswith("…/") or short.endswith("лекция 1.mp4")
    assert paths.short_path("a.mp4", 40) == "a.mp4"


def test_atomic_write_text(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.txt"
    paths.atomic_write_text(target, "привет")
    assert target.read_text(encoding="utf-8") == "привет"
    assert not list(tmp_path.rglob("*.part"))


def test_ensure_free_space_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        paths.ensure_free_space(tmp_path, 10**18)
    # маленький объём проходит
    paths.ensure_free_space(tmp_path, 1024)


def test_dir_size(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"12345")
    (tmp_path / "inner").mkdir()
    (tmp_path / "inner" / "b.bin").write_bytes(b"67890")
    assert paths.dir_size(tmp_path) == 10
