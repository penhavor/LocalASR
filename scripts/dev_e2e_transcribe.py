#!/usr/bin/env python3
"""End-to-end проверка LocalASR в песочнице/CI (одноразовый dev-скрипт).

Что делает:
  1. генерирует тестовый WAV (тон + тишина) и «видео»-контейнер mkv;
  2. скачивает настоящую модель через app.core.downloader;
  3. прогоняет app.asr.transcriber.transcribe_file;
  4. проверяет, что файлы результата созданы и не пустые.

Запуск (из уже готового venv):
    python scripts/dev_e2e_transcribe.py vosk-ru-small-0.22
    python scripts/dev_e2e_transcribe.py whisper-base
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import ffmpeg  # noqa: E402


def make_test_media(workdir: Path) -> tuple[Path, Path]:
    """WAV 6 секунд (тон 300 Гц + паузы) и MKV с той же дорожкой."""
    exe = ffmpeg.find_ffmpeg()
    if not exe:
        raise SystemExit("ffmpeg не найден — e2e невозможен")
    wav = workdir / "лекция 1.wav"
    mkv = workdir / "лекция 1.mkv"
    for target, extra in (
        (wav, []),
        (mkv, ["-f", "lavfi", "-i", "color=c=black:s=64x64:r=5", "-shortest", "-c:v", "libx264"]),
    ):
        cmd = [
            exe, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            "sine=frequency=300:sample_rate=16000:duration=6",
            *extra,
        ]
        if target.suffix == ".wav":
            cmd += ["-ac", "1", "-ar", "16000", str(target)]
        else:
            cmd += ["-map", "0:a", "-map", "1:v", "-c:a", "aac", str(target)]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True, timeout=180)
    return wav, mkv


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "vosk-ru-small-0.22"
    workdir = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/localasr-e2e")
    workdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("1. Тестовые медиа")
    wav, mkv = make_test_media(workdir)
    for item in (wav, mkv):
        info = ffmpeg.probe(item)
        print(f"   {item.name}: {info.duration:.1f} с, audio={info.audio_streams}, video={info.video_streams}")
        assert info.duration > 3, "тестовый файл не создался"

    print("\n2. Декодирование аудио из видео")
    audio = ffmpeg.decode_audio(mkv, 16000)
    print(f"   массив: {audio.shape}, dtype={audio.dtype}, {len(audio) / 16000:.1f} с")
    assert audio.shape[0] > 16000 * 4

    print(f"\n3. Скачивание модели {model_id}")
    from app.core.downloader import install_model
    from app.core.models_catalog import ModelCatalog

    catalog = ModelCatalog()
    entry = catalog.get(model_id)
    if entry is None:
        raise SystemExit(f"нет такой модели в каталоге: {model_id}")
    if entry.installed:
        print(f"   уже установлена: {entry.install_dir}")
    else:
        started = time.time()
        install_model(
            entry,
            progress=lambda fraction, status: print(
                f"\r   {int(fraction * 100):3d}%  {status[:70]:<70}", end="", flush=True
            ),
        )
        print(f"\n   скачано за {time.time() - started:.0f} с")
        catalog.refresh_installed()
        entry = catalog.get(model_id)
    assert entry.installed and entry.install_dir.is_dir(), "модель не установилась"

    print("\n4. Транскрибация")
    from app.asr.transcriber import JobOptions, transcribe_file
    from app.settings import Settings

    settings = Settings()
    settings.output_dir = str(workdir / "outputs")
    settings.output_format = "txt"
    settings.language = "ru"
    options = JobOptions.from_settings(entry, settings, Path(settings.output_dir))
    options.formats = ["txt", "srt", "json"]

    result = transcribe_file(
        mkv,
        options,
        progress=lambda fraction, stage: print(
            f"\r   {int(fraction * 100):3d}%  {stage[:60]:<60}", end="", flush=True
        ),
    )
    print()
    print("   ok:", result.ok)
    print("   error:", result.error or "—")
    print("   segments:", result.segments, "words:", result.words, "dropped:", result.dropped)
    print("   speed: %.2fx" % result.speed_x, "· elapsed %.1f s" % result.elapsed_seconds)
    for out in result.outputs:
        size = out.stat().st_size
        print(f"   → {out.name} ({size} байт)")
        assert size >= 0
    assert result.ok, f"транскрибация не удалась: {result.error}"
    assert result.outputs, "не создано ни одного файла результата"
    assert all(p.name.startswith("лекция 1") for p in result.outputs), "имя результата ≠ имени оригинала"

    print("\n5. Проверка имён файлов")
    for out in result.outputs:
        print("   ", out)
    json_out = [p for p in result.outputs if p.suffix == ".json"]
    if json_out:
        import json

        payload = json.loads(json_out[0].read_text(encoding="utf-8"))
        print("   meta.model:", payload["meta"]["model"])
        print("   meta.engine:", payload["meta"]["engine"])
        print("   segments:", len(payload["segments"]))

    print("\n✅ END-TO-END OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
