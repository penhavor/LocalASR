"""Проверка движка whisper.cpp и «своей модели по ссылке» (dev-скрипт).

    python scripts/dev_e2e_custom_model.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GGML_URL = "https://huggingface.co/antirez/whisper-tiny/resolve/main/ggml-model.bin"
GGML_URL_RU = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else GGML_URL_RU
    workdir = Path("/tmp/localasr-e2e-custom")
    workdir.mkdir(parents=True, exist_ok=True)

    from app.core.downloader import inspect_custom_source, install_model
    from app.core import ffmpeg
    from app.asr.transcriber import JobOptions, transcribe_file
    from app.settings import Settings

    print(f"1. Автоопределение модели по ссылке:\n   {url}")
    entry = inspect_custom_source(url)
    print(f"   id       : {entry.id}")
    print(f"   engine   : {entry.engine}")
    print(f"   file     : {entry.file}")
    print(f"   size     : {entry.size_text}")
    print(f"   ram (оценка): {entry.ram_mb} МБ")
    assert entry.engine == "whispercpp", f"ожидали whisper.cpp, получили {entry.engine}"

    print("\n2. Установка")
    started = time.time()
    install_model(
        entry,
        progress=lambda fraction, status: print(
            f"\r   {int(fraction * 100):3d}%  {status[:66]:<66}", end="", flush=True
        ),
    )
    print(f"\n   готово за {time.time() - started:.0f} с -> {entry.install_dir}")
    assert entry.install_dir.is_dir()
    assert list(entry.install_dir.rglob("*.bin")), "GGML-файл не скачался"

    print("\n3. Тестовое видео")
    exe = ffmpeg.find_ffmpeg()
    mkv = workdir / "лекция 2.mkv"
    import subprocess

    subprocess.run(
        [exe, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=16000:duration=5",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5",
         "-shortest", "-map", "0:a", "-map", "1:v", "-c:v", "libx264", "-c:a", "aac", str(mkv)],
        check=True, timeout=180,
    )
    info = ffmpeg.probe(mkv)
    print(f"   {mkv.name}: {info.duration:.1f} с, audio={info.audio_streams}, video={info.video_streams}")

    print("\n4. Транскрибация через whisper.cpp")
    settings = Settings()
    settings.output_dir = str(workdir / "outputs")
    settings.language = "ru"
    options = JobOptions.from_settings(entry, settings, Path(settings.output_dir))
    options.formats = ["txt", "vtt"]

    result = transcribe_file(
        mkv,
        options,
        progress=lambda fraction, stage: print(
            f"\r   {int(fraction * 100):3d}%  {stage[:60]:<60}", end="", flush=True
        ),
    )
    print("\n   ok:", result.ok, "| error:", result.error or "—")
    print("   segments:", result.segments, "words:", result.words)
    print("   engine:", result.engine_label)
    print("   скорость: %.2fx" % result.speed_x)
    for out in result.outputs:
        print(f"   → {out.name} ({out.stat().st_size} байт)")
    assert result.ok, result.error
    assert all(p.name.startswith("лекция 2") for p in result.outputs)

    print("\n✅ WHISPER.CPP + СВОЯ МОДЕЛЬ OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
