"""Работа со звуком: поиск ffmpeg, чтение метаданных, декодирование в 16 кГц моно.

Видео-файлы «тупо переводятся в звук»: нам нужна только аудиодорожка, поэтому
декодирование идёт через пайп (`-f f32le`) — без промежуточных WAV на диске.

Порядок поиска декодера:
  1. системный ffmpeg (самый быстрый, поддерживает всё);
  2. бинарник из пакета imageio-ffmpeg (ставится вместе с приложением — работает
     всегда, даже если в системе ffmpeg нет);
  3. PyAV (чистый Python-биндинг, запасной вариант для экзотики).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .. import paths
from ..constants import DECODER_TIMEOUT_PER_HOUR, SAMPLE_RATE
from ..logutils import get_logger

log = get_logger("audio")

IS_WINDOWS = os.name == "nt"

CancelToken = Callable[[], bool]

_FFMPEG_CACHE: dict[str, Optional[str]] = {}


class AudioError(RuntimeError):
    """Не удалось получить звук из файла."""


# --------------------------------------------------------------------------- #
#  Поиск ffmpeg
# --------------------------------------------------------------------------- #


def find_ffmpeg() -> Optional[str]:
    """Путь к ffmpeg (кэшируется). None — если не нашли вообще ничего."""
    if "ffmpeg" in _FFMPEG_CACHE:
        return _FFMPEG_CACHE["ffmpeg"]

    exe = "ffmpeg.exe" if IS_WINDOWS else "ffmpeg"

    # 1) рядом с проектом (если пользователь положил свой бинарник)
    for candidate in (
        paths.ROOT_DIR / "bin" / exe,
        paths.ROOT_DIR / exe,
        Path(sys.executable).parent / exe,
    ):
        if candidate.is_file():
            _FFMPEG_CACHE["ffmpeg"] = str(candidate)
            return str(candidate)

    # 2) системный
    found = shutil.which("ffmpeg")
    if found:
        _FFMPEG_CACHE["ffmpeg"] = found
        return found

    # 3) из пакета imageio-ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            _FFMPEG_CACHE["ffmpeg"] = bundled
            log.info("Используется ffmpeg из пакета imageio-ffmpeg: %s", bundled)
            return bundled
    except Exception as exc:  # noqa: BLE001
        log.debug("imageio-ffmpeg недоступен: %s", exc)

    _FFMPEG_CACHE["ffmpeg"] = None
    return None


def find_ffprobe() -> Optional[str]:
    if "ffprobe" in _FFMPEG_CACHE:
        return _FFMPEG_CACHE["ffprobe"]
    exe = "ffprobe.exe" if IS_WINDOWS else "ffprobe"
    found = shutil.which("ffprobe")
    if not found:
        candidate = paths.ROOT_DIR / "bin" / exe
        found = str(candidate) if candidate.is_file() else None
    _FFMPEG_CACHE["ffprobe"] = found
    return found


def ffmpeg_version() -> str:
    exe = find_ffmpeg()
    if not exe:
        return "не найден"
    try:
        kwargs: dict = {"capture_output": True, "text": True, "timeout": 15}
        if IS_WINDOWS:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run([exe, "-version"], **kwargs)
        first = (out.stdout or "").splitlines()
        return first[0] if first else "неизвестно"
    except (OSError, subprocess.SubprocessError):
        return "не удалось определить"


def has_ffmpeg() -> bool:
    return find_ffmpeg() is not None


def _popen_kwargs() -> dict:
    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "bufsize": 1024 * 1024,
    }
    if IS_WINDOWS:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kwargs


# --------------------------------------------------------------------------- #
#  Метаданные
# --------------------------------------------------------------------------- #


@dataclass
class MediaInfo:
    path: Path
    duration: float = 0.0          # секунды
    size_bytes: int = 0
    audio_streams: int = 0
    video_streams: int = 0
    is_video: bool = False
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def probe(path: str | Path) -> MediaInfo:
    """Быстрое чтение длительности/потоков. Не декодирует звук целиком."""
    path = Path(path)
    info = MediaInfo(path=path)
    try:
        info.size_bytes = path.stat().st_size
    except OSError:
        pass

    ffprobe = find_ffprobe()
    if ffprobe:
        try:
            kwargs: dict = {
                "capture_output": True, "text": True, "timeout": 60,
                "encoding": "utf-8", "errors": "replace",
            }
            if IS_WINDOWS:
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            out = subprocess.run(
                [
                    ffprobe, "-v", "error", "-print_format", "json",
                    "-show_format", "-show_streams", str(path),
                ],
                **kwargs,
            )
            if out.returncode == 0 and out.stdout.strip():
                import json

                data = json.loads(out.stdout)
                fmt = data.get("format", {}) or {}
                try:
                    info.duration = float(fmt.get("duration") or 0.0)
                except (TypeError, ValueError):
                    info.duration = 0.0
                for stream in data.get("streams", []) or []:
                    kind = stream.get("codec_type")
                    if kind == "audio":
                        info.audio_streams += 1
                        info.codec = stream.get("codec_name") or info.codec
                        try:
                            info.sample_rate = int(stream.get("sample_rate") or 0)
                            info.channels = int(stream.get("channels") or 0)
                        except (TypeError, ValueError):
                            pass
                    elif kind == "video":
                        info.video_streams += 1
                info.is_video = info.video_streams > 0
                if info.duration > 0 and info.audio_streams > 0:
                    return info
        except Exception as exc:  # noqa: BLE001
            log.debug("ffprobe не справился с %s: %s", path.name, exc)

    # запасной путь: парсим stderr самого ffmpeg
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        try:
            kwargs = {
                "capture_output": True, "text": True, "timeout": 60,
                "encoding": "utf-8", "errors": "replace",
            }
            if IS_WINDOWS:
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], **kwargs)
            text = (out.stderr or "") + (out.stdout or "")
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
            if match:
                hours, minutes, seconds = match.groups()
                info.duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            info.audio_streams = len(re.findall(r"Stream.*Audio:", text))
            info.video_streams = len(re.findall(r"Stream.*Video:", text))
            info.is_video = info.video_streams > 0
            codec = re.search(r"Stream.*Audio:\s*([a-z0-9_]+)", text)
            if codec:
                info.codec = codec.group(1)
            rate = re.search(r"Audio:.*?,\s*(\d+)\s*Hz,\s*([a-z0-9.()]+)", text)
            if rate:
                try:
                    info.sample_rate = int(rate.group(1))
                except ValueError:
                    pass
                layout = rate.group(2)
                info.channels = 2 if "stereo" in layout else 1 if "mono" in layout else 0
            if info.audio_streams == 0:
                info.error = "В файле нет аудиодорожки"
            return info
        except Exception as exc:  # noqa: BLE001
            log.debug("ffmpeg -i не справился с %s: %s", path.name, exc)

    # последний вариант — PyAV
    try:
        import av  # type: ignore

        with av.open(str(path)) as container:
            info.duration = float(container.duration or 0) / 1_000_000.0
            info.audio_streams = len(container.streams.audio)
            info.video_streams = len(container.streams.video)
            info.is_video = info.video_streams > 0
            if container.streams.audio:
                stream = container.streams.audio[0]
                info.codec = stream.codec_context.name or ""
                info.sample_rate = int(stream.codec_context.sample_rate or 0)
                info.channels = int(stream.codec_context.channels or 0)
            if info.audio_streams == 0:
                info.error = "В файле нет аудиодорожки"
        return info
    except Exception as exc:  # noqa: BLE001
        info.error = f"Не удалось прочитать файл: {exc}"
        return info


# --------------------------------------------------------------------------- #
#  Декодирование
# --------------------------------------------------------------------------- #


def decode_audio(
    path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    cancel: Optional[CancelToken] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> "object":
    """Возвращает numpy-массив float32, 1 канал, `sample_rate` Гц.

    Бросает AudioError, если звук получить не удалось.
    """

    path = Path(path)
    if not path.is_file():
        raise AudioError(f"Файл не найден: {path}")

    info = probe(path)
    if info.error and info.audio_streams == 0:
        raise AudioError(info.error)
    if info.audio_streams == 0 and not info.is_video:
        raise AudioError("В файле нет аудиодорожки")

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        try:
            return _decode_with_ffmpeg(
                ffmpeg, path, sample_rate, cancel, progress, info.duration
            )
        except AudioError as exc:
            log.warning("ffmpeg не смог декодировать %s: %s — пробую PyAV", path.name, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("Сбой ffmpeg на %s: %s — пробую PyAV", path.name, exc)

    return _decode_with_av(path, sample_rate, cancel, progress, info.duration)


def _decode_with_ffmpeg(
    ffmpeg: str,
    path: Path,
    sample_rate: int,
    cancel: Optional[CancelToken],
    progress: Optional[Callable[[float], None]],
    duration: float,
):
    import numpy as np

    cmd: List[str] = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-vn",                     # отбрасываем видео — нужен только звук
        "-sn", "-dn",              # и субтитры/данные
        "-map", "0:a:0",           # первая аудиодорожка
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "pipe:1",
    ]

    log.info("Декодирую звук: %s", path.name)
    log.debug("ffmpeg cmd: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_popen_kwargs(),
    )

    assert proc.stdout is not None
    chunks: List[bytes] = []
    total_bytes = 0
    bytes_per_second = sample_rate * 4
    try:
        while True:
            if cancel is not None and cancel():
                proc.kill()
                raise AudioError("Декодирование отменено пользователем")
            block = proc.stdout.read(1024 * 512)
            if not block:
                break
            chunks.append(block)
            total_bytes += len(block)
            if progress is not None and duration > 0:
                progress(min(1.0, (total_bytes / bytes_per_second) / duration))

        timeout = max(120, int((duration or 3600) / 3600 * DECODER_TIMEOUT_PER_HOUR))
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise AudioError(f"ffmpeg не завершился за {timeout} с (файл повреждён?)") from exc
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass

    if proc.returncode not in (0, None) and total_bytes == 0:
        message = (stderr or b"").decode("utf-8", "replace").strip()
        raise AudioError(message or f"ffmpeg вернул код {proc.returncode}")

    if total_bytes == 0:
        message = (stderr or b"").decode("utf-8", "replace").strip()
        raise AudioError(message or "ffmpeg не вернул аудиоданные (в файле нет звука?)")

    audio = np.frombuffer(b"".join(chunks), dtype=np.float32)
    if audio.size == 0:
        raise AudioError("Пустая аудиодорожка")
    # защита от NaN/inf в битых файлах
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    log.info("Декодировано %.1f мин звука (%.1f МБ)", audio.size / sample_rate / 60, audio.nbytes / 1e6)
    return audio


def _decode_with_av(
    path: Path,
    sample_rate: int,
    cancel: Optional[CancelToken],
    progress: Optional[Callable[[float], None]],
    duration: float,
):
    import numpy as np

    try:
        import av  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise AudioError(
            "Не найден рабочий декодер звука. Установи ffmpeg или пакет av: "
            f"{exc}"
        ) from exc

    try:
        container = av.open(str(path))
    except Exception as exc:  # noqa: BLE001
        raise AudioError(f"PyAV не смог открыть файл: {exc}") from exc

    try:
        if not container.streams.audio:
            raise AudioError("В файле нет аудиодорожки")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"  # многопоточное декодирование

        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        pieces: List["np.ndarray"] = []
        decoded = 0
        for frame in container.decode(stream):
            if cancel is not None and cancel():
                raise AudioError("Декодирование отменено пользователем")
            for out_frame in resampler.resample(frame):
                array = out_frame.to_ndarray()
                if array.ndim > 1:
                    array = array.reshape(-1)
                pieces.append(array.astype(np.float32, copy=False))
                decoded += array.size
                if progress is not None and duration > 0:
                    progress(min(1.0, (decoded / sample_rate) / duration))
        for out_frame in resampler.resample(None):  # «хвост» ресемплера
            array = out_frame.to_ndarray().reshape(-1)
            pieces.append(array.astype(np.float32, copy=False))

        if not pieces:
            raise AudioError("PyAV не получил аудиоданные")
        audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.ascontiguousarray(audio, dtype=np.float32)
    except AudioError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AudioError(f"Ошибка декодирования (PyAV): {exc}") from exc
    finally:
        try:
            container.close()
        except Exception:  # noqa: BLE001
            pass
