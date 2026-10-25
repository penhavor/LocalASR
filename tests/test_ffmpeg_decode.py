# -*- coding: utf-8 -*-
"""Regression test for the ffmpeg decode path.

History: inside `_decode_with_ffmpeg` the unpacking
`_, stderr = proc.communicate(...)` shadowed the i18n helper `_()`, so the
function raised UnboundLocalError on EVERY call and every file silently fell
back to PyAV. This test decodes a generated sine-wave WAV straight through
the ffmpeg path and fails on any exception.

Requires: numpy + an ffmpeg binary (system PATH or the imageio-ffmpeg
package). Without ffmpeg the module SKIPs (exit 0).

Run:  cd <project root> && python tests/test_ffmpeg_decode.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import wave
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("LOCALASR_QUIET", "1")

import numpy as np  # noqa: E402

from app.core.ffmpeg import (  # noqa: E402
    _decode_with_ffmpeg,
    decode_audio,
    find_ffmpeg,
)

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


def skip(message: str) -> None:
    print(f"  - SKIP: {message}")


def make_sine_wav(path: Path, seconds: float = 2.0, sr: int = 44100) -> None:
    """Stdlib-generated stereo 16-bit WAV — also exercises resampling to 16k mono."""
    frames = bytearray()
    for i in range(int(seconds * sr)):
        value = int(12000 * math.sin(2 * math.pi * 440 * i / sr))
        frames += value.to_bytes(2, "little", signed=True) * 2  # two channels
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(bytes(frames))


def main() -> int:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        skip("no ffmpeg binary found (system PATH or imageio-ffmpeg)")
        return 0

    print("== ffmpeg decode path (the '_' shadowing regression) ==")
    expected_samples = int(2.0 * 16000)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "sine.wav"
        make_sine_wav(wav)

        # 1) the function that used to die with UnboundLocalError on EVERY call
        audio = None
        try:
            audio = _decode_with_ffmpeg(ffmpeg, wav, 16000, None, None, 2.0)
            check(True, "_decode_with_ffmpeg runs without UnboundLocalError")
        except Exception as exc:  # noqa: BLE001
            check(False, f"_decode_with_ffmpeg must not raise: {type(exc).__name__}: {exc}")

        if audio is not None:
            check(audio.dtype == np.float32, "returns float32 samples")
            check(abs(audio.size - expected_samples) <= 1600,
                  f"~2 s resampled to 16 kHz mono ({audio.size} samples)")
            check(bool(np.all(np.isfinite(audio))), "no NaN/inf samples")
            check(float(np.abs(audio).max()) > 0.01, "audio is not silent")

        # 2) the public entry point must succeed through ffmpeg as well
        try:
            audio2 = decode_audio(wav)
            check(abs(audio2.size - expected_samples) <= 1600,
                  "decode_audio returns the same ~2 s of audio")
        except Exception as exc:  # noqa: BLE001
            check(False, f"decode_audio must not raise: {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("FFMPEG DECODE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
