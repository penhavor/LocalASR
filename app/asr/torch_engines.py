"""PyTorch/transformers engines: Qwen3-ASR (Alibaba) and GigaAM (Sber).

Both models are downloaded from Hugging Face into models/ as regular
transformers repos; inference runs locally via torch.

* Qwen3-ASR — multimodal model (audio encoder + Qwen3-LM): 30 languages,
  language autodetection, built-in punctuation. Works through
  ``AutoProcessor.apply_transcription_request`` + ``generate``.
* GigaAM — Sber's conformer model (CTC/RNNT, best-in-class for Russian).
  Loaded via ``AutoModel.from_pretrained(..., trust_remote_code=True)``.
  The stock ``transcribe`` only handles <=25 s and spawns ffmpeg via
  subprocess on every call (flashing consoles on Windows), so long audio is
  chunked locally and computed the FAST way: encoder batches, multi-threaded
  CPU RNNT decoding (no per-token GPU<->CPU sync), no temporary WAV files.
  A non-standard model structure falls back to the stock per-chunk path with
  a patched audio loader (no subprocess/ffmpeg).

Both engines feed on silence-split chunks to avoid eating gigabytes of RAM
on a 3-hour lecture and to report live progress and support cancellation.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import tempfile
import time
import types
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..constants import ENGINE_GIGAAM, ENGINE_QWEN_ASR, SAMPLE_RATE
from ..core.postprocess import Segment
from ..i18n import _
from ..logutils import get_logger
from .base import (
    ASREngine,
    CancelToken,
    EngineError,
    ModelRef,
    ProgressCallback,
    SegmentCallback,
    TranscriptResult,
)

log = get_logger("asr.torch")

# Maximum chunk length per engine (seconds)
QWEN_CHUNK_SECONDS = 120.0     # Qwen3-ASR handles long windows, but 120 s is
#                                a safe cap for consumer GPUs (6 GB VRAM):
#                                on OOM the chunk is split in half (_infer_qwen)
QWEN_SPLIT_FLOOR_SECONDS = 5.0 # OOM-split floor: shorter chunks are not split
GIGAAM_CHUNK_SECONDS = 20.0    # stock GigaAM transcribe is limited to ~25 s

# Do not feed chunks shorter than this: on <1 s scraps seq2seq models
# hallucinate ("Thanks for watching") and CTC/RNNT emit junk tokens. A short
# tail is merged into the previous chunk (_merge_short_tail) — no content lost.
MIN_CHUNK_SECONDS = 1.0

# GigaAM batching: ~20 s chunks per encoder forward group
GIGAAM_DEFAULT_BATCH = 8
GIGAAM_MAX_BATCH = 16


# --------------------------------------------------------------------------- #
#  Silence-based audio chunking
# ---------------------------------------------------------------------------


def _moving_energy_min_index(segment: np.ndarray, win: int) -> int:
    """Index of the lowest-energy (RMS) window — O(n) via cumulative sums.

    Replaces np.convolve(segment^2, ones(win)/win, 'valid'): the naive
    convolution costs O(n*win). argmin of the window sum equals argmin RMS,
    so the result is identical.
    """
    if win <= 0 or segment.size <= win:
        return -1
    square = np.square(segment, dtype=np.float32)
    cumulative = np.cumsum(square)
    # sums[i] = sum of square[i:i+win] — same as a 'valid' convolution
    sums = cumulative[win - 1:]
    sums[1:] -= cumulative[:-win]
    return int(np.argmin(sums))


def _merge_short_tail(
    bounds: List[Tuple[int, int]],
    sample_rate: int,
    chunk_seconds: float,
    search_seconds: float,
) -> List[Tuple[int, int]]:
    """Merge a short tail chunk (<1 s) into the previous one.

    The last chunk can be a few hundred ms — exactly the input that makes
    models invent "thanks for watching" and junk tokens. Merging is allowed
    while the combined chunk stays under `chunk + search + MIN_CHUNK`
    (HF: max 29.5 s < Whisper's 30 s window; GigaAM: max 22 s < ~25 s limit).
    If merging would exceed the limit, the tail stays a separate chunk
    (losing content is worse than a small artifact on a short fragment).
    """
    if len(bounds) < 2:
        return bounds
    min_samples = int(MIN_CHUNK_SECONDS * sample_rate)
    limit_samples = int((chunk_seconds + search_seconds + MIN_CHUNK_SECONDS) * sample_rate)
    start, end = bounds[-1]
    if end - start >= min_samples:
        return bounds
    prev_start, _prev_end = bounds[-2]
    if end - prev_start <= limit_samples:
        return [*bounds[:-2], (prev_start, end)]
    return bounds


def iter_silence_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    search_seconds: float = 2.0,
    window_seconds: float = 0.4,
) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) bounds in samples: chunks up to chunk_seconds with
    boundaries searched within +-search_seconds at the volume minimum."""
    total = len(audio)
    if total == 0:
        return
    win = max(1, int(window_seconds * sample_rate))
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        target = start + int(chunk_seconds * sample_rate)
        if target >= total:
            end = total
        else:
            lo = max(start + win, start + int((chunk_seconds - search_seconds) * sample_rate))
            hi = min(total, start + int((chunk_seconds + search_seconds) * sample_rate))
            end = target
            if hi - lo > win:
                idx = _moving_energy_min_index(audio[lo:hi], win)
                if idx >= 0:
                    end = lo + idx + win // 2
            if end <= start:
                end = target
        bounds.append((start, min(end, total)))
        start = min(end, total)
    yield from _merge_short_tail(bounds, sample_rate, chunk_seconds, search_seconds)


def _vad_pause_points(
    audio: np.ndarray,
    sample_rate: int,
    min_pause_ms: int = 300,
) -> Optional[List[int]]:
    """Speech pause centers (in samples) from Silero VAD.

    None when VAD is unavailable (no faster-whisper/onnxruntime, model
    error): the caller falls back to energy-based chunking. The
    silero_vad_v6.onnx model ships inside faster-whisper — no internet.
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception:  # noqa: BLE001
        return None
    try:
        options = VadOptions(
            min_speech_duration_ms=100,
            min_silence_duration_ms=min_pause_ms,
            speech_pad_ms=0,          # want an honest boundary, no padding
        )
        speeches = get_speech_timestamps(audio, vad_options=options, sampling_rate=sample_rate)
    except Exception as exc:  # noqa: BLE001
        log.debug("Silero VAD failed — will cut at the volume minimum: %s", exc)
        return None
    points: List[int] = []
    for previous, following in zip(speeches, speeches[1:]):
        gap_start = int(previous.get("end") or 0)
        gap_end = int(following.get("start") or 0)
        if gap_end > gap_start:
            points.append((gap_start + gap_end) // 2)
    return points


def iter_smart_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    search_seconds: float = 1.0,
    use_vad: bool = True,
    min_pause_ms: int = 200,
) -> Iterator[Tuple[int, int]]:
    """Chunk boundaries that do NOT cut words in half.

    Preference: pause centers found by Silero VAD (the latest pause in
    [half chunk, chunk+search]). Without VAD or without pauses in the
    window (continuous speech/music) — the volume minimum. Greatly improves
    text on hard recordings (music, noise, fast pauseless speech).
    """
    total = len(audio)
    if total == 0:
        return
    points = _vad_pause_points(audio, sample_rate, min_pause_ms) if use_vad else None
    if not points:
        yield from iter_silence_chunks(audio, sample_rate, chunk_seconds, search_seconds=search_seconds)
        return

    import bisect

    max_len = int(chunk_seconds * sample_rate)
    search = int(search_seconds * sample_rate)
    win = max(1, int(0.4 * sample_rate))
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        target = start + max_len
        if target >= total:
            bounds.append((start, total))
            break
        lo = start + max_len // 2
        hi = min(total, target + search)
        index_lo = bisect.bisect_left(points, lo)
        index_hi = bisect.bisect_right(points, hi)
        end = 0
        if index_hi > index_lo:
            end = points[index_hi - 1]        # the latest pause in the window
        else:
            # continuous speech: energy minimum in a narrow window near target
            seg_lo = max(start + win, target - search)
            seg_hi = min(total, target + search)
            if seg_hi - seg_lo > win:
                idx = _moving_energy_min_index(audio[seg_lo:seg_hi], win)
                if idx >= 0:
                    end = seg_lo + idx + win // 2
            if end <= start:
                end = target
        end = max(end, start + 1)
        bounds.append((start, min(end, total)))
        start = min(end, total)
    yield from _merge_short_tail(bounds, sample_rate, chunk_seconds, search_seconds)


def _build_gigaam_manual(mod: Any, config: Any, folder: Path) -> Any:
    """Assemble GigaAM manually: init on cpu + explicit weight loading.

    Works around transformers v5 meta-initialization where hydra internals
    (mel filters, RNNT registers) are created on cpu and hit a device clash.
    """
    import torch

    config.name_or_path = str(folder)
    model = mod.GigaAMModel(config)
    weights = folder / "pytorch_model.bin"
    safetensors = folder / "model.safetensors"
    if safetensors.is_file():
        from safetensors.torch import load_file

        state = load_file(str(safetensors))
    elif weights.is_file():
        try:
            # mmap noticeably lowers peak memory on large models
            state = torch.load(str(weights), map_location="cpu", weights_only=True, mmap=True)
        except (RuntimeError, OSError, ValueError):
            state = torch.load(str(weights), map_location="cpu", weights_only=True)
    else:
        raise EngineError(_("Neither pytorch_model.bin nor model.safetensors in {folder}").format(folder=folder))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # base_model_prefix may differ — retry after stripping "model."
    if unexpected and all(key.startswith("model.") for key in unexpected):
        state = {key[len("model."):]: value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise EngineError(_(
            "GigaAM weights are missing {n} tensors (e.g. {examples})"
        ).format(n=len(missing), examples=missing[:3]))
    return model


def decode_item_text(item: Any) -> str:
    """Extract text from a decoder result.

    Recent modeling_gigaam.py returns a ``(text, token_ids, token_frames)``
    tuple from CTC/RNNT decode; older ones a bare string. Take the first
    string element of a tuple/list; a string passes through.
    """
    if isinstance(item, (tuple, list)):
        for part in item:
            if isinstance(part, str):
                return part
        return ""
    return str(item or "")


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 -> WAV s16le (stdlib only)."""
    pcm = np.clip(audio, -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(data.tobytes())


def missing_files_hint(exc: BaseException) -> str:
    """Hint for "model folder lacks file X" errors.

    Typical cause: an interrupted download. Explains what to do instead of
    showing a bare traceback.
    """
    low = str(exc).lower()
    markers = (
        "preprocessor_config.json", "config.json", "tokenizer", "does not appear to have a file named",
        "can't load", "cannot load", "is not a local folder", "no such file or directory",
        "feature extractor", "processor_config.json",
    )
    if any(marker in low for marker in markers):
        return _(
            "\n\nThe model files appear incomplete or the folder is damaged "
            "(this happens when the network drops during installation).\n"
            "What to do: delete the model (model card → Delete) and install it again — "
            "complete files will be skipped, only the missing ones re-downloaded."
        )
    return ""


def transformers_version_hint(exc: BaseException, model_kind: str, min_version: str) -> str:
    """Hint when the installed transformers is too old for the model."""
    low = str(exc).lower()
    if any(marker in low for marker in (
        "unrecognized", "not recognized", "unknown model", "keyerror", "does not support",
        "has no attribute", "invalid model type",
    )):
        return _(
            "\n\nThe installed transformers does not know \"{kind}\" — transformers>={ver} is required.\n"
            "Update it: the app will offer the installation on the next model run "
            "(or: Tools → Diagnostics & recovery).\n"
            "IMPORTANT: if the new transformers version was just installed, RESTART the "
            "app — an already imported old version stays in process memory until restart."
        ).format(kind=model_kind, ver=min_version)
    return ""


def reset_transformers_caches() -> int:
    """Reset the "package available?" lru caches inside transformers.

    transformers caches is_accelerate_available()/is_torch_available() results
    for the whole process lifetime. When a package is installed WHILE the app
    is running (the normal install-on-demand flow), the cache keeps answering
    "unavailable". Clearing the caches in all loaded transformers modules
    refreshes the checks WITHOUT restarting. Returns the number cleared.
    """
    cleared = 0
    seen: set = set()
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if not (name == "transformers" or name.startswith("transformers.")):
            continue
        try:
            attributes = list(vars(module).values())
        except TypeError:
            continue
        for attr in attributes:
            # Skip classes and modules: lazy transformers objects try to
            # import backends on ANY getattr and build requires_backends
            # messages (seconds per attribute). All needed lru_caches are
            # functions; walking without the others takes ~0.1 s.
            if isinstance(attr, (type, types.ModuleType)):
                continue
            try:
                if id(attr) in seen:
                    continue
                cache_clear = getattr(attr, "cache_clear", None)
                if cache_clear is None or not callable(getattr(attr, "cache_info", None)):
                    continue    # not an lru_cache wrapper
            except Exception:  # noqa: BLE001
                continue
            seen.add(id(attr))
            try:
                cache_clear()
                cleared += 1
            except Exception:  # noqa: BLE001
                continue
    if cleared:
        log.debug("transformers: availability lru caches cleared: %d", cleared)
    return cleared


def read_wav_mono(path: Path, sample_rate: int):
    """Read WAV s16le into a float32 torch tensor (no ffmpeg/subprocess)."""
    import torch

    with wave.open(str(path), "rb") as fh:
        params_ok = fh.getnchannels() == 1 and fh.getsampwidth() == 2
        raw = fh.readframes(fh.getnframes())
    if not params_ok:
        raise ValueError(_("Expected a mono s16le WAV: {path}").format(path=path))
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


def safe_load_audio(path: Any, sample_rate: int = SAMPLE_RATE):
    """Safe replacement for gigaam.load_audio: NO subprocess/ffmpeg consoles.

    The model's native load_audio runs ``ffmpeg`` via subprocess.run without
    CREATE_NO_WINDOW — from pythonw.exe that flashes a black console per audio
    chunk. Here: WAV via stdlib wave, everything else via our decoder (which
    launches ffmpeg invisibly).
    """
    path = Path(str(path))
    if path.suffix.lower() == ".wav":
        try:
            return read_wav_mono(path, sample_rate)
        except Exception:  # noqa: BLE001 - not our WAV (different sample format)
            pass
    from ..core.ffmpeg import decode_audio

    import torch

    audio = decode_audio(path, sample_rate)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


# --------------------------------------------------------------------------- #
#  Shared base class for torch engines
# --------------------------------------------------------------------------- #


class TorchASREngine(ASREngine):
    """Shared: availability check, device selection, memory release."""

    chunk_seconds: float = 60.0
    #: all modules without which the engine will not start (GUI offers install)
    import_names: Tuple[str, ...] = ("torch", "transformers")

    @classmethod
    def is_available(cls) -> bool:
        return all(importlib.util.find_spec(name) is not None for name in cls.import_names)

    # -- helpers --------------------------------------------------------------- #
    def _pick_device(self, hardware: Dict[str, Any]) -> Tuple[str, Any]:
        import torch

        wanted = str(hardware.get("device") or "auto")
        # torch ships the CUDA runtime in its pip wheel, so the ctranslate2
        # check (hardware["cuda_ok"]) does not apply — torch.cuda.is_available()
        # is sufficient.
        if wanted == "cuda":
            use_cuda = torch.cuda.is_available()
        elif wanted == "cpu":
            use_cuda = False
        else:  # auto
            use_cuda = bool(torch.cuda.is_available())
        self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32
        self.compute_type = "float16" if use_cuda else "float32"
        return self.device, dtype

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self.loaded_model_id = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _ensure_model(self) -> Any:
        if self._model is None:
            raise EngineError(_("Model of engine {name} is not loaded").format(name=self.name))
        return self._model

    def __init__(self) -> None:
        super().__init__()
        self._processor: Any = None


# --------------------------------------------------------------------------- #
#  Qwen3-ASR
# --------------------------------------------------------------------------- #


class QwenASREngine(TorchASREngine):
    name = ENGINE_QWEN_ASR
    # qwen3_asr appeared in transformers 5.13.0 — earlier 5.x do NOT know the
    # model, hence the strict pin
    packages = ["torch>=2.6", "transformers>=5.13", "accelerate>=1.0", "soundfile>=0.12"]
    import_name = "transformers"
    import_names = ("torch", "transformers", "accelerate")
    chunk_seconds = QWEN_CHUNK_SECONDS
    MIN_TRANSFORMERS = (5, 13)

    # ------------------------------------------------------------------ #
    @classmethod
    def is_available(cls) -> bool:
        """Available = packages present AND transformers knows qwen3_asr.

        The version is checked via importlib.metadata WITHOUT importing
        transformers: the import is heavy and is_available() is called from
        the GUI (Engines tab, RUN button) — it must not hang the window.
        """
        if not super().is_available():
            return False
        try:
            from importlib.metadata import version as pkg_version

            raw = pkg_version("transformers")
        except Exception:  # noqa: BLE001
            return False
        parts = []
        for chunk in raw.split(".")[:2]:
            digits = "".join(ch for ch in chunk if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        while len(parts) < 2:
            parts.append(0)
        return tuple(parts) >= cls.MIN_TRANSFORMERS

    # ------------------------------------------------------------------ #
    @staticmethod
    def _from_pretrained_dtype(auto_cls: Any, path: str, dtype: Any, **kwargs: Any) -> Any:
        """from_pretrained accounting for the torch_dtype -> dtype rename in 5.10+."""
        try:
            return auto_cls.from_pretrained(path, dtype=dtype, **kwargs)
        except TypeError:
            # transformers < 5.10: the argument was called torch_dtype
            return auto_cls.from_pretrained(path, torch_dtype=dtype, **kwargs)

    def _load_qwen_model(self, auto_cls: Any, path: str, dtype: Any, device: str) -> Any:
        """Weight loading with a triple safety net for accelerate quirks.

        Cascade: 1) device_map=device (fast path, needs accelerate);
        2) on "requires accelerate" — reset transformers lru caches (the
        package may have just been installed while the app runs) and retry;
        3) if accelerate is truly absent — load WITHOUT device_map and use
        plain .to(device): identical result for a single GPU.
        """
        if importlib.util.find_spec("accelerate") is not None:
            try:
                return self._from_pretrained_dtype(auto_cls, path, dtype, device_map=device)
            except ValueError as exc:
                if "accelerate" not in str(exc).lower():
                    raise
                cleared = reset_transformers_caches()
                log.warning(
                    _("transformers did not see the installed accelerate (stale lru_cache, "
                      "caches cleared: %d) — retrying the load"), cleared,
                )
                try:
                    return self._from_pretrained_dtype(auto_cls, path, dtype, device_map=device)
                except ValueError as exc2:
                    if "accelerate" not in str(exc2).lower():
                        raise
                    log.warning(_("accelerate still unavailable — loading without device_map (.to(%s))"), device)
        else:
            log.info(_("accelerate is not installed — loading Qwen3-ASR without device_map (.to(%s))"), device)
        # fallback path WITHOUT accelerate: load on CPU, then move to device
        model = self._from_pretrained_dtype(auto_cls, path, dtype)
        return model.to(device)

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except Exception as exc:  # noqa: BLE001
            raise EngineError(_(
                "Failed to import torch/transformers: {exc}\n"
                "Missing components are offered for automatic installation when a model "
                "is started (or: Tools → Diagnostics & recovery)."
            ).format(exc=exc)) from exc

        path = str(model.resolved_path)
        device, dtype = self._pick_device(hardware)
        log.debug("Qwen3-ASR: loading %s (device=%s, dtype=%s)", path, device, dtype)
        started = time.time()
        try:
            self._processor = AutoProcessor.from_pretrained(path)
            self._model = self._load_qwen_model(AutoModelForMultimodalLM, path, dtype, device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            hint = missing_files_hint(exc) or transformers_version_hint(exc, "Qwen3-ASR", "5.13")
            log.error("Qwen3-ASR failed to load from %s: %s", path, exc, exc_info=True)
            raise EngineError(_("Failed to load Qwen3-ASR from {path}: {exc}{hint}").format(
                path=path, exc=exc, hint=hint)) from exc
        self.loaded_model_id = model.model_id
        log.info(
            _("Qwen3-ASR loaded: %s (%s, %s) in %.1f s"),
            model.model_id, device, self.compute_type, time.time() - started,
        )

    # ------------------------------------------------------------------ #
    def _qwen_inputs(self, chunk: np.ndarray, language: Optional[str], prompt: Optional[str]) -> Any:
        """Prepare model inputs via apply_transcription_request.

        IMPORTANT: audio is passed as a 16 kHz ARRAY (transformers 5.x
        make_list_of_audio accepts a numpy/torch array, path or URL). The
        "(array, sample_rate) tuple" style from old examples triggers
        "Invalid input type. Must be a single audio…".
        """
        apply = getattr(self._processor, "apply_transcription_request", None)
        if apply is None:
            raise EngineError(_(
                "The installed transformers does not support apply_transcription_request "
                "for Qwen3-ASR — update transformers (>= 5.13 required)."
            ))
        try:
            return apply(audio=chunk, language=language, prompt=prompt)
        except TypeError as exc:
            # guard against signature changes between transformers versions
            text = str(exc)
            if "prompt" in text:
                log.debug("apply_transcription_request without prompt: %s", exc)
                return apply(audio=chunk, language=language)
            if "language" in text:
                log.debug("apply_transcription_request without language: %s", exc)
                return apply(audio=chunk)
            raise

    def _infer_qwen_once(
        self, model: Any, chunk: np.ndarray, language: Optional[str], prompt: Optional[str],
    ) -> Tuple[str, str]:
        """One forward+generate per chunk. Returns (text, language)."""
        import torch

        inputs = self._qwen_inputs(chunk, language, prompt)
        inputs = inputs.to(model.device, model.dtype)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        decode = getattr(self._processor, "decode", None)
        if decode is None:
            raise EngineError(_("Qwen3-ASR processor has no decode() — update transformers (>= 5.13)"))
        try:
            parsed = decode(generated, return_format="parsed")[0]
        except (TypeError, ValueError):
            # very old/new API variants: decode raw and strip markers
            raw = decode(generated, skip_special_tokens=True)
            text = raw[0] if isinstance(raw, (list, tuple)) else str(raw)
            parsed = {"transcription": text, "language": ""}
        if isinstance(parsed, (list, tuple)):
            parsed = parsed[0] if parsed else {}
        if not isinstance(parsed, dict):
            parsed = {"transcription": str(parsed), "language": ""}
        return str(parsed.get("transcription") or "").strip(), str(parsed.get("language") or "")

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        try:
            import torch

            oom_class = getattr(torch.cuda, "OutOfMemoryError", None)
            if oom_class is not None and isinstance(exc, oom_class):
                return True
        except Exception:  # noqa: BLE001
            pass
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _infer_qwen(
        self,
        model: Any,
        chunk: np.ndarray,
        language: Optional[str],
        prompt: Optional[str],
        offset_seconds: float,
    ) -> Tuple[str, str]:
        """Recognize a chunk; on VRAM shortage split it in half (recursively).

        Consumer GPUs (6 GB VRAM) do not always fit the 120 s Qwen3-ASR
        encoder window — instead of a "CUDA out of memory" crash the chunk is
        halved and both parts are computed separately (texts are joined).
        Split floor: QWEN_SPLIT_FLOOR_SECONDS.
        """
        import torch

        try:
            return self._infer_qwen_once(model, chunk, language, prompt)
        except Exception as exc:  # noqa: BLE001
            if not self._is_cuda_oom(exc):
                raise
            floor_samples = int(QWEN_SPLIT_FLOOR_SECONDS * SAMPLE_RATE)
            if len(chunk) < floor_samples * 2:
                raise
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            mid = len(chunk) // 2
            log.warning(
                _("Qwen3-ASR: not enough VRAM for a %.1f s chunk (at %.1f s) — splitting in half and retrying"),
                len(chunk) / SAMPLE_RATE, offset_seconds,
            )
            left = self._infer_qwen(model, np.ascontiguousarray(chunk[:mid]), language, prompt, offset_seconds)
            right = self._infer_qwen(
                model, np.ascontiguousarray(chunk[mid:]), language, prompt,
                offset_seconds + mid / SAMPLE_RATE,
            )
            text = (left[0] + (" " if left[0] and right[0] else "") + right[0]).strip()
            detected = left[1] or right[1]
            return text, detected

    # ------------------------------------------------------------------ #
    def transcribe(
        self,
        audio: Any,
        *,
        options: Dict[str, Any],
        duration: float,
        progress: Optional[ProgressCallback] = None,
        on_segment: Optional[SegmentCallback] = None,
        cancel: Optional[CancelToken] = None,
    ) -> TranscriptResult:
        model = self._ensure_model()
        started = time.time()
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        total = len(audio) or 1

        language = (options.get("language") or "").strip() or None
        prompt_parts = [options.get("initial_prompt") or "", options.get("hotwords") or ""]
        prompt = " ".join(part.strip() for part in prompt_parts if part.strip()) or None

        # pause-based chunking (VAD, like the other engines): chunks do not
        # cut words in half, short tails merge into the previous chunk
        bounds = list(
            iter_smart_chunks(
                audio, SAMPLE_RATE, self.chunk_seconds, search_seconds=2.0,
                use_vad=bool(options.get("vad", True)),
            )
        )
        log.debug(
            "Qwen3-ASR: %d chunks (language=%s, prompt=%s, audio %.1f s)",
            len(bounds), language or "auto", bool(prompt), duration,
        )

        segments: List[Segment] = []
        language_detected = ""
        warnings: List[str] = []

        for start, end in bounds:
            if cancel and cancel():
                warnings.append(_("Recognition stopped by user"))
                break
            chunk = audio[start:end]
            chunk_started = time.time()
            try:
                text, detected = self._infer_qwen(model, chunk, language, prompt, start / SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001
                hint = ""
                if self._is_cuda_oom(exc):
                    hint = _(
                        "\n\nNot enough VRAM even for the minimal chunk. "
                        "Try the Qwen3-ASR 0.6B model or another device "
                        "(Tools → Settings → \"Speed & hardware\")."
                    )
                log.error(
                    "Qwen3-ASR failed on chunk %.1f–%.1f s: %s",
                    start / SAMPLE_RATE, end / SAMPLE_RATE, exc, exc_info=True,
                )
                raise EngineError(_("Qwen3-ASR could not recognize the chunk: {exc}{hint}").format(
                    exc=exc, hint=hint)) from exc

            language_detected = detected or language_detected
            log.debug(
                "Qwen3-ASR: chunk %.1f–%.1f s -> %d chars in %.1f s",
                start / SAMPLE_RATE, end / SAMPLE_RATE, len(text), time.time() - chunk_started,
            )
            if text:
                segment = Segment(start=start / SAMPLE_RATE, end=end / SAMPLE_RATE, text=text)
                segments.append(segment)
                if on_segment:
                    on_segment(segment)

            if progress:
                progress(end / total, _("Recognizing: {done:.0f} s of {total:.0f} s").format(
                    done=end / SAMPLE_RATE, total=duration))

        if options.get("word_timestamps"):
            warnings.append(_("Qwen3-ASR provides no word timestamps — segments follow audio chunks"))

        return TranscriptResult(
            segments=segments,
            language=language_detected,
            audio_duration=duration,
            processed_duration=duration,
            elapsed_seconds=time.time() - started,
            engine=self.name,
            device=self.device,
            compute_type=self.compute_type,
            warnings=warnings,
        )


# --------------------------------------------------------------------------- #
#  GigaAM (Sber)
# --------------------------------------------------------------------------- #


def _rnnt_greedy_one(
    enc_frames: Any,
    length: int,
    decoder: Any,
    pred: Any,
    joint_net: Any,
    blank_id: int,
    max_symbols: int,
    tokenizer: Any,
) -> str:
    """RNNT greedy decoding of ONE sequence on CPU.

    Mathematically repeats GigaAM RNNTGreedyDecoding._greedy_decode, but:
      * the encoder projection (joint.enc) is already computed on GPU;
      * LSTM/joint run on CPU in float32 — no per-token GPU<->CPU sync
        (the main bottleneck of the native path);
      * safe to run in parallel across threads.
    """
    import torch

    hyp: List[int] = []
    state: Any = None
    last_label: Any = None
    with torch.inference_mode():
        for t in range(length):
            frame = enc_frames[t].view(1, 1, -1)
            not_blank = True
            emitted = 0
            while not_blank and emitted < max_symbols:
                g, hidden = decoder.predict(last_label, state)
                logits = joint_net(frame + pred(g))
                token = int(logits.argmax(dim=-1).item())
                if token == blank_id:
                    not_blank = False
                    # IMPORTANT: on blank the LSTM state is NOT updated — exactly
                    # like the native _greedy_decode (dec_state = hidden only on
                    # token emission). Otherwise decoding "eats" syllables.
                else:
                    hyp.append(token)
                    state = hidden
                    last_label = torch.tensor([[token]])
                    emitted += 1
    return str(tokenizer.decode(hyp) or "")


class GigaAMEngine(TorchASREngine):
    name = ENGINE_GIGAAM
    packages = [
        "torch>=2.6",
        "torchaudio>=2.6",
        "transformers>=4.52",
        "hydra-core>=1.3,<2",
        "omegaconf>=2.3,<3",
        "sentencepiece>=0.2.2",
    ]
    import_name = "transformers"
    # soundfile is checked explicitly: the model code (modeling_gigaam.py)
    # imports it; without it loading dies with ModuleNotFoundError
    import_names = ("torch", "transformers", "torchaudio", "hydra", "omegaconf", "sentencepiece", "soundfile")
    chunk_seconds = GIGAAM_CHUNK_SECONDS

    def __init__(self) -> None:
        super().__init__()
        self._rnnt_kit: Optional[Tuple[Any, ...]] = None
        self._fast_batch = GIGAAM_DEFAULT_BATCH

    # ------------------------------------------------------------------ #
    def load(self, model: ModelRef, hardware: Dict[str, Any]) -> None:
        self.ensure_available()
        try:
            from transformers import AutoModel
        except Exception as exc:  # noqa: BLE001
            raise EngineError(_(
                "Failed to import transformers: {exc}\n"
                "Missing components are offered for automatic installation when a model "
                "is started (or: Tools → Diagnostics & recovery)."
            ).format(exc=exc)) from exc

        # GigaAM code calls ffmpeg from PATH internally — point it to our
        # binary (only the fallback path needs it; the fast path does not)
        try:
            from ..core.ffmpeg import find_ffmpeg

            ffmpeg = find_ffmpeg()
            if ffmpeg:
                extra = str(Path(ffmpeg).parent)
                current = os.environ.get("PATH", "")
                if extra not in current:
                    os.environ["PATH"] = extra + os.pathsep + current
        except Exception:  # noqa: BLE001
            pass

        path = str(model.resolved_path)
        device, dtype = self._pick_device(hardware)
        # IMPORTANT: dtype is NOT passed to from_pretrained — model internals
        # (mel filters etc.) are created by hydra on cpu and clash with
        # transformers meta-initialization. Precision is applied after load.
        try:
            loaded = AutoModel.from_pretrained(
                path, trust_remote_code=True, low_cpu_mem_usage=False
            )
        except ImportError as exc:
            # modeling_gigaam.py lazily imports pyannote (only for their
            # longform with an HF token). We do not need longform: long audio
            # is chunked locally, so load the module directly.
            log.info("AutoModel requires %s — loading modeling_gigaam.py directly", exc)
            loaded = self._load_direct(path, dtype)
        except Exception as exc:  # noqa: BLE001
            try:
                loaded = self._load_direct(path, dtype)
            except Exception as direct_exc:  # noqa: BLE001
                log.error("GigaAM failed to load from %s: %s / %s", path, exc, direct_exc, exc_info=True)
                raise EngineError(_(
                    "Failed to load GigaAM from {path}: {exc} / {direct}\n"
                    "Required packages: torch, torchaudio, transformers, hydra-core, omegaconf, sentencepiece."
                ).format(path=path, exc=exc, direct=direct_exc)
                    + missing_files_hint(direct_exc)
                ) from exc
        try:
            loaded = loaded.to(device)
            # IMPORTANT: do NOT cast GPU weights to fp16. Native GigaAM.forward
            # enables torch.autocast(fp16) around the encoder itself, and its
            # rotary PE buffer is always fp32: with fp16 weights q(fp16)*cos(fp32)
            # yields fp32 and linear_q crashes with a dtype mismatch.
            # fp32 weights + autocast = the vendor's scheme: fast and crash-free.
            if device == "cuda":
                self.compute_type = "float32+autocast(fp16)"
            loaded.eval()
        except Exception as exc:  # noqa: BLE001
            raise EngineError(_("Failed to move GigaAM to {device}: {exc}").format(device=device, exc=exc)) from exc
        self._model = loaded
        self._rnnt_kit = None
        self._fast_batch = GIGAAM_DEFAULT_BATCH
        self._patch_audio_loader()
        self.loaded_model_id = model.model_id
        log.info(_("GigaAM loaded: %s (%s, %s)"), model.model_id, device, self.compute_type)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_direct(path: str, dtype: Any = None) -> Any:
        """Load modeling_gigaam.py directly (no dynamic-module check)."""
        import importlib.util as iu

        module_path = Path(path) / "modeling_gigaam.py"
        if not module_path.is_file():
            raise EngineError(_("No modeling_gigaam.py in {path}").format(path=path))
        spec = iu.spec_from_file_location("localasr_gigaam_modeling", str(module_path))
        if spec is None or spec.loader is None:
            raise EngineError(_("Failed to read {path}").format(path=module_path))
        mod = iu.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
            config = mod.GigaAMConfig.from_pretrained(path)
            try:
                # low_cpu_mem_usage=False: otherwise transformers creates the
                # model on the meta device while hydra internals (mel filters)
                # are on cpu, and loading dies with "Tensor on device cpu is not on meta"
                return mod.GigaAMModel.from_pretrained(
                    path, config=config, low_cpu_mem_usage=False
                )
            except Exception as exc:  # noqa: BLE001
                # transformers v5 may still push init to the meta device:
                # assemble the model manually and apply weights ourselves
                log.info("from_pretrained failed (%s) — assembling GigaAM manually", exc)
                return _build_gigaam_manual(mod, config, Path(path))
        except Exception:
            sys.modules.pop(spec.name, None)
            raise

    # ------------------------------------------------------------------ #
    def _patch_audio_loader(self) -> None:
        """Replace load_audio in ALL copies of the model module (no ffmpeg consoles).

        modeling_gigaam.py does ``sys.path.append(DIR_NAME)`` and hydra then
        re-imports ``modeling_gigaam`` — a second module copy whose load_audio
        (subprocess+ffmpeg) stayed unpatched and flashed a console per chunk.
        Patch every copy found: the outer class, the inner (hydra) class and
        all modules with that name.
        """
        patched = 0
        seen: set = set()
        candidates: List[Any] = []
        inner = getattr(self._model, "model", None)
        for obj in (self._model, inner):
            if obj is None:
                continue
            module = sys.modules.get(type(obj).__module__ or "")
            if module is not None:
                candidates.append(module)
        for name, module in list(sys.modules.items()):
            if name == "modeling_gigaam" or name.endswith(".modeling_gigaam"):
                candidates.append(module)
        for module in candidates:
            if id(module) in seen:
                continue
            seen.add(id(module))
            if hasattr(module, "load_audio"):
                module.load_audio = safe_load_audio
                patched += 1
        log.debug("load_audio patched in %d GigaAM modules", patched)

    def unload(self) -> None:
        self._rnnt_kit = None
        super().unload()

    # ------------------------------------------------------------------ #
    #  Fast path: groups + CPU RNNT decoding
    # ------------------------------------------------------------------ #
    def _inner_asr(self) -> Any:
        """The inner GigaAMASR (self._model.model) or None."""
        inner = getattr(self._model, "model", None)
        if inner is None or not hasattr(inner, "forward"):
            return None
        return inner

    def _fast_kind(self, inner: Any) -> str:
        """'rnnt' | 'ctc' | '' — whether the structure supports the fast path."""
        decoding = getattr(inner, "decoding", None)
        head = getattr(inner, "head", None)
        if decoding is None or head is None:
            return ""
        kind = type(decoding).__name__.upper()
        if "RNNT" in kind:
            joint = getattr(head, "joint", None)
            decoder = getattr(head, "decoder", None)
            tokenizer = getattr(decoding, "tokenizer", None)
            if (
                decoder is not None
                and hasattr(decoder, "predict")
                and tokenizer is not None
                and all(hasattr(joint, attr) for attr in ("enc", "pred", "joint_net"))
            ):
                return "rnnt"
            return ""
        if "CTC" in kind:
            if callable(head) and hasattr(decoding, "decode"):
                return "ctc"
        return ""

    def _rnnt_cpu_kit(self, inner: Any) -> Tuple[Any, ...]:
        """CPU float32 copies of the small RNNT nets (decoder/pred/joint_net)."""
        if self._rnnt_kit is not None:
            return self._rnnt_kit
        import torch

        head = inner.head
        with torch.no_grad():
            decoder = copy.deepcopy(head.decoder).cpu().float().eval()
            pred = copy.deepcopy(head.joint.pred).cpu().float().eval()
            joint_net = copy.deepcopy(head.joint.joint_net).cpu().float().eval()

        # Native predict() creates torch.zeros(...) for the first step with the
        # GLOBAL default dtype — if the environment changed it (e.g. to fp16),
        # the zeros mismatch the copy's fp32 weights. Use the weights' dtype.
        kit_dtype = next(decoder.parameters()).dtype
        original_predict = decoder.predict

        def predict(x: Any, state: Any, _dec=decoder, _orig=original_predict, _dt=kit_dtype):
            if x is not None:
                return _orig(x, state)
            import torch as _torch

            emb = _torch.zeros((1, 1, _dec.pred_hidden), dtype=_dt)
            g, hid = _dec.lstm(emb.transpose(0, 1), state)
            return g.transpose(0, 1), hid

        decoder.predict = predict
        self._rnnt_kit = (decoder, pred, joint_net)
        return self._rnnt_kit

    def _forward_single(self, inner: Any, piece: np.ndarray) -> Tuple[Any, Any]:
        """Encoder forward for ONE chunk: (encoded (1,C,T), length (1,)).

        IMPORTANT: one by one, no batch padding. The GigaAM encoder does not
        fully mask padding inside conv modules, so in a batch short chunks are
        "poisoned" by the neighbor's zero tail (measured: up to 70% relative
        error vs single forward) — text drifts. Single forward matches the
        model's native prepare_wav+forward byte for byte.
        """
        import torch

        parameters = next(inner.parameters())
        wav = torch.from_numpy(np.ascontiguousarray(piece)).unsqueeze(0)
        wav = wav.to(device=parameters.device, dtype=parameters.dtype)
        length = torch.tensor([len(piece)], dtype=torch.int64, device=parameters.device)
        return inner.forward(wav, length)

    def _decode_group(self, inner: Any, kind: str, encoded_items: List[Tuple[Any, Any]]) -> List[str]:
        """Decode a list of (encoded, length) — one per chunk.

        RNNT: the joint.enc projection runs on the model device, then the
        greedy loop runs on CPU in several threads (no per-token GPU sync).
        CTC: GigaAM's native vectorized decode per chunk.
        """
        import torch

        if kind == "ctc":
            texts: List[str] = []
            with torch.inference_mode():
                for encoded, encoded_len in encoded_items:
                    for item in inner.decoding.decode(inner.head, encoded, encoded_len):
                        texts.append(decode_item_text(item))
            return texts

        decoding = inner.decoding
        head = inner.head
        blank_id = int(getattr(decoding, "blank_id", len(decoding.tokenizer)))
        max_symbols = int(getattr(decoding, "max_symbols", 10))
        decoder, pred, joint_net = self._rnnt_cpu_kit(inner)

        tasks = []
        with torch.inference_mode():
            for encoded, encoded_len in encoded_items:
                frames = encoded.transpose(1, 2)                  # (1, T, C)
                enc_proj = head.joint.enc(frames).float().cpu()[0]  # (T, joint_hidden)
                tasks.append(
                    (enc_proj, int(encoded_len[0]), decoder, pred, joint_net,
                     blank_id, max_symbols, decoding.tokenizer)
                )

        workers = min(len(tasks), max(1, os.cpu_count() or 1))
        if workers <= 1:
            return [_rnnt_greedy_one(*task) for task in tasks]

        # The LSTM step is tiny: intra-op torch parallelism only gets in the
        # way — disable it for decoding and parallelize across sequences
        # (torch releases the GIL in its ops).
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rnnt") as pool:
                return list(pool.map(lambda task: _rnnt_greedy_one(*task), tasks))
        finally:
            torch.set_num_threads(previous_threads)

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        import torch

        oom_class = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_class is not None and isinstance(exc, oom_class):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    # ------------------------------------------------------------------ #
    def transcribe(
        self,
        audio: Any,
        *,
        options: Dict[str, Any],
        duration: float,
        progress: Optional[ProgressCallback] = None,
        on_segment: Optional[SegmentCallback] = None,
        cancel: Optional[CancelToken] = None,
    ) -> TranscriptResult:
        import torch

        self._ensure_model()
        started = time.time()
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        total = len(audio) or 1

        bounds = list(
            iter_smart_chunks(
                audio, SAMPLE_RATE, self.chunk_seconds, search_seconds=1.0,
                use_vad=bool(options.get("vad", True)),
            )
        )
        segments: List[Segment] = []
        warnings: List[str] = []

        inner = self._inner_asr()
        kind = self._fast_kind(inner) if inner is not None else ""
        if options.get("batch_size"):
            self._fast_batch = max(1, min(GIGAAM_MAX_BATCH, int(options["batch_size"])))
        if options.get("batched") is False:
            self._fast_batch = 1

        used_fast = False
        if kind:
            if kind == "rnnt":
                # prepare the small RNNT net copies for CPU decoding IN
                # ADVANCE (outside inference_mode) so tensors are "normal"
                self._rnnt_cpu_kit(inner)
            try:
                self._transcribe_fast(
                    inner, kind, audio, bounds, total, duration,
                    segments, warnings, progress, on_segment, cancel,
                )
                used_fast = True
            except Exception as exc:  # noqa: BLE001
                if segments:
                    # results exist — do not fail the whole file silently
                    raise EngineError(_("GigaAM: fast path failed: {exc}").format(exc=exc)) from exc
                log.warning(_("GigaAM fast path unavailable (%s) — falling back to sequential"), exc)
                if self._is_oom(exc):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                segments.clear()
                warnings.clear()

        if not used_fast:
            self._transcribe_stock(
                audio, bounds, total, duration, segments, warnings, progress, on_segment, cancel
            )

        if options.get("word_timestamps"):
            warnings.append(_("GigaAM used without word timestamps (chunks mode)"))

        return TranscriptResult(
            segments=segments,
            language="ru",
            audio_duration=duration,
            processed_duration=duration,
            elapsed_seconds=time.time() - started,
            engine=self.name,
            device=self.device,
            compute_type=self.compute_type,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    def _emit(self, segments, start, end, text, on_segment) -> None:
        text = str(text or "").strip()
        if text:
            segment = Segment(start=start / SAMPLE_RATE, end=end / SAMPLE_RATE, text=text)
            segments.append(segment)
            if on_segment:
                on_segment(segment)

    def _report(self, progress, end_sample, total, duration) -> None:
        if progress:
            progress(
                min(1.0, end_sample / max(total, 1)),
                _("Recognizing: {done:.0f} s of {total:.0f} s").format(
                    done=end_sample / SAMPLE_RATE, total=duration),
            )

    def _transcribe_fast(
        self, inner, kind, audio, bounds, total, duration,
        segments, warnings, progress, on_segment, cancel,
    ) -> None:
        """In groups: encoder ONE BY ONE (no padding!), decoding in parallel.

        A group of N chunks: N single encoder forwards (exactly like the
        native path), then all N decoded at once — RNNT in a CPU thread pool.
        The speed comes from parallel decoding without GPU sync; encoder
        batching is off due to incomplete padding masking in the model.
        """
        import torch

        index = 0
        count = len(bounds)
        while index < count:
            if cancel and cancel():
                warnings.append(_("Recognition stopped by user"))
                return
            take = max(1, min(self._fast_batch, count - index))
            group_bounds = bounds[index: index + take]

            encoded_items: List[Tuple[Any, Any]] = []
            for start_sample, end_sample in group_bounds:
                if cancel and cancel():
                    warnings.append(_("Recognition stopped by user"))
                    return
                while True:  # OOM: clear the cache and retry
                    try:
                        with torch.inference_mode():
                            encoded_items.append(
                                self._forward_single(inner, audio[start_sample:end_sample])
                            )
                        break
                    except Exception as exc:  # noqa: BLE001
                        if not self._is_oom(exc):
                            raise
                        log.warning(_("GPU ran out of memory for a chunk — clearing cache and retrying"))
                        try:
                            torch.cuda.empty_cache()
                        except Exception:  # noqa: BLE001
                            pass

            with torch.inference_mode():
                texts = self._decode_group(inner, kind, encoded_items)
            del encoded_items

            for (start_sample, end_sample), text in zip(group_bounds, texts):
                self._emit(segments, start_sample, end_sample, text, on_segment)
            index += len(group_bounds)
            self._report(progress, group_bounds[-1][1], total, duration)

    def _transcribe_stock(
        self, audio, bounds, total, duration, segments, warnings, progress, on_segment, cancel,
    ) -> None:
        """Fallback path: the native model.transcribe() per chunk via WAV.

        load_audio in the model modules is patched to safe_load_audio, so no
        subprocess/ffmpeg is spawned and no console flashes.
        """
        model = self._ensure_model()
        tmp_path: Optional[Path] = None
        try:
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = Path(handle.name)
            handle.close()

            for start, end in bounds:
                if cancel and cancel():
                    warnings.append(_("Recognition stopped by user"))
                    return
                write_wav(tmp_path, audio[start:end], SAMPLE_RATE)
                try:
                    text = model.transcribe(str(tmp_path))
                except Exception as exc:  # noqa: BLE001
                    raise EngineError(_("GigaAM could not recognize the chunk: {exc}").format(exc=exc)) from exc
                self._emit(segments, start, end, text, on_segment)
                self._report(progress, end, total, duration)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
