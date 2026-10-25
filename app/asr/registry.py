"""Engine registry: loaded-model cache and hardware info.

The model is loaded once and reused for the whole queue — the main speed
source when batch-processing a folder of lectures.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from ..constants import (
    ENGINE_FASTER_WHISPER,
    ENGINE_GIGAAM,
    ENGINE_HF,
    ENGINE_QWEN_ASR,
    ENGINE_VOSK,
    ENGINE_WHISPERCPP,
)
from ..core.models_catalog import ModelEntry
from ..i18n import _
from ..logutils import get_logger
from .base import (
    ASREngine,
    EngineError,
    EngineNotInstalled,
    ModelRef,
    engine_not_installed_from_exc,
    engine_packages,
)
from .faster_whisper_engine import FasterWhisperEngine
from .hf_pipeline_engine import HFPipelineEngine
from .torch_engines import GigaAMEngine, QwenASREngine
from .vosk_engine import VoskEngine
from .whispercpp_engine import WhisperCppEngine

log = get_logger("asr.registry")

ENGINE_CLASSES: Dict[str, type] = {
    ENGINE_FASTER_WHISPER: FasterWhisperEngine,
    ENGINE_WHISPERCPP: WhisperCppEngine,
    ENGINE_VOSK: VoskEngine,
    ENGINE_QWEN_ASR: QwenASREngine,
    ENGINE_GIGAAM: GigaAMEngine,
    ENGINE_HF: HFPipelineEngine,
}


def engine_class(engine: str) -> type:
    try:
        return ENGINE_CLASSES[engine]
    except KeyError:
        raise EngineError(_(
            "Unknown engine \"{engine}\". Supported: {list}"
        ).format(engine=engine, list=", ".join(ENGINE_CLASSES))) from None


def engine_available(engine: str) -> bool:
    try:
        # the engine may have just been installed from the GUI — invalidate
        # importer caches, otherwise find_spec() misses fresh packages
        import importlib

        importlib.invalidate_caches()
        return engine_class(engine).is_available()
    except EngineError:
        return False


def missing_engine_packages(engine: str) -> List[str]:
    if engine_available(engine):
        return []
    return engine_packages(engine) or [engine]


def ref_from_entry(entry: ModelEntry) -> ModelRef:
    if not entry.install_dir:
        raise EngineError(_("Model \"{name}\" is not installed").format(name=entry.name))
    meta_subfolder = ""
    if entry.install_dir.is_dir():
        from ..core.downloader import load_meta

        meta = load_meta(entry.install_dir)
        meta_subfolder = str(meta.get("subdir") or meta.get("subfolder") or "")
    return ModelRef(
        model_id=entry.id,
        name=entry.name,
        engine=entry.engine,
        path=entry.install_dir,
        subfolder=meta_subfolder or entry.subfolder,
        languages=list(entry.languages),
        defaults=dict(entry.defaults or {}),
    )


class EngineRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._load_lock = threading.Lock()
        self._engine: Optional[ASREngine] = None
        self._engine_model_id: Optional[str] = None
        self._hardware_cache: Optional[Dict[str, Any]] = None
        self._hardware_raw: Any = None

    # ------------------------------------------------------------------ #
    def hardware(self, refresh: bool = False) -> Dict[str, Any]:
        """Hardware summary (cached: calling nvidia-smi every time is costly)."""
        with self._lock:
            if self._hardware_cache is not None and not refresh:
                return dict(self._hardware_cache)

        from ..core import sysinfo

        info = sysinfo.collect()
        device, compute_type = sysinfo.recommended_device(info)
        data = {
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": 0,
            "recommended_threads": info.recommended_threads(),
            "gpu_index": 0,
            "cuda_ok": bool(info.cuda_runtime_ok),
            "vram_mb": info.vram_mb,
            "gpu_driver": (info.best_gpu.driver if info.best_gpu else ""),
            "gpu_name": (info.best_gpu.name if info.best_gpu else ""),
            "ram_available_mb": info.ram_available_mb,
            "ram_total_mb": info.ram_total_mb,
            "has_nvidia": info.has_nvidia,
            "batched": device == "cuda",
            "batch_size": 8 if device == "cuda" else 4,
            "summary": info.summary(),
            "notes": list(info.notes),
        }
        with self._lock:
            self._hardware_cache = data
            self._hardware_raw = info
        log.debug("Hardware detected: %s", data)
        return dict(data)

    def hardware_info(self) -> Any:
        self.hardware()
        return self._hardware_raw

    # ------------------------------------------------------------------ #
    def get(self, entry: ModelEntry, settings: Optional[Dict[str, Any]] = None) -> ASREngine:
        """Return a LOADED engine for the model (reuses the cache).

        Loading happens OUTSIDE `_lock` (under a separate `_load_lock`):
        otherwise any GUI hardware() call would block the interface for the
        tens of seconds the model loads.
        """
        settings = dict(settings or {})
        ref = ref_from_entry(entry)

        with self._lock:
            if (
                self._engine is not None
                and self._engine_model_id == ref.model_id
                and self._engine.name == ref.engine
            ):
                log.debug("Registry: %s already loaded — reusing engine %s", ref.name, ref.engine)
                return self._engine

        with self._load_lock:
            # double-check: a neighbor thread may have loaded the engine
            with self._lock:
                if (
                    self._engine is not None
                    and self._engine_model_id == ref.model_id
                    and self._engine.name == ref.engine
                ):
                    return self._engine
                stale = self._engine
                self._engine = None
                self._engine_model_id = None
            if stale is not None:
                try:
                    stale.unload()
                    log.info(_("Previous model unloaded (%s)"), stale.name)
                except Exception as exc:  # noqa: BLE001
                    log.warning(_("Failed to unload the previous model: %s"), exc)

            cls = engine_class(ref.engine)
            engine: ASREngine = cls()
            engine.ensure_available()

            hardware = self.hardware()
            hardware.update(
                {
                    "device": settings.get("device", hardware["device"]),
                    "compute_type": settings.get("compute_type", hardware["compute_type"]),
                    "cpu_threads": int(settings.get("cpu_threads") or 0),
                    "gpu_index": int(settings.get("gpu_index") or 0),
                }
            )
            log.info(_("Loading engine %s for model %s"), ref.engine, ref.name)
            log.debug(
                "Load params: path=%s subfolder=%r device=%s compute=%s threads=%s gpu_index=%s",
                ref.path, ref.subfolder, hardware.get("device"), hardware.get("compute_type"),
                hardware.get("cpu_threads"), hardware.get("gpu_index"),
            )
            load_started = time.time()
            try:
                engine.load(ref, hardware)
            except EngineNotInstalled:
                raise
            except (ImportError, ModuleNotFoundError) as exc:
                # Model code (trust_remote_code) needs a library missing from
                # the env: turn the traceback into a clear install dialog.
                converted = engine_not_installed_from_exc(ref.engine, exc)
                if converted is not None:
                    log.warning(_("Engine %s is missing a module: %s"), ref.engine, exc)
                    raise converted from exc
                raise
            log.debug(_("Engine %s loaded in %.1f s"), ref.engine, time.time() - load_started)
            with self._lock:
                self._engine = engine
                self._engine_model_id = ref.model_id
            return engine

    def current(self) -> Optional[ASREngine]:
        with self._lock:
            return self._engine

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        engine = self._engine
        self._engine = None
        self._engine_model_id = None
        if engine is not None:
            try:
                engine.unload()
                log.info(_("Model unloaded from memory (%s)"), engine.name)
            except Exception as exc:  # noqa: BLE001
                log.warning(_("Failed to unload the model cleanly: %s"), exc)

    def ensure_engine_installed(self, engine: str) -> None:
        if not engine_available(engine):
            raise EngineNotInstalled(engine, missing_engine_packages(engine))


registry = EngineRegistry()


# --------------------------------------------------------------------------- #
#  Per-thread engine instances for parallel file processing
# --------------------------------------------------------------------------- #

_thread_local = threading.local()


def thread_engine(entry: ModelEntry, settings: Optional[Dict[str, Any]] = None) -> ASREngine:
    """Engine bound to the current thread (for parallel_files > 1).

    Each thread gets its own loaded model: CTranslate2/whisper.cpp dislike
    one object being pulled from several threads at once.
    """
    settings = dict(settings or {})
    cache: Dict[str, ASREngine] = getattr(_thread_local, "engines", None) or {}
    key = f"{entry.id}|{settings.get('device')}|{settings.get('compute_type')}|{settings.get('cpu_threads')}"
    engine = cache.get(key)
    if engine is not None:
        return engine

    ref = ref_from_entry(entry)
    cls = engine_class(ref.engine)
    engine = cls()
    engine.ensure_available()
    hardware = registry.hardware()
    hardware.update(
        {
            "device": settings.get("device", hardware["device"]),
            "compute_type": settings.get("compute_type", hardware["compute_type"]),
            "cpu_threads": int(settings.get("cpu_threads") or 0),
            "gpu_index": int(settings.get("gpu_index") or 0),
        }
    )
    log.info(_("[thread %s] loading %s for %s"), threading.current_thread().name, ref.engine, ref.name)
    try:
        engine.load(ref, hardware)
    except EngineNotInstalled:
        raise
    except (ImportError, ModuleNotFoundError) as exc:
        converted = engine_not_installed_from_exc(ref.engine, exc)
        if converted is not None:
            raise converted from exc
        raise
    cache[key] = engine
    _thread_local.engines = cache
    return engine


def release_thread_engines() -> None:
    cache: Dict[str, ASREngine] = getattr(_thread_local, "engines", None) or {}
    for engine in cache.values():
        try:
            engine.unload()
        except Exception as exc:  # noqa: BLE001
            log.warning(_("Failed to unload a thread-local model: %s"), exc)
    _thread_local.engines = {}
