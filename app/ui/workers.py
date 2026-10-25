"""Qt background threads: package installs, model downloads, recognition.

All heavy work happens HERE, not in the GUI thread: the interface stays
responsive, progress arrives via signals, cancellation via a flag.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PySide6.QtCore import QThread, Signal

from .. import paths
from ..asr.base import EngineNotInstalled
from ..asr.registry import registry, release_thread_engines, thread_engine
from ..asr.transcriber import FileResult, JobOptions, transcribe_file
from ..core import downloader
from ..core.downloader import InstallError
from ..core.ffmpeg import decode_audio, find_ffmpeg
from ..core.models_catalog import ModelCatalog, ModelEntry
from ..constants import MAX_PREFETCH, SAMPLE_RATE
from ..i18n import _
from ..logutils import get_logger
from ..settings import Settings

log = get_logger("workers")


def venv_python() -> str:
    """Interpreter packages are installed into (we are inside it ourselves)."""
    return sys.executable or str(paths.venv_dir() / "Scripts" / "python.exe")


def stop_worker(worker: Optional[QThread], wait_ms: int = 4000, label: str = "") -> None:
    """Reliably stop a QThread (used when the window closes).

    Order: cancel() (if any) -> wait -> terminate. Without terminate() a
    thread stuck in a network retry or subprocess outlived Qt object
    deletion ("QThread: Destroyed while thread is still running").
    """
    if worker is None:
        return
    try:
        if not worker.isRunning():
            return
        cancel = getattr(worker, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # noqa: BLE001
                pass
        if worker.wait(wait_ms):
            return
        log.warning(_("Worker %s did not stop in %d ms — terminating forcibly"),
                    label or type(worker).__name__, wait_ms)
        worker.terminate()
        worker.wait(2000)
    except RuntimeError:
        pass    # the C++ object is already deleted


# --------------------------------------------------------------------------- #
#  Package installation into the project environment
# --------------------------------------------------------------------------- #


class PackageInstaller(QThread):
    """Install (or remove — mode="uninstall") pip packages strictly into the project env."""

    outputLine = Signal(str)
    progress = Signal(float, str)
    finishedWith = Signal(bool, str)

    def __init__(
        self,
        packages: Sequence[str],
        title: str = "",
        parent=None,
        pre_uninstall: Optional[Sequence[str]] = None,
        mode: str = "install",
    ) -> None:
        super().__init__(parent)
        self.packages = list(packages)
        self.title = title or (_("Uninstalling packages") if mode == "uninstall" else _("Installing packages"))
        self.mode = mode
        self._cancel = False
        self._proc: Optional[subprocess.Popen] = None
        # what to remove BEFORE installing (e.g. the CPU torch build before
        # the CUDA one: otherwise pip says "already satisfied" and installs nothing)
        self.pre_uninstall = list(pre_uninstall or [])

    def cancel(self) -> None:
        self._cancel = True
        # kill the child pip/uv immediately: otherwise the thread hangs
        # reading stdout while the resolver is silent and the window stays
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    def run(self) -> None:  # noqa: D102
        started = time.time()
        python = venv_python()
        self.progress.emit(0.02, _("Checking the environment: {py}").format(py=python))
        self.outputLine.emit(f"python: {python}")

        if not self.packages:
            self.finishedWith.emit(True, _("Nothing to install"))
            return

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        if self.mode == "uninstall":
            cmd = [python, "-m", "pip", "uninstall", "-y", *self.packages]
            self.progress.emit(0.05, _("Uninstalling packages…"))
            self.outputLine.emit("$ " + " ".join(cmd))
            code = self._stream(cmd, env)
            if self._cancel:
                self.finishedWith.emit(False, _("Uninstall cancelled"))
                return
            if code != 0:
                self.finishedWith.emit(False, _("pip uninstall failed (code {code})").format(code=code))
                return
            self.progress.emit(1.0, _("Done"))
            self.finishedWith.emit(True, _("Uninstalled in {:.0f} s").format(time.time() - started))
            return

        requirements_file = next((p for p in self.packages if p.startswith("-r ")), "")
        if requirements_file and not any(p.startswith("-") is False for p in self.packages):
            cmd = [python, "-m", "pip", "install", *self.packages]
        else:
            uv = self._find_uv(python)
            if uv and not requirements_file:
                cmd = [uv, "pip", "install", "--python", python, *self.packages]
                # uv takes a package from the FIRST index that has it: with an
                # extra PyTorch CUDA index it would take the CPU torch from
                # PyPI. unsafe-best-match compares versions across indexes and
                # picks +cu126 (newer than the plain version per PEP 440).
                # pip does not need this.
                if "--extra-index-url" in self.packages:
                    cmd += ["--index-strategy", "unsafe-best-match"]
            else:
                cmd = [
                    python, "-m", "pip", "install",
                    "--disable-pip-version-check", "--no-input", "--prefer-binary",
                    "--timeout", "60", "--retries", "3", *self.packages,
                ]

        if self.pre_uninstall:
            self.progress.emit(0.03, _("Removing the old version: ") + ", ".join(self.pre_uninstall))
            uninstall_cmd = [python, "-m", "pip", "uninstall", "-y", *self.pre_uninstall]
            self.outputLine.emit("$ " + " ".join(uninstall_cmd))
            self._stream(uninstall_cmd, env)   # return code does not matter: it may be absent

        self.progress.emit(0.05, _("Starting the installation…"))
        self.outputLine.emit("$ " + " ".join(cmd))

        code = self._stream(cmd, env)
        if self._cancel:
            self.finishedWith.emit(False, _("Install cancelled"))
            return
        if code != 0:
            # uv sometimes cannot resolve the environment — try plain pip
            if cmd[0] != python:
                self.outputLine.emit(_("First method failed, trying pip…"))
                code = self._stream(
                    [python, "-m", "pip", "install", "--disable-pip-version-check",
                     "--no-input", "--prefer-binary", *self.packages],
                    env,
                )
            if code != 0:
                from .. import logutils

                current = logutils.current_log_file()
                log_hint = str(current) if current else str(paths.logs_dir())
                self.finishedWith.emit(
                    False,
                    _("Installation failed (code {code}).\nDetails: {log}\n"
                      "(the verbose log is enabled in Tools → Settings → 'Logging & diagnostics')"
                      ).format(code=code, log=log_hint),
                )
                return

        self.progress.emit(1.0, _("Done"))
        # if transformers was already imported in this process (e.g. GigaAM
        # was loaded before), its availability lru caches are stale: freshly
        # installed packages (accelerate etc.) are "invisible" to them. Reset —
        # the next model run picks up the new packages WITHOUT a restart.
        try:
            if "transformers" in sys.modules:
                from ..asr.torch_engines import reset_transformers_caches

                cleared = reset_transformers_caches()
                log.debug("transformers lru caches cleared after install: %d", cleared)
        except Exception as exc:  # noqa: BLE001
            log.debug("Failed to clear transformers caches after install: %s", exc)
        self.finishedWith.emit(True, _("Installed in {:.0f} s").format(time.time() - started))

    # ------------------------------------------------------------------ #
    def _stream(self, cmd: List[str], env: Dict[str, str]) -> int:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(paths.ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.outputLine.emit(_("!! failed to start: {exc}").format(exc=exc))
            return 127

        self._proc = proc
        log.debug("PackageInstaller subprocess: %s", " ".join(cmd))
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if self._cancel:
                    proc.kill()
                    break
                text = line.rstrip()
                if text:
                    self.outputLine.emit(text)
                    low = text.lower()
                    if "downloading" in low or "collecting" in low:
                        self.progress.emit(-1.0, text[:160])
            proc.wait()
            return proc.returncode
        finally:
            self._proc = None

    @staticmethod
    def _find_uv(python: str) -> Optional[str]:
        import shutil

        found = shutil.which("uv")
        if found:
            return found
        candidate = Path(python).parent / ("uv.exe" if os.name == "nt" else "uv")
        return str(candidate) if candidate.is_file() else None


# --------------------------------------------------------------------------- #
#  Model download
# --------------------------------------------------------------------------- #


class ModelDownloadWorker(QThread):
    outputLine = Signal(str)
    progress = Signal(float, str)
    finishedWith = Signal(bool, str, object)   # ok, message, ModelEntry|None

    def __init__(self, entry: ModelEntry, root: Optional[Path] = None, parent=None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.root = Path(root) if root else None
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        name = self.entry.display_name
        log.debug("ModelDownloadWorker started: %s (source=%s, root=%s)",
                  self.entry.id, self.entry.source, self.root)
        self.progress.emit(0.0, _("Preparing the download: {name}").format(name=name))
        self.outputLine.emit(_("Model: {name} ({id})").format(name=name, id=self.entry.id))
        self.outputLine.emit(_("Source: {source}").format(source=self.entry.source))

        try:
            result = downloader.install_model(
                self.entry,
                root=self.root,
                progress=lambda fraction, status: (
                    self.progress.emit(fraction, status),
                    self.outputLine.emit(status),
                )[0],
                cancel=lambda: self._cancel,
            )
        except downloader.net.DownloadCancelled:  # type: ignore[attr-defined]
            self.finishedWith.emit(False, _("Download cancelled. Incomplete files can be resumed later."), None)
            return
        except InstallError as exc:
            self.finishedWith.emit(False, str(exc), None)
            return
        except Exception as exc:  # noqa: BLE001
            log.error("Model install failed: %s\n%s", exc, traceback.format_exc())
            self.finishedWith.emit(False, f"{type(exc).__name__}: {exc}", None)
            return

        if self._cancel:
            self.finishedWith.emit(False, _("Download cancelled"), None)
            return

        size_text = paths.human_size(result.downloaded_bytes)
        self.progress.emit(1.0, _("Done"))
        self.outputLine.emit(_("Installed into {dir} ({size} in {t:.0f} s)").format(
            dir=result.directory, size=size_text, t=result.duration_seconds))
        self.finishedWith.emit(
            True,
            _("{name} installed into {dir}/ ({size})").format(
                name=name, dir=result.directory.name, size=size_text),
            result.entry,
        )


# --------------------------------------------------------------------------- #
#  Remote catalog refresh
# --------------------------------------------------------------------------- #


class CatalogRefreshWorker(QThread):
    finishedWith = Signal(bool, str)

    def run(self) -> None:  # noqa: D102
        try:
            catalog = ModelCatalog()
            updated = catalog.refresh_remote(force=True)
            count = len(catalog.entries)
            if updated:
                self.finishedWith.emit(True, _("Catalog updated: {n} models").format(n=count))
            else:
                self.finishedWith.emit(False, _("No updates (using the local catalog, {n} models)").format(n=count))
        except Exception as exc:  # noqa: BLE001
            self.finishedWith.emit(False, _("Failed to update the catalog: {exc}").format(exc=exc))


# --------------------------------------------------------------------------- #
#  Integrity diagnostics (project files + environment audit)
# --------------------------------------------------------------------------- #


class EnvAuditWorker(QThread):
    """Check project files, folders and env packages (see core/integrity.py)."""

    finishedWith = Signal(object)   # dict report

    def run(self) -> None:  # noqa: D102
        from ..core import integrity

        try:
            report = integrity.full_audit()
        except Exception as exc:  # noqa: BLE001
            log.error("Diagnostics failed: %s\n%s", exc, traceback.format_exc())
            report = {
                "python": "",
                "missing_files": [],
                "folders": [],
                "base_missing": [],
                "extra_packages": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        self.finishedWith.emit(report)


class FileRestoreWorker(QThread):
    """Download missing program files from GitHub."""

    outputLine = Signal(str)
    progress = Signal(float, str)
    finishedWith = Signal(bool, str)

    def __init__(self, files: Sequence[str], parent=None) -> None:
        super().__init__(parent)
        self.files = list(files)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        from ..core import integrity

        if not self.files:
            self.finishedWith.emit(True, _("Nothing to restore — all files are present"))
            return
        self.progress.emit(0.0, _("Restoring {n} file(s) from GitHub…").format(n=len(self.files)))
        restored, failed = integrity.restore_files(
            self.files,
            progress=lambda fraction, status: (
                self.progress.emit(fraction, status),
                self.outputLine.emit(status),
            )[0],
            cancel=lambda: self._cancel,
        )
        if failed:
            self.finishedWith.emit(
                False,
                _("Restored: {ok}, failed: {fail}\n").format(ok=len(restored), fail=len(failed))
                + "\n".join(failed[:8]),
            )
        else:
            self.progress.emit(1.0, _("Done"))
            self.finishedWith.emit(True, _("Files restored: {n}").format(n=len(restored)))


# --------------------------------------------------------------------------- #
#  Probe: does the env's torch see CUDA (in a separate process!)
# --------------------------------------------------------------------------- #


class TorchCudaProbeWorker(QThread):
    """Check via subprocess whether torch is installed and sees CUDA.

    Why not in the GUI process:
      * import torch blocks for seconds — not allowed in the GUI thread;
      * an imported torch "pins" the build (CPU/CUDA) for the whole session,
        and we need the on-disk state without touching this process.
    """

    probed = Signal(bool, bool, str)   # torch_installed, cuda_available, detail

    _SCRIPT = (
        "import importlib.util as iu, json\n"
        "if iu.find_spec('torch') is None:\n"
        "    print('PROBE' + json.dumps({'installed': False}))\n"
        "else:\n"
        "    import torch\n"
        "    print('PROBE' + json.dumps({"
        "'installed': True,"
        "'cuda': bool(torch.cuda.is_available()),"
        "'version': torch.__version__,"
        "'cuda_ver': torch.version.cuda or ''}))\n"
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        proc: Optional[subprocess.Popen] = None
        try:
            proc = subprocess.Popen(
                [venv_python(), "-c", self._SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            # wait in 0.5 s slices: closing the window must not wait the full timeout
            deadline = time.time() + 180
            out_text, err_text = "", ""
            while time.time() < deadline:
                if self._cancel:
                    proc.kill()
                    return
                try:
                    out_text, err_text = proc.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
            for line in reversed((out_text or "").splitlines()):
                if line.startswith("PROBE"):
                    data = json.loads(line[len("PROBE"):])
                    detail = str(data.get("version") or "")
                    if data.get("cuda_ver"):
                        detail += f" + CUDA {data['cuda_ver']}"
                    self.probed.emit(
                        bool(data.get("installed")), bool(data.get("cuda")), detail
                    )
                    return
            log.debug("torch probe produced no result: rc=%s stderr=%s",
                      proc.returncode if proc else "?", (err_text or "")[-300:])
        except Exception as exc:  # noqa: BLE001
            log.debug("torch probe did not start: %s", exc)
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
        if not self._cancel:
            self.probed.emit(False, False, "")


# --------------------------------------------------------------------------- #
#  File queue recognition
# --------------------------------------------------------------------------- #


class TranscribeWorker(QThread):
    """Process the file queue.

    Signals:
      overallProgress(done, total)        — overall file progress
      fileProgress(index, fraction, text) — current file progress
      segmentReady(index, text, start, end) — live text feed
      fileFinished(index, FileResult)
      finishedWith(ok_count, fail_count, seconds)
      engineMissing(engine, packages)     — engine not installed
    """

    overallProgress = Signal(int, int)
    fileProgress = Signal(int, float, str)
    segmentReady = Signal(int, str, float, float)
    fileFinished = Signal(int, object)
    logLine = Signal(str)
    finishedWith = Signal(int, int, float)
    failed = Signal(str)
    engineMissing = Signal(str, object)

    def __init__(
        self,
        files: Sequence[Path],
        options: JobOptions,
        settings: Settings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.files = [Path(f) for f in files]
        self.options = options
        self.settings = settings
        self._cancel = threading.Event()
        #: the engine lacked libraries — the GUI shows an install dialog and retries
        self.engine_missing = False

    # ------------------------------------------------------------------ #
    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------------ #
    def run(self) -> None:  # noqa: D102
        started = time.time()
        log.debug(
            "TranscribeWorker started: files=%d, model=%s, engine=%s, parallel=%s",
            len(self.files), self.options.model.name, self.options.model.engine,
            self.settings.parallel_files,
        )
        try:
            if not self.files:
                self.finishedWith.emit(0, 0, 0.0)
                return

            entry = self.options.model
            if not entry.installed or not entry.install_dir:
                self.failed.emit(_("Model {name} is not installed").format(name=entry.display_name))
                return

            try:
                registry.ensure_engine_installed(entry.engine)
            except EngineNotInstalled as exc:
                self.engineMissing.emit(exc.engine, list(exc.packages))
                return

            # Engine packages may have been installed WHILE the app was
            # running (install-on-demand). transformers caches availability
            # checks (lru_cache): without a reset it keeps treating a freshly
            # installed accelerate as missing and model loading dies with
            # "requires accelerate". Resetting is safe and cheap.
            try:
                if "transformers" in sys.modules:
                    from ..asr.torch_engines import reset_transformers_caches

                    cleared = reset_transformers_caches()
                    if cleared:
                        log.debug("transformers lru caches cleared before engine load: %d", cleared)
            except Exception as exc:  # noqa: BLE001
                log.debug("transformers cache reset failed: %s", exc)

            hardware = registry.hardware()
            log.debug("Hardware for the job: %s", hardware.get("summary", "").replace("\n", " | "))
            self.logLine.emit(hardware.get("summary", "").replace("\n", " | "))
            ffmpeg = find_ffmpeg()
            self.logLine.emit(_("ffmpeg: {path}").format(path=ffmpeg) if ffmpeg
                              else _("ffmpeg: NOT FOUND (PyAV will be used)"))

            parallel = max(1, min(4, int(self.settings.parallel_files or 1)))
            if parallel == 1:
                ok, fail = self._run_sequential()
            else:
                ok, fail = self._run_parallel(parallel)

            log.debug(
                "TranscribeWorker finished in %.1f s: ok=%d fail=%d cancel=%s",
                time.time() - started, ok, fail, self.cancelled,
            )
            self.finishedWith.emit(ok, fail, time.time() - started)
        except Exception as exc:  # noqa: BLE001
            # full traceback goes to the log; the user sees only the gist
            log.error("Worker crashed: %s\n%s", exc, traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                release_thread_engines()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    def _run_sequential(self) -> tuple[int, int]:
        ok = fail = 0
        total = len(self.files)

        # Decode prefetch: while the model computes file N, a side thread
        # already extracts audio from file N+1. Enabled only when RAM allows —
        # a decoded 3-hour lecture weighs ~700 MB by itself.
        prefetch_depth = self._prefetch_depth()
        self.logLine.emit(
            _("Decode prefetch: {n} file(s) ahead").format(n=prefetch_depth)
            if prefetch_depth else _("Decode prefetch disabled (low free memory)")
        )

        executor: Optional[ThreadPoolExecutor] = None
        futures: Dict[int, Future] = {}
        if prefetch_depth:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode")
            for ahead in range(1, prefetch_depth + 1):
                if ahead < total:
                    futures[ahead] = executor.submit(self._decode_one, self.files[ahead])

        for index, source in enumerate(self.files):
            if self.cancelled:
                self.logLine.emit(_("Cancelled by user"))
                break
            self.overallProgress.emit(index, total)

            audio: Any = None
            duration = 0.0
            decode_error = ""
            if executor is not None:
                audio, duration, decode_error = self._take_future(futures.pop(index, None))
                next_index = index + prefetch_depth
                if next_index < total:
                    futures[next_index] = executor.submit(self._decode_one, self.files[next_index])

            if decode_error:
                result = FileResult(source=source, error=decode_error)
                fail += 1
                self.fileFinished.emit(index, result)
                self.logLine.emit(_("✗ {name}: {error}").format(name=source.name, error=decode_error))
                continue

            try:
                engine = registry.get(
                    self.options.model,
                    {
                        "device": self.options.device,
                        "compute_type": self.options.compute_type,
                        "cpu_threads": self.options.cpu_threads,
                        "gpu_index": self.options.gpu_index,
                    },
                )
            except EngineNotInstalled as exc:
                # Model code lacked a library: show an install dialog with
                # auto-continuation of the job after the install, not a traceback.
                self.engine_missing = True
                self.engineMissing.emit(exc.engine, list(exc.packages))
                break
            result = self._process_one(index, source, audio, duration, engine)
            del audio
            ok += int(result.ok)
            fail += int(not result.ok)

        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self.overallProgress.emit(total, total)
        return ok, fail

    def _prefetch_depth(self) -> int:
        """How many files to decode ahead of time (0 = disable)."""
        if self.cancelled:
            return 0
        try:
            hardware = registry.hardware()
            available = int(hardware.get("ram_available_mb") or 0)
            model_ram = int(self.options.model.ram_mb or 0)
        except Exception:  # noqa: BLE001
            return MAX_PREFETCH
        if not available:
            return MAX_PREFETCH
        # need headroom: model + one file being decoded (~800 MB per audio hour)
        if available - model_ram < 2400:
            return 0
        if available - model_ram < 4800:
            return 1
        return MAX_PREFETCH

    def _run_parallel(self, workers: int) -> tuple[int, int]:
        total = len(self.files)
        ok = fail = 0
        lock = threading.Lock()
        self.overallProgress.emit(0, total)

        def handle(index: int, source: Path) -> FileResult:
            if self.cancelled:
                return FileResult(source=source, error=_("cancelled"))
            try:
                engine = thread_engine(
                    self.options.model,
                    {
                        "device": self.options.device,
                        "compute_type": self.options.compute_type,
                        "cpu_threads": self.options.cpu_threads,
                        "gpu_index": self.options.gpu_index,
                    },
                )
            except EngineNotInstalled as exc:
                self.engine_missing = True
                self.engineMissing.emit(exc.engine, list(exc.packages))
                return FileResult(source=source, error=_("missing engine components"))
            return self._process_one(index, source, None, 0.0, engine)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asr") as pool:
            futures = {pool.submit(handle, index, source): index for index, source in enumerate(self.files)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = FileResult(source=self.files[index], error=f"{type(exc).__name__}: {exc}")
                with lock:
                    ok += int(result.ok)
                    fail += int(not result.ok)
                    self.fileFinished.emit(index, result)
                    self.overallProgress.emit(ok + fail, total)
            # unload models INSIDE the worker threads: the engine cache lives
            # in their thread-local storage and a release call from our thread
            # could not reach it — memory (incl. VRAM) hung until thread death
            for _i in range(workers):
                pool.submit(release_thread_engines)

        self.overallProgress.emit(total, total)
        return ok, fail

    # ------------------------------------------------------------------ #
    def _process_one(
        self,
        index: int,
        source: Path,
        audio: Any,
        duration: float,
        engine: Any,
    ) -> FileResult:
        def progress(fraction: float, stage: str) -> None:
            self.fileProgress.emit(index, fraction, stage)

        def on_segment(segment) -> None:
            self.segmentReady.emit(index, segment.text, segment.start, segment.end)

        self.logLine.emit(f"→ {source.name}")
        try:
            result = transcribe_file(
                source,
                self.options,
                progress=progress,
                on_segment=on_segment,
                cancel=lambda: self.cancelled,
                audio=audio,
                audio_duration=duration,
                engine=engine,
            )
        except EngineNotInstalled as exc:
            self.engineMissing.emit(exc.engine, list(exc.packages))
            result = FileResult(source=source, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            log.error("Processing error for %s: %s\n%s", source.name, exc, traceback.format_exc())
            result = FileResult(source=source, error=f"{type(exc).__name__}: {exc}")

        if result.ok:
            outputs = ", ".join(p.name for p in result.outputs)
            self.logLine.emit(
                _("✓ {name} → {outputs} · {words} words · {speed:.1f}× · {t:.0f} s").format(
                    name=source.name, outputs=outputs, words=result.words,
                    speed=result.speed_x, t=result.elapsed_seconds)
                + (_(" · dropped segments: {n}").format(n=result.dropped) if result.dropped else "")
            )
        else:
            self.logLine.emit(_("✗ {name}: {error}").format(name=source.name, error=result.error))
        self.fileFinished.emit(index, result)
        return result

    def _decode_one(self, source: Path) -> tuple[Any, float, str]:
        """Prefetch: decode audio while the model is busy with the previous file."""
        try:
            audio = decode_audio(source, SAMPLE_RATE, cancel=lambda: self.cancelled)
            duration = len(audio) / SAMPLE_RATE
            return audio, duration, ""
        except Exception as exc:  # noqa: BLE001
            return None, 0.0, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _take_future(future: Optional[Future]) -> tuple[Any, float, str]:
        if future is None:
            return None, 0.0, ""
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001
            return None, 0.0, f"{type(exc).__name__}: {exc}"
