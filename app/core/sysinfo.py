"""Hardware detection: CPU, RAM, GPU — used to recommend a suitable model."""

from __future__ import annotations

import os
import platform as platform_mod
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..i18n import _
from ..logutils import get_logger

log = get_logger("sysinfo")

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"


@dataclass
class GpuInfo:
    name: str = ""
    vram_mb: int = 0
    driver: str = ""
    backend: str = ""  # cuda | directml | metal | ""

    def __bool__(self) -> bool:
        return bool(self.name)


@dataclass
class HardwareInfo:
    # IMPORTANT: the field must not be named `platform` — it would shadow the
    # module name inside the class body and break `platform.platform()`.
    os_name: str = field(default_factory=platform_mod.system)
    platform_detail: str = field(default_factory=platform_mod.platform)
    python: str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    arch: str = platform_mod.machine()
    cpu_logical: int = os.cpu_count() or 2
    cpu_physical: int = 0
    cpu_name: str = ""
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    gpus: List[GpuInfo] = field(default_factory=list)
    cuda_runtime_ok: Optional[bool] = None  # None = not checked
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def best_gpu(self) -> Optional[GpuInfo]:
        return self.gpus[0] if self.gpus else None

    @property
    def vram_mb(self) -> int:
        return self.best_gpu.vram_mb if self.best_gpu else 0

    @property
    def has_nvidia(self) -> bool:
        return any(g.backend == "cuda" for g in self.gpus)

    def recommended_threads(self) -> int:
        """Threads for the model: physical cores, capped at 16."""
        cores = self.cpu_physical or self.cpu_logical
        return max(1, min(16, cores))

    def summary(self) -> str:
        gpu_text = ", ".join(
            f"{g.name}" + (_(" ({:.1f} GB VRAM)").format(g.vram_mb / 1024) if g.vram_mb else "")
            for g in self.gpus
        ) or _("no discrete GPU")
        return _(
            "CPU: {cpu} · {cores} cores ({threads} threads)\n"
            "RAM: {ram:.1f} GB (free {free:.1f} GB)\n"
            "GPU: {gpu}\n"
            "Python: {py} · {plat}"
        ).format(
            cpu=self.cpu_name or self.arch,
            cores=self.cpu_physical or self.cpu_logical,
            threads=self.cpu_logical,
            ram=self.ram_total_mb / 1024, free=self.ram_available_mb / 1024,
            gpu=gpu_text, py=self.python, plat=self.platform_detail,
        )

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["gpus"] = [g.__dict__ for g in self.gpus]
        return data


# --------------------------------------------------------------------------- #
#  Collecting info
# --------------------------------------------------------------------------- #


def _run(cmd: List[str], timeout: int = 8) -> Optional[str]:
    try:
        kwargs: dict = {"capture_output": True, "text": True, "timeout": timeout}
        if IS_WINDOWS:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout or ""


def _cpu_name() -> str:
    if IS_WINDOWS:
        text = _run(["wmic", "cpu", "get", "name"], timeout=10)
        if text:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) > 1:
                return lines[1]
        try:
            return os.environ.get("PROCESSOR_IDENTIFIER", "") or platform_mod.processor()
        except Exception:  # noqa: BLE001
            return ""
    if IS_MACOS:
        return (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or platform_mod.processor()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform_mod.processor() or platform_mod.machine()


def _cpu_physical_windows_ctypes() -> int:
    """Physical cores via GetLogicalProcessorInformation (wmic is removed
    from Windows 11 24H2+, so WinAPI is the primary path).
    """
    try:
        import ctypes

        class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("ProcessorMask", ctypes.c_ulonglong),
                ("Relationship", ctypes.c_int),
                ("Reserved", ctypes.c_ulonglong * 2),
            ]

        RELATION_PROCESSOR_CORE = 0
        kernel32 = ctypes.windll.kernel32
        buffer_length = ctypes.c_ulong(0)
        # the first call always "fails" — it reports the required buffer size
        kernel32.GetLogicalProcessorInformation(None, ctypes.byref(buffer_length))
        if not buffer_length.value:
            return 0
        item_size = ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
        count = buffer_length.value // item_size
        buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * count)()
        if not kernel32.GetLogicalProcessorInformation(buffer, ctypes.byref(buffer_length)):
            return 0
        return sum(
            1 for item in buffer if item.Relationship == RELATION_PROCESSOR_CORE
        )
    except Exception:  # noqa: BLE001
        return 0


def _cpu_physical() -> int:
    if IS_WINDOWS:
        cores = _cpu_physical_windows_ctypes()
        if cores:
            return cores
        text = _run(["wmic", "cpu", "get", "NumberOfCores"], timeout=10)
        if text:
            numbers = re.findall(r"\d+", text)
            if numbers:
                return int(numbers[0])
    elif IS_MACOS:
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        if out and out.strip().isdigit():
            return int(out.strip())
    else:
        try:
            out = _run(["nproc", "--all"]) or _run(["getconf", "_NPROCESSORS_ONLN"])
            if out and out.strip().isdigit():
                # nproc reports logical; count physical cores from cpuinfo
                cores = set()
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
                    phys = core_id = None
                    for line in fh:
                        low = line.lower()
                        if low.startswith("physical id"):
                            phys = line.split(":")[1].strip()
                        elif low.startswith("core id"):
                            core_id = line.split(":")[1].strip()
                            cores.add((phys, core_id))
                if cores:
                    return len(cores)
                return int(out.strip())
        except (OSError, ValueError):
            pass
    return os.cpu_count() or 2


def _memory() -> tuple[int, int]:
    """(total MB, available MB) — no external dependencies."""
    total = avail = 0
    try:
        if IS_WINDOWS:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total = int(stat.ullTotalPhys // (1024 * 1024))
                avail = int(stat.ullAvailPhys // (1024 * 1024))
        elif IS_MACOS:
            out = _run(["sysctl", "-n", "hw.memsize"])
            if out and out.strip().isdigit():
                total = int(int(out.strip()) // (1024 * 1024))
                avail = total // 2
        else:
            with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    value = re.findall(r"\d+", rest)
                    if not value:
                        continue
                    kb = int(value[0])
                    if key == "MemTotal":
                        total = kb // 1024
                    elif key in ("MemAvailable", "MemFree"):
                        avail = max(avail, kb // 1024)
    except Exception as exc:  # noqa: BLE001
        log.debug("Failed to read memory info: %s", exc)
    return total, avail


def _nvidia_gpus() -> List[GpuInfo]:
    smi = shutil.which("nvidia-smi")
    if IS_WINDOWS and not smi:
        for cand in (
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ):
            if os.path.isfile(cand):
                smi = cand
                break
    if not smi:
        return []
    out = _run(
        [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        timeout=12,
    )
    gpus: List[GpuInfo] = []
    if out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2 or not parts[0]:
                continue
            try:
                vram = int(float(parts[1]))
            except ValueError:
                vram = 0
            gpus.append(
                GpuInfo(
                    name=parts[0],
                    vram_mb=vram,
                    driver=parts[2] if len(parts) > 2 else "",
                    backend="cuda",
                )
            )
    return gpus


def _check_cuda_runtime() -> bool:
    """Check that CTranslate2 actually sees CUDA (without loading a model)."""
    try:
        import ctranslate2  # type: ignore

        devices = ctranslate2.get_cuda_device_count()
        return bool(devices and devices > 0)
    except Exception as exc:  # noqa: BLE001
        log.debug("ctranslate2 CUDA check: %s", exc)
        return False


def collect() -> HardwareInfo:
    info = HardwareInfo()
    info.cpu_name = _cpu_name()
    info.cpu_logical = os.cpu_count() or 2
    info.cpu_physical = _cpu_physical() or info.cpu_logical
    info.ram_total_mb, info.ram_available_mb = _memory()

    gpus = _nvidia_gpus()
    if gpus:
        info.gpus = gpus
        info.cuda_runtime_ok = _check_cuda_runtime()
        if not info.cuda_runtime_ok:
            info.notes.append(_(
                "An NVIDIA GPU was found but CTranslate2 does not see CUDA "
                "(cuBLAS/cuDNN 9 are missing). Install: Tools → Settings → Engines tab."
            ))
    elif IS_MACOS and info.arch.lower() in ("arm64", "aarch64"):
        info.notes.append(_("Apple Silicon: acceleration goes through CPU optimizations (Accelerate/AMX)."))

    if info.ram_total_mb and info.ram_total_mb < 6000:
        info.notes.append(_(
            "Less than 6 GB of RAM — models up to ~500 MB are more comfortable; "
            "each model's size is visible in the catalog (Size, MB filter)."
        ))
    return info


# --------------------------------------------------------------------------- #
#  Recommendations for this hardware
# --------------------------------------------------------------------------- #

def _hw_get(info: Any, key: str, default: Any = 0) -> Any:
    """Hardware field from HardwareInfo OR a dict (the GUI passes a dict)."""
    if isinstance(info, dict):
        value = info.get(key, default)
    else:
        value = getattr(info, key, default)
    return default if value is None else value


# Thresholds: (min RAM MB, min VRAM MB) -> model size "ceiling" on disk (MB)
def max_model_size_mb(info: Any) -> int:
    cuda_ok = bool(_hw_get(info, "cuda_runtime_ok", False) or _hw_get(info, "cuda_ok", False))
    vram = int(_hw_get(info, "vram_mb", 0) or 0)
    if cuda_ok and vram:
        if vram >= 10_000:
            return 3_200        # large-v3
        if vram >= 7_000:
            return 1_800        # large-v3-turbo float16
        if vram >= 4_500:
            return 900          # turbo int8 / small
        return 500

    ram = int(_hw_get(info, "ram_available_mb", 0) or _hw_get(info, "ram_total_mb", 0) or 0)
    if ram >= 12_000:
        return 1_800
    if ram >= 8_000:
        return 900
    if ram >= 5_000:
        return 500
    return 200


def recommended_device(info: HardwareInfo) -> tuple[str, str]:
    """Return (device, compute_type) for the detected hardware."""
    if getattr(info, "cuda_runtime_ok", False) and int(getattr(info, "vram_mb", 0) or 0) >= 4_000:
        return "cuda", "float16" if int(info.vram_mb) >= 6_000 else "int8_float16"
    return "cpu", "int8"


