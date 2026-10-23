"""Определение железа: CPU, RAM, GPU. Нужно, чтобы советовать адекватную модель."""

from __future__ import annotations

import json
import os
import platform as platform_mod
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

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
    # ВАЖНО: поле не должно называться `platform` — иначе внутри тела класса
    # имя модуля перекрывается аннотацией поля и `platform.platform()` падает.
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
    cuda_runtime_ok: Optional[bool] = None  # None = не проверяли
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
        """Сколько потоков отдать модели: физические ядра, но не больше 16."""
        cores = self.cpu_physical or self.cpu_logical
        return max(1, min(16, cores))

    def summary(self) -> str:
        gpu_text = ", ".join(
            f"{g.name}" + (f" ({g.vram_mb / 1024:.1f} ГБ VRAM)" if g.vram_mb else "")
            for g in self.gpus
        ) or "нет дискретной GPU"
        return (
            f"CPU: {self.cpu_name or self.arch} · {self.cpu_physical or self.cpu_logical} ядер "
            f"({self.cpu_logical} потоков)\n"
            f"RAM: {self.ram_total_mb / 1024:.1f} ГБ (свободно {self.ram_available_mb / 1024:.1f} ГБ)\n"
            f"GPU: {gpu_text}\n"
            f"Python: {self.python} · {self.platform_detail}"
        )

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["gpus"] = [g.__dict__ for g in self.gpus]
        return data


# --------------------------------------------------------------------------- #
#  Сбор информации
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


def _cpu_physical() -> int:
    if IS_WINDOWS:
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
                # nproc отдаёт логические; попробуем посчитать физические из cpuinfo
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
    """(всего МБ, доступно МБ) — без внешних зависимостей."""
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
        log.debug("Не удалось прочитать память: %s", exc)
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
    """Проверяет, что CTranslate2 реально видит CUDA (без загрузки модели)."""
    try:
        import ctranslate2  # type: ignore

        devices = ctranslate2.get_cuda_device_count()
        return bool(devices and devices > 0)
    except Exception as exc:  # noqa: BLE001
        log.debug("CUDA-проверка ctranslate2: %s", exc)
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
            info.notes.append(
                "Видеокарта NVIDIA найдена, но CTranslate2 не видит CUDA "
                "(нет cuBLAS/cuDNN 9). Настройки → Ускорение → «Установить CUDA-библиотеки»."
            )
    elif IS_MACOS and info.arch.lower() in ("arm64", "aarch64"):
        info.notes.append("Apple Silicon: ускорение идёт через CPU-оптимизации (Accelerate/AMX).")

    if info.ram_total_mb and info.ram_total_mb < 6000:
        info.notes.append("Меньше 6 ГБ RAM — выбирай модели до 500 МБ (turbo-q5_0, small int8, Vosk).")
    return info


# --------------------------------------------------------------------------- #
#  Что рекомендуем под это железо
# --------------------------------------------------------------------------- #

# Пороги: (мин. RAM МБ, мин. VRAM МБ) -> «потолок» размера модели на диске (МБ)
def max_model_size_mb(info: HardwareInfo) -> int:
    if getattr(info, "cuda_runtime_ok", False) and getattr(info, "vram_mb", 0):
        vram = int(info.vram_mb)
        if vram >= 10_000:
            return 3_200        # large-v3
        if vram >= 7_000:
            return 1_800        # large-v3-turbo float16
        if vram >= 4_500:
            return 900          # turbo int8 / small
        return 500

    ram = int(getattr(info, "ram_available_mb", 0) or getattr(info, "ram_total_mb", 0) or 0)
    if ram >= 12_000:
        return 1_800
    if ram >= 8_000:
        return 900
    if ram >= 5_000:
        return 500
    return 200


def recommended_device(info: HardwareInfo) -> tuple[str, str]:
    """Возвращает (device, compute_type) под найденное железо."""
    if getattr(info, "cuda_runtime_ok", False) and int(getattr(info, "vram_mb", 0) or 0) >= 4_000:
        return "cuda", "float16" if int(info.vram_mb) >= 6_000 else "int8_float16"
    return "cpu", "int8"


def fits(model_ram_mb: int, info: HardwareInfo) -> tuple[bool, str]:
    """Проверяет, потянет ли железо модель. Возвращает (ok, причина)."""
    if getattr(info, "cuda_runtime_ok", False) and getattr(info, "vram_mb", 0):
        if model_ram_mb <= info.vram_mb * 0.85:
            return True, ""
        if model_ram_mb <= (info.ram_available_mb or info.ram_total_mb):
            return True, "не влезает в VRAM — пойдёт через CPU (медленнее)"
        return False, f"нужно ~{model_ram_mb} МБ памяти, свободно {info.ram_available_mb} МБ"
    available = info.ram_available_mb or info.ram_total_mb
    if available and model_ram_mb > available:
        return False, f"нужно ~{model_ram_mb} МБ RAM, свободно {available} МБ"
    return True, ""


def dump_report(info: Optional[HardwareInfo] = None) -> str:
    info = info or collect()
    return json.dumps(info.to_dict(), ensure_ascii=False, indent=2)
