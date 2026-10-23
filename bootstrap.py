#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bootstrap.py — подготовка окружения LocalASR.

Этот модуль **обязан** работать на голом Python: только стандартная библиотека,
никаких импортов `app.*` на уровне модуля (иначе «курица и яйцо»: чтобы
прочитать список зависимостей, нужны зависимости).

Что делает `run()`:
  1. Находит системный Python (3.10+, 64-bit; проверено до 3.14) и создаёт `.venv`.
  2. Сравнивает «отпечаток» требований с сохранённым: если совпал — проверка
     зависимостей пропускается, запуск мгновенный.
  3. Иначе проверяет импорты и доустанавливает недостающее (uv, если есть,
     иначе pip) — строго в `.venv`.
  4. Перезапускает приложение интерпретатором из `.venv`, на Windows через
     `pythonw.exe` → консоль исчезает, остаётся только GUI.

Любой сбой показывается в нативном диалоге (Tkinter) и пишется в лог, а не
«молча вылетает».
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Константы
# --------------------------------------------------------------------------- #

APP_NAME = "LocalASR"
VENV_DIRNAME = ".venv"
STAMP_FILENAME = ".localasr-env.json"
LOCK_FILENAME = ".localasr-install.lock"
CHILD_ENV_FLAG = "LOCALASR_BOOTSTRAPPED"

MIN_PYTHON = (3, 10)
# Последняя версия, для которой наличие wheel'ов ВСЕХ зависимостей проверено
# вручную (cp314/win64+linux: ctranslate2, PySide6-abi3, numpy, av, onnxruntime,
# tokenizers, pywhispercpp, vosk). Это НЕ запрет: более новые версии тоже
# принимаются — см. _version_acceptable().
TESTED_PYTHON_MAX = (3, 14)
# Если в системе несколько Python, предпочитаем самые обкатанные.
PREFERRED_LAUNCHER_FLAGS = ("-3.12", "-3.13", "-3.11", "-3.10", "-3.14", "-3")

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"

# Дистрибутив PyPI -> имя импорта. Порядок = порядок установки.
# `app` проверяется последним, чтобы тяжёлые пакеты ставились первыми.
BASE_REQUIREMENTS: List[Tuple[str, str]] = [
    ("PySide6", "PySide6"),
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("imageio-ffmpeg", "imageio_ffmpeg"),
    ("faster-whisper", "faster_whisper"),
    ("av", "av"),
]

# Экстра-наборы, которые ставятся по требованию уже из GUI (не при первом запуске).
EXTRA_REQUIREMENTS: Dict[str, List[Tuple[str, str]]] = {
    "whispercpp": [("pywhispercpp", "pywhispercpp")],
    "vosk": [("vosk", "vosk")],
    "cuda": [
        ("nvidia-cublas-cu12", None),
        ("nvidia-cudnn-cu12", None),
    ],
}

PIP_TIMEOUT_SECONDS = 60


# --------------------------------------------------------------------------- #
#  Пути
# --------------------------------------------------------------------------- #


def project_root(start: Optional[str] = None) -> str:
    """Корень проекта = папка, в которой лежит bootstrap.py."""
    return os.path.dirname(os.path.abspath(start or __file__))


def venv_dir(root: str) -> str:
    return os.path.join(root, VENV_DIRNAME)


def venv_python(root: str) -> str:
    """Интерпретатор внутри .venv (pythonw НЕ возвращаем — он нужен только для GUI)."""
    vd = venv_dir(root)
    if IS_WINDOWS:
        return os.path.join(vd, "Scripts", "python.exe")
    return os.path.join(vd, "bin", "python3") if os.path.exists(
        os.path.join(vd, "bin", "python3")
    ) else os.path.join(vd, "bin", "python")


def venv_pythonw(root: str) -> str:
    """Интерпретатор без консоли (Windows) — именно им перезапускается GUI."""
    vd = venv_dir(root)
    if IS_WINDOWS:
        candidate = os.path.join(vd, "Scripts", "pythonw.exe")
        return candidate if os.path.exists(candidate) else venv_python(root)
    return venv_python(root)


def requirements_files(root: str) -> List[str]:
    req_dir = os.path.join(root, "requirements")
    primary = os.path.join(req_dir, "base.txt")
    if os.path.isfile(primary):
        return [primary]
    fallback = os.path.join(root, "requirements.txt")
    return [fallback] if os.path.isfile(fallback) else []


def stamp_path(root: str) -> str:
    return os.path.join(venv_dir(root), STAMP_FILENAME)


def log_path(root: str) -> str:
    return os.path.join(root, "logs", "bootstrap.log")


# --------------------------------------------------------------------------- #
#  Логирование (в файл + в callback для GUI-окна установки)
# --------------------------------------------------------------------------- #


class Logger:
    def __init__(self, path: str, echo: bool = True) -> None:
        self.path = path
        self.echo = echo
        self._lock = threading.Lock()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except OSError:
            pass

    def __call__(self, message: str = "", *, force_echo: bool = False) -> None:
        with self._lock:
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}".rstrip()
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
            if self.echo or force_echo:
                try:
                    print(line, flush=True)
                except (OSError, ValueError):
                    pass  # stdout закрыт (pythonw) — это нормально


# --------------------------------------------------------------------------- #
#  Поиск подходящего системного Python
# --------------------------------------------------------------------------- #


def _probe_python(cmd: Sequence[str]) -> Optional[Tuple[str, int]]:
    """Запускает интерпретатор и возвращает ("3.14.8", 64) или None.

    Битность читаем сразу: 32-битный Python не подходит (под него нет wheel'ов
    большинства зависимостей), а молча пропустить его — значит получить
    необъяснимый провал установки.
    """
    code = (
        "import sys,struct;"
        "print('%d.%d.%d %d' % (*sys.version_info[:3], struct.calcsize('P') * 8))"
    )
    try:
        probe = subprocess.run(
            [str(c) for c in cmd] + ["-c", code],
            capture_output=True,
            text=True,
            timeout=25,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if probe.returncode != 0:
        return None
    lines = (probe.stdout or "").strip().splitlines()
    if not lines:
        return None
    parts = lines[-1].split()
    if not parts:
        return None
    version = parts[0]
    bits = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return version, bits


def _version_tuple(version: str) -> Tuple[int, int]:
    try:
        parts = [int(x) for x in version.split(".")[:2]]
    except ValueError:
        return (0, 0)
    while len(parts) < 2:
        parts.append(0)
    return parts[0], parts[1]


def _version_acceptable(parts: Tuple[int, int]) -> bool:
    """Нижняя граница жёсткая, верхней НЕТ: новый Python — не ошибка."""
    return parts >= MIN_PYTHON


def _python_is_store_stub(exe: str) -> bool:
    """Заглушка Microsoft Store (`WindowsApps\\python.exe`) — не настоящий Python."""
    norm = exe.replace("\\", "/").lower()
    return "windowsapps" in norm or "microsoft.store" in norm


def find_base_python(log: Logger) -> Tuple[Optional[str], str]:
    """Ищет интерпретатор, которым можно создать venv. Возвращает (путь, отчёт)."""
    report: List[str] = []
    candidates: List[str] = []

    if IS_WINDOWS:
        launcher = shutil.which("py")
        if launcher:
            for flag in PREFERRED_LAUNCHER_FLAGS:
                candidates.append(f"{launcher} {flag}")
    candidates.append(sys.executable)

    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    if IS_WINDOWS:
        local = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python")
        if os.path.isdir(local):
            for entry in sorted(os.listdir(local), reverse=True):
                exe = os.path.join(local, entry, "python.exe")
                if os.path.isfile(exe):
                    candidates.append(exe)

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        parts = cand.split(" ")
        exe = parts[0]
        if _python_is_store_stub(exe):
            report.append(f"пропущен (заглушка Microsoft Store): {exe}")
            continue
        if not os.path.exists(exe):
            report.append(f"пропущен (не найден): {cand}")
            continue
        probed = _probe_python(parts)
        if probed is None:
            report.append(f"пропущен (не запускается или нет такой версии): {cand}")
            continue
        version, bits = probed
        if bits and bits != 64:
            report.append(f"пропущен ({bits}-битный, нужен 64-битный): {cand}")
            continue
        version_parts = _version_tuple(version)
        if not _version_acceptable(version_parts):
            report.append(
                f"пропущен (нужен Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, найден {version}): {cand}"
            )
            continue
        if version_parts > TESTED_PYTHON_MAX:
            note = (
                f"Python {version} новее проверенного "
                f"{TESTED_PYTHON_MAX[0]}.{TESTED_PYTHON_MAX[1]} — включаю поддержку "
                "(wheel'ы зависимостей используют стабильный ABI и его поддерживают)"
            )
            report.append(note)
            log(note)
        report.append(f"подходит: {cand} (Python {version}, {bits or '?'}-бит)")
        log(f"Базовый Python: {cand} -> {version}")
        return cand, "\n".join(report)

    report.append("ни один кандидат не подошёл")
    return None, "\n".join(report)


# --------------------------------------------------------------------------- #
#  Создание venv
# --------------------------------------------------------------------------- #


def _run(cmd: Sequence[str], log: Logger, timeout: int = 900, on_line=None) -> int:
    """Запуск команды с потоковым чтением stdout/stderr."""
    printable = " ".join(f'"{c}"' if " " in c else str(c) for c in cmd)
    log(f"$ {printable}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env.pop("PYTHONHOME", None)
    # Виртуальное окружение «родителя» не должно протекать в дочерний pip
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)

    creationflags = 0
    if IS_WINDOWS:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
    except OSError as exc:
        log(f"!! не удалось запустить команду: {exc}")
        return 127

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        log(line, force_echo=False)
        if on_line is not None:
            try:
                on_line(line)
            except Exception:
                pass
    proc.wait(timeout=timeout)
    log(f"-> код возврата {proc.returncode}")
    return proc.returncode


def create_venv(base_python: str, root: str, log: Logger, on_line=None) -> str:
    """Создаёт (или чинит) .venv. Возвращает путь к python внутри venv."""
    vd = venv_dir(root)
    py = venv_python(root)

    if os.path.isfile(py):
        probe = subprocess.run(
            [py, "-c", "print('ok')"], capture_output=True, text=True, timeout=60
        )
        if probe.returncode == 0:
            log("Виртуальное окружение уже существует — использую его.")
            return py
        log("Существующее .venv повреждено — пересоздаю.")
        try:
            shutil.rmtree(vd, ignore_errors=True)
        except OSError as exc:
            log(f"Не удалось удалить .venv: {exc}")

    base_cmd = base_python.split(" ")
    log(f"Создаю виртуальное окружение: {vd}")
    on_line and on_line("Создаю виртуальное окружение (.venv)...")
    code = _run(base_cmd + ["-m", "venv", vd], log, timeout=600, on_line=on_line)

    if code != 0 or not os.path.isfile(venv_python(root)):
        # Типичный случай Debian/Ubuntu: не установлен python3-venv (ensurepip).
        log("Обычный venv не получился — пробую --without-pip.")
        on_line and on_line("Пробую создать окружение без pip (--without-pip)...")
        shutil.rmtree(vd, ignore_errors=True)
        code = _run(
            base_cmd + ["-m", "venv", "--without-pip", vd],
            log,
            timeout=600,
            on_line=on_line,
        )
        if code != 0 or not os.path.isfile(venv_python(root)):
            raise RuntimeError(
                "Не удалось создать виртуальное окружение.\n"
                "Windows: переустанови Python с галочкой «Install launcher for all users»\n"
                "Linux (Debian/Ubuntu): sudo apt install python3-venv python3-full\n"
                f"\nПодробности в логе: {log.path}"
            )
        # добираем pip вручную
        on_line and on_line("Устанавливаю pip внутрь окружения...")
        _run(
            [venv_python(root), "-c", "import ensurepip, sys; ensurepip._bootstrap()",
             ],
            log,
            timeout=600,
            on_line=on_line,
        )
        if not _pip_works(venv_python(root), log):
            get_pip = os.path.join(vd, "get-pip.py")
            _download_file(
                "https://bootstrap.pypa.io/get-pip.py", get_pip, log
            )
            _run([venv_python(root), get_pip, "--no-warn-script-location"], log, 900, on_line)

    if not os.path.isfile(venv_python(root)):
        raise RuntimeError(f"venv создан, но интерпретатор не найден: {venv_python(root)}")

    # Актуальный pip = меньше сюрпризов с wheel/метаданными
    on_line and on_line("Обновляю pip...")
    _run(
        [venv_python(root), "-m", "pip", "install", "--upgrade",
         "pip>=24.0", "setuptools>=68", "wheel"],
        log,
        timeout=900,
        on_line=on_line,
    )
    return venv_python(root)


def _pip_works(py: str, log: Logger) -> bool:
    try:
        out = subprocess.run(
            [py, "-m", "pip", "--version"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return False
    log(f"pip: {out.stdout.strip() or out.stderr.strip()}")
    return out.returncode == 0


def _download_file(url: str, dest: str, log: Logger) -> bool:
    """stdlib-загрузка (urllib) — requests на этом этапе ещё может не быть."""
    import urllib.request

    log(f"Скачиваю {url}")
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        return True
    except Exception as exc:  # noqa: BLE001 - сеть может отказать по любой причине
        log(f"!! не удалось скачать {url}: {exc}")
        return False


# --------------------------------------------------------------------------- #
#  Проверка зависимостей
# --------------------------------------------------------------------------- #


def requirements_fingerprint(root: str, python_exe: str) -> str:
    payload = {
        "python": python_exe,
        "version": "%d.%d" % sys.version_info[:2],
        "platform": platform.platform(),
        "base": [dist for dist, _ in BASE_REQUIREMENTS],
        "files": {},
    }
    for path in requirements_files(root):
        try:
            with open(path, "rb") as fh:
                payload["files"][os.path.basename(path)] = hashlib.sha256(
                    fh.read()
                ).hexdigest()
        except OSError:
            continue
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _read_stamp(root: str) -> dict:
    try:
        with open(stamp_path(root), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_stamp(root: str, fingerprint: str, extras: Sequence[str]) -> None:
    data = {
        "fingerprint": fingerprint,
        "extras": sorted(extras),
        "python": sys.executable,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "app_version": "1.0.0",
    }
    try:
        os.makedirs(os.path.dirname(stamp_path(root)), exist_ok=True)
        tmp = stamp_path(root) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, stamp_path(root))
    except OSError:
        pass


def check_imports(python_exe: str, pairs: Sequence[Tuple[str, str]], log: Logger) -> List[str]:
    """Один вызов интерпретатора на все пакеты — так быстрее всего.

    Возвращает список дистрибутивов, которых не хватает.
    """
    active = [(dist, mod) for dist, mod in pairs if mod]
    if not active:
        return [dist for dist, mod in pairs if not mod]

    code = (
        "import importlib,json,sys\n"
        "mods=json.loads(sys.argv[1])\n"
        "missing=[]\n"
        "for dist,mod in mods:\n"
        "    try:\n"
        "        importlib.import_module(mod)\n"
        "    except BaseException:\n"
        "        missing.append(dist)\n"
        "print('MISSING:'+json.dumps(missing))\n"
    )
    payload = json.dumps(active)
    try:
        out = subprocess.run(
            [python_exe, "-c", code, payload],
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"!! проверка импортов упала: {exc} — считаю, что не хватает всего")
        return [dist for dist, _ in active]

    marker = "MISSING:"
    missing: List[str] = []
    for line in (out.stdout or "").splitlines():
        if line.startswith(marker):
            try:
                missing = json.loads(line[len(marker):])
            except ValueError:
                missing = [dist for dist, _ in active]
            break
    else:
        log(f"!! не получил ответ от проверки импортов: {out.stdout!r} {out.stderr!r}")
        missing = [dist for dist, _ in active]

    # пакеты без проверяемого импорта (например, cudnn) — считаем установленными,
    # если их требует уже установленный набор; отдельно не валидируем
    missing.extend(dist for dist, mod in pairs if not mod and False)
    return missing


# --------------------------------------------------------------------------- #
#  Установка пакетов (uv, если доступен, иначе pip)
# --------------------------------------------------------------------------- #


def _uv_executable(python_exe: str) -> Optional[str]:
    uv = shutil.which("uv")
    if uv:
        return uv
    scripts = os.path.dirname(python_exe)
    for cand in (os.path.join(scripts, "uv.exe" if IS_WINDOWS else "uv"),):
        if os.path.isfile(cand):
            return cand
    return None


def install_packages(
    python_exe: str,
    distributions: Sequence[str],
    root: str,
    log: Logger,
    on_line=None,
    extra_args: Optional[Sequence[str]] = None,
) -> int:
    if not distributions:
        return 0

    extra_args = list(extra_args or [])
    uv = _uv_executable(python_exe)
    if uv:
        cmd = [uv, "pip", "install", "--python", python_exe, *extra_args, *distributions]
        log("Использую uv (быстрая установка).")
        on_line and on_line(f"Ставлю (uv): {', '.join(distributions)}")
        code = _run(cmd, log, timeout=3600, on_line=on_line)
        if code == 0:
            return 0
        log("uv не справился — пробую pip.")

    on_line and on_line(f"Ставлю (pip): {', '.join(distributions)}")
    cmd = [
        python_exe,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--prefer-binary",
        "--timeout",
        str(PIP_TIMEOUT_SECONDS),
        "--retries",
        "3",
        *extra_args,
        *distributions,
    ]
    return _run(cmd, log, timeout=5400, on_line=on_line)


def install_requirements_file(
    python_exe: str, req_file: str, log: Logger, on_line=None
) -> int:
    uv = _uv_executable(python_exe)
    if uv:
        code = _run(
            [uv, "pip", "install", "--python", python_exe, "-r", req_file],
            log,
            timeout=5400,
            on_line=on_line,
        )
        if code == 0:
            return 0
    return _run(
        [
            python_exe, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-input", "--prefer-binary", "--timeout", str(PIP_TIMEOUT_SECONDS),
            "--retries", "3", "-r", req_file,
        ],
        log,
        timeout=5400,
        on_line=on_line,
    )


# --------------------------------------------------------------------------- #
#  Блокировка повторного запуска (чтобы два окна не ставили пакеты одновременно)
# --------------------------------------------------------------------------- #


class InstallLock:
    def __init__(self, root: str) -> None:
        self.path = os.path.join(venv_dir(root), LOCK_FILENAME)
        self.acquired = False

    def __enter__(self) -> "InstallLock":
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            self.acquired = True
        except FileExistsError:
            # если лок старый (>30 мин) — считаем его мёртвым
            try:
                if time.time() - os.path.getmtime(self.path) > 1800:
                    os.remove(self.path)
                    return self.__enter__()
            except OSError:
                pass
            self.acquired = False
        except OSError:
            self.acquired = True  # не критично: лучше продолжить
        return self

    def __exit__(self, *exc_info) -> None:
        if self.acquired:
            try:
                os.remove(self.path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
#  Перезапуск без консоли
# --------------------------------------------------------------------------- #


def relaunch_gui(root: str, python_exe: str, log: Logger, argv: Sequence[str]) -> int:
    """Стартует приложение в .venv и (на Windows) без консольного окна."""
    target = venv_pythonw(root) if not IS_WINDOWS or os.path.exists(venv_pythonw(root)) else python_exe
    app_entry = os.path.join(root, "run_app.py")
    if not os.path.isfile(app_entry):
        app_entry = os.path.join(root, "main.py")

    env = os.environ.copy()
    env[CHILD_ENV_FLAG] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["LOCALASR_ROOT"] = root
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)

    cmd = [target, app_entry, *argv]
    log("Перезапускаю GUI: " + " ".join(cmd))

    kwargs: dict = {"cwd": root, "env": env, "stdin": subprocess.DEVNULL}
    if IS_WINDOWS:
        # DETACHED_PROCESS и CREATE_NO_WINDOW — взаимоисключающие флаги,
        # вместе их передавать нельзя (CreateProcess вернёт ошибку).
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        kwargs.update(
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=detached | new_group,
        )
    else:
        kwargs.update(
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    try:
        proc = subprocess.Popen(cmd, **kwargs)
    except OSError as exc:
        log(f"!! не удалось перезапустить GUI: {exc}")
        # план Б: запускаем в текущем процессе (консоль останется, но приложение откроется)
        return 2

    # ждём пару секунд: если процесс сразу умер — сообщаем об ошибке
    try:
        code = proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        log(f"GUI запущен, pid={proc.pid}")
        return 0
    log(f"!! GUI немедленно завершился с кодом {code}")
    return code or 3


# --------------------------------------------------------------------------- #
#  Нативное окно установки (Tkinter) — видно прогресс, консоль не нужна
# --------------------------------------------------------------------------- #


class ProgressUI:
    """Мини-окно на Tkinter. Если Tk недоступен — молча печатает в консоль."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.ok = True
        self._root = None
        self._text = None
        self._status = None
        self._progress = None
        self._lock = threading.Lock()
        self._cancelled = False
        try:
            import tkinter as tk
            from tkinter import ttk

            self._tk = tk
            self._ttk = ttk
        except Exception:
            self.ok = False
            return

        try:
            root = tk.Tk()
            root.title(title)
            root.geometry("820x520")
            root.minsize(560, 360)
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass

            header = tk.Frame(root, bg="#15171c")
            header.pack(fill="x")
            tk.Label(
                header,
                text=f"{APP_NAME} — подготовка окружения",
                bg="#15171c",
                fg="#f2f4f8",
                font=("Segoe UI" if IS_WINDOWS else "TkDefaultFont", 14, "bold"),
                anchor="w",
                padx=16,
                pady=12,
            ).pack(fill="x")
            self._status = tk.Label(
                header,
                text="Инициализация...",
                bg="#15171c",
                fg="#9aa4b2",
                anchor="w",
                padx=16,
                pady=(0, 10),
            )
            self._status.pack(fill="x")

            self._progress = ttk.Progressbar(root, mode="indeterminate")
            self._progress.pack(fill="x", padx=16, pady=(8, 4))
            self._progress.start(12)

            self._text = tk.Text(
                root, bg="#0e1014", fg="#c9d1d9", insertbackground="#c9d1d9",
                wrap="none", font=("Consolas" if IS_WINDOWS else "TkFixedFont", 9),
            )
            self._text.pack(fill="both", expand=True, padx=16, pady=8)

            bottom = tk.Frame(root)
            bottom.pack(fill="x", padx=16, pady=(0, 12))
            tk.Label(
                bottom,
                text="Окно закроется автоматически, когда запустится интерфейс.",
                fg="#6b7280",
                anchor="w",
            ).pack(side="left")

            self._root = root
            root.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            self.ok = False

    # -- публичные методы -------------------------------------------------- #
    def set_status(self, message: str) -> None:
        if not self.ok:
            return
        with self._lock:
            try:
                self._root.after(0, lambda: self._status.configure(text=message))
            except Exception:
                pass

    def log_line(self, line: str) -> None:
        if not self.ok:
            return
        with self._lock:
            try:
                self._root.after(0, self._append, line[:400])
            except Exception:
                pass

    def _append(self, line: str) -> None:
        try:
            self._text.configure(state="normal")
            self._text.insert("end", line + "\n")
            # не раздуваем буфер бесконечно
            if int(self._text.index("end-1c").split(".")[0]) > 3000:
                self._text.delete("1.0", "500.0")
            self._text.see("end")
            self._text.configure(state="disabled")
        except Exception:
            pass

    def finish(self) -> None:
        if not self.ok:
            return
        try:
            self._progress.stop()
            self._root.destroy()
        except Exception:
            pass
        finally:
            self.ok = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _on_close(self) -> None:
        self._cancelled = True
        self.finish()

    def pump(self, seconds: float = 0.05) -> None:
        """Прокачать события Tk из рабочего потока (вызывается редко и безопасно)."""
        return None


# --------------------------------------------------------------------------- #
#  Основная логика
# --------------------------------------------------------------------------- #


class BootstrapError(RuntimeError):
    pass


_WHEEL_FALLBACK_HINT = (
    "Если ошибка выглядит как «нет wheels под вашу версию Python»:\n"
    "поставь рядом Python 3.12 или 3.13 (https://www.python.org/downloads/) —\n"
    "при следующем запуске LocalASR автоматически выберет его, ничего удалять не нужно."
)


def run(root: Optional[str] = None, argv: Optional[Sequence[str]] = None) -> int:
    root = root or project_root()
    argv = list(argv if argv is not None else sys.argv[1:])

    # сервисные режимы (можно звать даже без готового окружения)
    if "--doctor" in argv:
        return doctor(root)
    if "--force-install" in argv:
        os.environ["LOCALASR_FORCE_INSTALL"] = "1"
        argv = [a for a in argv if a != "--force-install"]
    if os.environ.get("LOCALASR_FORCE_INSTALL") == "1":
        argv = [a for a in argv if a != "--force-install"]

    os.makedirs(os.path.join(root, "logs"), exist_ok=True)

    already_child = os.environ.get(CHILD_ENV_FLAG) == "1"
    log = Logger(log_path(root), echo=not already_child and sys.stdout is not None)
    log("=" * 70)
    log(f"{APP_NAME} bootstrap | python={sys.executable} | {platform.platform()}")

    # 1. Мы уже внутри правильного venv? -> быстрый путь, сразу GUI.
    py = venv_python(root)
    if already_child or _same_interpreter(sys.executable, py):
        missing = _quick_check(py, root, log)
        if not missing:
            if already_child:
                log("Окружение готово — запускаю GUI в этом же процессе.")
                return _run_gui_inprocess(root, argv, log)
            _write_stamp(root, requirements_fingerprint(root, py), [])
            return relaunch_gui(root, py, log, argv)
        log(f"В venv не хватает: {missing} — доустанавливаю.")

    # 2. Нужна полноценная подготовка окружения.
    ui = ProgressUI(f"{APP_NAME} — установка")
    on_line = ui.log_line if ui.ok else None
    ui.set_status("Ищу подходящий Python...") if ui.ok else None

    try:
        base_python, report = find_base_python(log)
        if base_python is None:
            raise BootstrapError(
                "Не найден подходящий Python.\n\n"
                f"Нужен Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} или новее, 64-бит "
                f"(проверено вплоть до {TESTED_PYTHON_MAX[0]}.{TESTED_PYTHON_MAX[1]}; "
                "более новые версии тоже принимаются).\n"
                "Скачай с https://www.python.org/downloads/ и при установке отметь\n"
                "«Add python.exe to PATH». 32-битный Python не подойдёт.\n\n"
                "Что проверено:\n" + textwrap.indent(report, "  ")
            )

        with InstallLock(root) as lock:
            if not lock.acquired:
                raise BootstrapError(
                    "Другое окно LocalASR уже устанавливает окружение.\n"
                    "Дождись его окончания и запусти main.py ещё раз."
                )

            ui.set_status("Создаю виртуальное окружение...") if ui.ok else None
            py = create_venv(base_python, root, log, on_line=on_line)

            fingerprint = requirements_fingerprint(root, py)
            stamp = _read_stamp(root)
            force = os.environ.get("LOCALASR_FORCE_INSTALL") == "1"

            if stamp.get("fingerprint") == fingerprint and not force:
                missing = _quick_check(py, root, log)
            else:
                ui.set_status("Устанавливаю зависимости (это может занять несколько минут)...") if ui.ok else None
                for req in requirements_files(root):
                    code = install_requirements_file(py, req, log, on_line=on_line)
                    if code != 0:
                        raise BootstrapError(
                            f"pip/uv завершился с ошибкой (код {code}) при установке {os.path.basename(req)}.\n"
                            f"Полный лог: {log.path}\n\n" + _WHEEL_FALLBACK_HINT
                        )
                missing = _quick_check(py, root, log)

            if missing:
                ui.set_status(f"Доустанавливаю: {', '.join(missing)}") if ui.ok else None
                code = install_packages(py, missing, root, log, on_line=on_line)
                if code != 0:
                    raise BootstrapError(
                        "Не удалось установить пакеты: " + ", ".join(missing) +
                        f"\n\nПолный лог: {log.path}\n\n" + _WHEEL_FALLBACK_HINT
                    )
                missing = _quick_check(py, root, log)

            if missing:
                raise BootstrapError(
                    "После установки пакеты всё ещё не импортируются: " + ", ".join(missing) +
                    f"\n\nВозможные причины: антивирус заблокировал файлы, нет места на диске, "
                    f"или несовместимая версия Python.\nПолный лог: {log.path}"
                )

            _write_stamp(root, fingerprint, [])
            log("Окружение готово.")

        ui.set_status("Запускаю LocalASR...") if ui.ok else None
        code = relaunch_gui(root, py, log, argv)
        ui.finish()
        if code != 0:
            raise BootstrapError(
                f"Приложение не запустилось (код {code}).\n"
                f"Лог запуска: {os.path.join(root, 'logs', 'localasr.log')}\n"
                f"Лог установки: {log.path}"
            )
        return 0

    except BootstrapError as exc:
        ui.finish()
        _show_error(str(exc), log)
        return 1
    except Exception:  # noqa: BLE001
        ui.finish()
        import traceback

        _show_error(
            "Непредвиденная ошибка при подготовке окружения:\n\n"
            + traceback.format_exc(limit=12),
            log,
        )
        return 1


def _quick_check(py: str, root: str, log: Logger) -> List[str]:
    return check_imports(py, BASE_REQUIREMENTS, log)


def _same_interpreter(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except (OSError, AttributeError, ValueError):
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _run_gui_inprocess(root: str, argv: Sequence[str], log: Logger) -> int:
    """Мы уже в готовом venv — запускаем GUI без перезапуска процесса."""
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from app import cli  # локальный импорт: к этому моменту зависимости уже есть
    except Exception as exc:  # noqa: BLE001
        log(f"Не удалось импортировать приложение: {exc!r}")
        _show_error(
            "Не удалось загрузить модули приложения.\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            "Попробуй удалить папку .venv и запустить main.py заново — "
            "окружение пересоздастся автоматически.\n\n"
            f"Лог: {log.path}",
            log,
        )
        return 1
    return cli.main(argv)


def _show_error(message: str, log: Logger) -> None:
    log("ОШИБКА: " + message.replace("\n", " | "))
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror(f"{APP_NAME} — ошибка запуска", message, parent=root)
        root.destroy()
    except Exception:
        pass

    print("\n" + "=" * 70, file=sys.stderr)
    print(f"[{APP_NAME}] ОШИБКА:", file=sys.stderr)
    print(message, file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    # держим консоль открытой: при двойном клике она иначе исчезнет мгновенно
    if not os.environ.get("LOCALASR_NO_PAUSE"):
        try:
            input("\nНажми Enter, чтобы закрыть окно...")
        except (EOFError, KeyboardInterrupt, OSError, ValueError):
            pass
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
#  Сервисные команды (для scripts/ и отладки)
# --------------------------------------------------------------------------- #


def doctor(root: Optional[str] = None) -> int:
    """Диагностика окружения: python, venv, ffmpeg, зависимости, место на диске."""
    root = root or project_root()
    log = Logger(log_path(root), echo=True)
    print(f"{APP_NAME} doctor")
    print("-" * 60)
    print(f"Корень проекта : {root}")
    print(f"Система        : {platform.platform()}")
    print(f"Текущий Python : {sys.executable} ({sys.version.split()[0]})")
    base, report = find_base_python(log)
    print(f"Базовый Python : {base or 'НЕ НАЙДЕН'}")
    print(textwrap.indent(report, "  "))
    py = venv_python(root)
    print(f"venv python    : {py} ({'есть' if os.path.isfile(py) else 'нет'})")
    if os.path.isfile(py):
        missing = check_imports(py, BASE_REQUIREMENTS, log)
        print(f"Не хватает     : {missing or 'ничего, всё установлено'}")
    stamp = _read_stamp(root)
    print(f"Отпечаток env  : {stamp.get('fingerprint', '—')}")
    try:
        free = shutil.disk_usage(root).free / (1024 ** 3)
        print(f"Свободно на FS : {free:.1f} GiB")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    if "--doctor" in sys.argv:
        sys.exit(doctor())
    sys.exit(run())
