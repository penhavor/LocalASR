"""Тесты bootstrap — самого критичного места (первый запуск, создание .venv)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bootstrap  # noqa: E402


def test_base_requirements_are_declared() -> None:
    dists = [dist for dist, _ in bootstrap.BASE_REQUIREMENTS]
    for required in ("PySide6", "faster-whisper", "numpy", "requests"):
        assert required in dists, f"в базовых зависимостях нет {required}"


def test_every_base_requirement_has_import_name() -> None:
    for dist, module in bootstrap.BASE_REQUIREMENTS:
        assert dist
        assert module, f"для {dist} не указано имя импорта — проверка не сработает"


def test_requirements_file_matches_bootstrap_list() -> None:
    root = Path(bootstrap.project_root())
    req = root / "requirements" / "base.txt"
    assert req.is_file(), "requirements/base.txt обязателен"
    text = req.read_text(encoding="utf-8")
    packages = {
        line.split("=")[0].split(">")[0].split("<")[0].split("[")[0].strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for dist, _ in bootstrap.BASE_REQUIREMENTS:
        assert dist.lower() in packages, f"{dist} есть в bootstrap, но отсутствует в requirements/base.txt"


def test_check_imports_detects_missing(tmp_path: Path) -> None:
    python = sys.executable
    missing = bootstrap.check_imports(
        python,
        [("os", "os"), ("definitely-not-installed-xyz", "definitely_not_installed_xyz")],
        bootstrap.Logger(str(tmp_path / "log.txt"), echo=False),
    )
    assert missing == ["definitely-not-installed-xyz"]


def test_check_imports_reports_all_present(tmp_path: Path) -> None:
    missing = bootstrap.check_imports(
        sys.executable,
        [("json-mod", "json"), ("os-mod", "os"), ("pathlib-mod", "pathlib")],
        bootstrap.Logger(str(tmp_path / "log.txt"), echo=False),
    )
    assert missing == []


def test_check_imports_survives_broken_interpreter(tmp_path: Path) -> None:
    log = bootstrap.Logger(str(tmp_path / "log.txt"), echo=False)
    missing = bootstrap.check_imports("/nonexistent/python", [("json-mod", "json")], log)
    assert missing == ["json-mod"], "при сбое проверки нужно считать пакет отсутствующим"


def test_fingerprint_is_stable_and_changes(tmp_path: Path) -> None:
    (tmp_path / "requirements").mkdir()
    req = tmp_path / "requirements" / "base.txt"
    req.write_text("PySide6>=6.7\n", encoding="utf-8")

    first = bootstrap.requirements_fingerprint(str(tmp_path), "/x/python")
    second = bootstrap.requirements_fingerprint(str(tmp_path), "/x/python")
    assert first == second

    req.write_text("PySide6>=6.8\n", encoding="utf-8")
    assert bootstrap.requirements_fingerprint(str(tmp_path), "/x/python") != first


def test_stamp_roundtrip(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    bootstrap._write_stamp(str(tmp_path), "abc123", ["vosk"])
    data = bootstrap._read_stamp(str(tmp_path))
    assert data["fingerprint"] == "abc123"
    assert data["extras"] == ["vosk"]
    assert bootstrap._read_stamp(str(tmp_path / "missing")) == {}


def test_install_lock(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    with bootstrap.InstallLock(str(tmp_path)) as first:
        assert first.acquired
        with bootstrap.InstallLock(str(tmp_path)) as second:
            assert not second.acquired, "второй процесс не должен ставить параллельно"


def test_install_lock_recovers_from_stale(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    venv.mkdir()
    lock = venv / bootstrap.LOCK_FILENAME
    lock.write_text("999999", encoding="utf-8")
    os.utime(lock, (0, 0))  # делаем лок «древним»
    with bootstrap.InstallLock(str(tmp_path)) as acquired:
        assert acquired.acquired


def test_venv_python_paths(tmp_path: Path) -> None:
    python = bootstrap.venv_python(str(tmp_path))
    if os.name == "nt":
        assert python.endswith(os.path.join("Scripts", "python.exe"))
    else:
        assert python.endswith("/bin/python3") or python.endswith("/bin/python")
    assert Path(python).parts[-3] == ".venv"
    assert Path(bootstrap.venv_dir(str(tmp_path))).name == ".venv"


def test_pythonw_is_used_on_windows_for_gui(tmp_path: Path) -> None:
    """На Windows GUI перезапускается pythonw.exe — иначе консоль не пропадёт."""
    pythonw = bootstrap.venv_pythonw(str(tmp_path))
    assert Path(pythonw).name.lower() in ("pythonw.exe", "python.exe", "python3", "python")
    if os.name == "nt":
        # если pythonw.exe есть в .venv — должен использоваться именно он
        candidate = Path(pythonw).with_name("pythonw.exe")
        candidate.write_text("", encoding="utf-8")
        assert Path(bootstrap.venv_pythonw(str(tmp_path))).name.lower() == "pythonw.exe"


def test_find_base_python_returns_something_or_report(tmp_path: Path) -> None:
    log = bootstrap.Logger(str(tmp_path / "log.txt"), echo=False)
    python, report = bootstrap.find_base_python(log)
    assert report
    if python is None:
        assert "не подошёл" in report or "пропущен" in report
    else:
        assert os.path.exists(python.split(" ")[0])


def test_version_gate_accepts_new_python() -> None:
    """3.14, 3.15 и дальше — принимаются: верхнего потолка больше нет."""
    assert bootstrap._version_acceptable((3, 10))
    assert bootstrap._version_acceptable((3, 12))
    assert bootstrap._version_acceptable((3, 14))
    assert bootstrap._version_acceptable((3, 15))
    assert bootstrap._version_acceptable((4, 0))
    assert not bootstrap._version_acceptable((3, 9))
    assert not bootstrap._version_acceptable((2, 7))


def test_version_tuple_parsing() -> None:
    assert bootstrap._version_tuple("3.14.8") == (3, 14)
    assert bootstrap._version_tuple("3.10") == (3, 10)
    assert bootstrap._version_tuple("барахло") == (0, 0)


def test_find_base_python_accepts_314_and_rejects_32bit(monkeypatch) -> None:
    """Имитируем машину пользователя: единственный Python — 3.14 64-bit."""
    def fake_probe(cmd):
        joined = " ".join(str(c) for c in cmd)
        # лаунчер-флаги старых версий «не установлены», всё остальное — 3.14 64-bit
        for flag in ("-3.12", "-3.13", "-3.11", "-3.10"):
            if flag in joined:
                return None
        return "3.14.8", 64

    monkeypatch.setattr(bootstrap, "_probe_python", fake_probe)
    monkeypatch.setattr(bootstrap.os.path, "exists", lambda p: True)
    monkeypatch.setattr(bootstrap, "_python_is_store_stub", lambda p: False)
    log = bootstrap.Logger(os.devnull, echo=False)
    exe, report = bootstrap.find_base_python(log)
    assert exe is not None, report
    assert "3.14" in report


def test_find_base_python_rejects_32bit(monkeypatch) -> None:
    def fake_probe(cmd):
        return "3.12.4", 32

    monkeypatch.setattr(bootstrap, "_probe_python", fake_probe)
    monkeypatch.setattr(bootstrap.os.path, "exists", lambda p: True)
    monkeypatch.setattr(bootstrap, "_python_is_store_stub", lambda p: False)
    log = bootstrap.Logger(os.devnull, echo=False)
    exe, report = bootstrap.find_base_python(log)
    assert exe is None
    assert "32-битный" in report


def test_store_stub_python_is_rejected() -> None:
    if os.name != "nt":
        pytest.skip("актуально только для Windows")
    assert bootstrap._python_is_store_stub(r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\python.exe")


def test_run_reports_missing_python_gracefully(tmp_path: Path, monkeypatch) -> None:
    """Если Python не найден — должно быть понятное сообщение, а не трейсбек."""
    monkeypatch.setattr(bootstrap, "find_base_python", lambda log: (None, "ничего не нашёл"))
    monkeypatch.setattr(bootstrap, "ProgressUI", lambda title: _NoUI())
    monkeypatch.setattr(bootstrap, "_show_error", lambda message, log: _NoUI.messages.append(message))

    code = bootstrap.run(root=str(tmp_path))
    assert code == 1
    assert _NoUI.messages, "пользователь должен увидеть ошибку"
    assert "Python" in _NoUI.messages[-1]


def test_run_uses_existing_venv_when_complete(tmp_path: Path, monkeypatch) -> None:
    """Если окружение готово — никакого pip, сразу запуск GUI."""
    calls: list[str] = []

    venv = tmp_path / ".venv"
    (venv / "Scripts" if os.name == "nt" else venv / "bin").mkdir(parents=True)
    fake_python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "venv_python", lambda root: str(fake_python))
    monkeypatch.setattr(bootstrap, "venv_pythonw", lambda root: str(fake_python))
    monkeypatch.setattr(bootstrap, "_quick_check", lambda py, root, log: [])
    monkeypatch.setattr(bootstrap, "create_venv", lambda base_python, root, log, on_line=None: str(fake_python))
    monkeypatch.setattr(
        bootstrap,
        "relaunch_gui",
        lambda root, py, log, argv: calls.append("relaunch") or 0,
    )
    monkeypatch.setattr(bootstrap, "requirements_files", lambda root: [])

    code = bootstrap.run(root=str(tmp_path))
    assert code == 0
    assert calls == ["relaunch"], "при готовом окружении должен быть только перезапуск GUI"


def test_doctor_runs(tmp_path: Path, capsys) -> None:
    assert bootstrap.doctor(str(tmp_path)) == 0
    captured = capsys.readouterr()
    assert "doctor" in captured.out


def test_bootstrap_does_not_import_app_at_module_level() -> None:
    """bootstrap обязан работать на голом Python: никаких app.* / PySide6 на уровне модуля."""
    source = Path(bootstrap.__file__).read_text(encoding="utf-8")
    tree_lines = [
        line
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for line in tree_lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
            assert "app." not in stripped, f"импорт уровня модуля из app: {stripped}"
            assert "PySide6" not in stripped, f"импорт PySide6 на уровне модуля: {stripped}"
            assert "requests" not in stripped, f"импорт requests на уровне модуля: {stripped}"


class _NoUI:
    messages: list[str] = []

    ok = False

    def set_status(self, message: str) -> None:
        return None

    def log_line(self, line: str) -> None:
        return None

    def finish(self) -> None:
        return None
