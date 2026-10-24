# -*- coding: utf-8 -*-
"""Тесты логики установки пакетов.

1) Выбор индекса PyTorch на Windows+NVIDIA (_with_cuda_index).
   Регрессия: смешанный список (torch + transformers + …) с --index-url
   полностью заменял PyPI индексом PyTorch, где нет transformers/hydra-core —
   установка падала с кодом 1. Теперь для смешанных списков CUDA-индекс
   подключается как ДОПОЛНИТЕЛЬНЫЙ (--extra-index-url).
2) Превращение ModuleNotFoundError кода модели в EngineNotInstalled —
   GUI показывает диалог установки вместо стектрейса.

Запуск:  cd <корень проекта> && QT_QPA_PLATFORM=offscreen python tests/test_install_logic.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.constants import TORCH_CUDA_INDEX_URL  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

GIGAAM_PKGS = [
    "torch>=2.6", "torchaudio>=2.6", "transformers>=4.52",
    "hydra-core>=1.3,<2", "omegaconf>=2.3,<3", "sentencepiece>=0.2.2",
    "soundfile>=0.12",
]
TORCH_ONLY = ["torch>=2.6", "torchaudio>=2.6"]
HW_NVIDIA = {"has_nvidia": True}


def test_mixed_list_uses_extra_index() -> None:
    """Смешанный список: PyTorch-индекс дополнительный, PyPI не теряется."""
    with mock.patch.object(os, "name", "nt"):
        out = MainWindow._with_cuda_index(list(GIGAAM_PKGS), HW_NVIDIA)
    assert out[0] == "--extra-index-url", out
    assert out[1] == TORCH_CUDA_INDEX_URL
    assert out[2:] == GIGAAM_PKGS


def test_torch_only_uses_strict_index() -> None:
    """Только torch-пакеты: строгая замена индекса (CUDA-сборка гарантирована)."""
    with mock.patch.object(os, "name", "nt"):
        out = MainWindow._with_cuda_index(list(TORCH_ONLY), HW_NVIDIA)
    assert out[0] == "--index-url", out
    assert out[1] == TORCH_CUDA_INDEX_URL
    assert out[2:] == TORCH_ONLY


def test_explicit_index_not_overridden() -> None:
    with mock.patch.object(os, "name", "nt"):
        manual = ["--index-url", TORCH_CUDA_INDEX_URL, *TORCH_ONLY]
        assert MainWindow._with_cuda_index(list(manual), HW_NVIDIA) == manual
        manual2 = ["--extra-index-url", TORCH_CUDA_INDEX_URL, *GIGAAM_PKGS]
        assert MainWindow._with_cuda_index(list(manual2), HW_NVIDIA) == manual2


def test_no_index_without_nvidia_or_on_linux() -> None:
    with mock.patch.object(os, "name", "nt"):
        assert MainWindow._with_cuda_index(list(GIGAAM_PKGS), {"has_nvidia": False}) == GIGAAM_PKGS
    with mock.patch.object(os, "name", "posix"):
        assert MainWindow._with_cuda_index(list(GIGAAM_PKGS), HW_NVIDIA) == GIGAAM_PKGS


def test_non_torch_list_untouched() -> None:
    pkgs = ["pywhispercpp>=1.4.0"]
    with mock.patch.object(os, "name", "nt"):
        assert MainWindow._with_cuda_index(list(pkgs), HW_NVIDIA) == pkgs


def test_module_error_converted_to_install_offer() -> None:
    """ModuleNotFoundError кода модели -> EngineNotInstalled со списком пакетов."""
    from app.asr.base import (
        EngineNotInstalled,
        engine_not_installed_from_exc,
        pip_package_for_module,
    )

    exc = ModuleNotFoundError("No module named 'soundfile'", name="soundfile")
    out = engine_not_installed_from_exc("gigaam", exc)
    assert isinstance(out, EngineNotInstalled), out
    assert "soundfile>=0.12" in out.packages, out.packages   # из engines.json
    assert "torch>=2.6" in out.packages                       # остальные пакеты движка

    # прочие исключения не превращаются в диалог установки
    assert engine_not_installed_from_exc("gigaam", ValueError("x")) is None

    # карта модуль -> pip-пакет
    assert pip_package_for_module("hydra") == "hydra-core"
    assert pip_package_for_module("pyannote.audio") == "pyannote.audio"
    assert pip_package_for_module("soundfile") == "soundfile"


if __name__ == "__main__":
    test_mixed_list_uses_extra_index()
    print("  ✓ смешанный список -> --extra-index-url")
    test_torch_only_uses_strict_index()
    print("  ✓ только torch -> --index-url")
    test_explicit_index_not_overridden()
    print("  ✓ явный индекс не перебивается")
    test_no_index_without_nvidia_or_on_linux()
    print("  ✓ без NVIDIA / не Windows — список не трогается")
    test_non_torch_list_untouched()
    print("  ✓ список без torch не трогается")
    test_module_error_converted_to_install_offer()
    print("  ✓ ModuleNotFoundError -> диалог установки")
    print("\nТЕСТ ЛОГИКИ УСТАНОВКИ ПРОЙДЕН")
