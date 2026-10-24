# -*- coding: utf-8 -*-
"""Регрессионные тесты Qwen3-ASR (задача 5a) — без реальных весов.

Баг из логов пользователя:
    ValueError: Invalid input type. Must be a single audio, a list of audio,
    or a nested list of audio
Возникал потому, что в apply_transcription_request передавался КОРТЕЖ
(array, sample_rate), а transformers 5.x принимает массив/путь/URL.

Проверяется:
  * формат передачи аудио проходит валидатор transformers (make_list_of_audio);
  * старый формат (tuple) действительно отклоняется — тест ловит регрессию;
  * _qwen_inputs вызывает apply_transcription_request с audio=<ndarray>;
  * каскад откатов при TypeError (сигнатура без prompt/language);
  * OOM-бисекция: кусок делится пополам, тексты склеиваются;
  * is_available(): пин transformers>=5.13 через метаданные (без импорта);
  * подсказки об ошибках: «недокачанные файлы» и «старый transformers»;
  * РЕГРЕССИЯ «requires accelerate»: lru-кэш transformers, запомнивший
    отсутствие accelerate до его установки, сбрасывается — и полная цепочка
    check_and_set_device_map(device_map) снова работает БЕЗ перезапуска;
  * каскад загрузки весов: device_map -> сброс кэшей -> путь без accelerate
    (загрузка + .to(device)), когда accelerate реально отсутствует.

Нужен только transformers (torch — для OOM-ветки; без него шаги skip).

Запуск:  cd <корень проекта> && python tests/test_qwen_fixes.py
"""
from __future__ import annotations

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("LOCALASR_QUIET", "1")

import numpy as np  # noqa: E402

from app.asr.torch_engines import (  # noqa: E402
    QwenASREngine,
    missing_files_hint,
    transformers_version_hint,
)
from app.constants import SAMPLE_RATE  # noqa: E402

failures: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


def skip(message: str) -> None:
    print(f"  - пропуск: {message}")


class FakeProcessor:
    """Процессор, повторяющий контракт Qwen3ASRProcessor (валидация входов)."""

    def __init__(self, mode: str = "full") -> None:
        self.mode = mode          # full | no_prompt | no_language
        self.calls: list[dict] = []

    def apply_transcription_request(self, audio=None, language=None, prompt=None, **kwargs):
        if self.mode == "no_prompt" and prompt is not None:
            raise TypeError("apply_transcription_request() got an unexpected keyword argument 'prompt'")
        if self.mode == "no_language" and language is not None:
            raise TypeError("apply_transcription_request() got an unexpected keyword argument 'language'")
        # повторяем реальную валидацию transformers 5.x
        from transformers.audio_utils import make_list_of_audio_chat_template

        items = list(make_list_of_audio_chat_template(audio))
        self.calls.append({"audio": items[0], "language": language, "prompt": prompt})
        return {"input_ids": items, "ok": True}

    def decode(self, ids, return_format="raw", **kwargs):
        return [{"transcription": "распознанный текст", "language": "Russian"}]


def test_audio_format_accepted_by_transformers() -> None:
    print("== формат аудио проходит валидатор transformers ==")
    try:
        from transformers.audio_utils import make_list_of_audio
    except ImportError:
        skip("нет transformers")
        return
    chunk = np.zeros(SAMPLE_RATE, dtype=np.float32)
    try:
        make_list_of_audio(chunk)
        check(True, "ndarray 16 кГц принимается make_list_of_audio")
    except ValueError as exc:
        check(False, f"ndarray должен приниматься: {exc}")
    try:
        make_list_of_audio((chunk, SAMPLE_RATE))
        check(False, "кортеж (array, sr) НЕ должен приниматься (старый баг)")
    except ValueError:
        check(True, "кортеж (array, sr) отклоняется — источник бага подтверждён")


def test_qwen_inputs_contract() -> None:
    print("== _qwen_inputs: audio=массив, каскад откатов ==")
    engine = QwenASREngine.__new__(QwenASREngine)   # без __init__/torch
    chunk = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

    proc = FakeProcessor("full")
    engine._processor = proc
    engine._qwen_inputs(chunk, "ru", "термины")
    call = proc.calls[-1]
    check(isinstance(call["audio"], np.ndarray), "в процессор уходит ndarray, а не кортеж")
    check(call["language"] == "ru" and call["prompt"] == "термины", "language/prompt переданы")

    proc2 = FakeProcessor("no_prompt")
    engine._processor = proc2
    engine._qwen_inputs(chunk, "ru", "термины")
    check(proc2.calls and proc2.calls[-1]["prompt"] is None, "откат: вызов без prompt")

    proc3 = FakeProcessor("no_language")
    engine._processor = proc3
    engine._qwen_inputs(chunk, "ru", None)
    check(proc3.calls and proc3.calls[-1]["language"] is None, "откат: вызов без language")

    engine._processor = object()
    try:
        engine._qwen_inputs(chunk, None, None)
        check(False, "без apply_transcription_request должна быть понятная ошибка")
    except Exception as exc:
        check("transformers" in str(exc), f"старый transformers -> подсказка обновить: {str(exc)[:60]}…")


def test_oom_bisection() -> None:
    print("== OOM-бисекция длинного куска ==")
    try:
        import torch  # noqa: F401
    except ImportError:
        skip("нет torch")
        return
    import torch

    engine = QwenASREngine.__new__(QwenASREngine)
    engine._processor = FakeProcessor("full")
    calls = {"n": 0}

    class FakeModel:
        device = "cpu"
        dtype = torch.float32

    original_once = QwenASREngine._infer_qwen_once

    def flaky_once(self, model, chunk, language, prompt):
        calls["n"] += 1
        if len(chunk) > int(20.0 * SAMPLE_RATE):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return original_once(self, model, chunk, language, prompt)

    QwenASREngine._infer_qwen_once = flaky_once
    try:
        # «длинный» кусок 45 с -> должен разделиться до <=20-секундных
        chunk = np.zeros(int(45 * SAMPLE_RATE), dtype=np.float32)
        engine._processor = types.SimpleNamespace(
            apply_transcription_request=lambda audio, language, prompt: {"input_ids": [0]},
            decode=lambda ids, return_format="raw", **kw: [
                {"transcription": "кусок текста", "language": "Russian"}
            ],
        )

        class _Inputs(dict):
            def to(self, *a, **k):
                return self

        engine._processor.apply_transcription_request = (
            lambda **kw: _Inputs(input_ids=torch.zeros((1, 1), dtype=torch.long))
        )
        engine._model = None

        class _FakeGen:
            def __init__(self):
                self.device = "cpu"
                self.dtype = torch.float32

            def generate(self, **kwargs):
                return torch.zeros((1, 5), dtype=torch.long)

        # подменяем model.generate через атрибут движка
        engine._ensure_model = lambda: _FakeGen()
        text, lang = engine._infer_qwen(engine._ensure_model(), chunk, "ru", None, 0.0)
        check(calls["n"] >= 3, f"длинный кусок разделён (вызовов: {calls['n']})")
        check("кусок текста" in text, f"тексты половинок склеены: {text[:60]!r}")
        check(lang == "Russian", "язык определёлся")
    finally:
        QwenASREngine._infer_qwen_once = original_once


def test_is_available_pin() -> None:
    print("== is_available: transformers>=5.13 ==")
    import importlib.metadata as md

    original = md.version

    def fake_version(name):
        if name == "transformers":
            return fake_version.value
        return original(name)

    fake_version.value = "5.20.2"
    # класс проверяет только метаданные: torch/accelerate могут отсутствовать —
    # подменим проверку базовых модулей
    parent_available = QwenASREngine.__mro__[1].is_available
    QwenASREngine.__mro__[1].is_available = classmethod(lambda cls: True)
    try:
        md.version = fake_version
        fake_version.value = "5.20.2"
        check(QwenASREngine.is_available() is True, "transformers 5.20.2 — доступен")
        fake_version.value = "5.13.0"
        check(QwenASREngine.is_available() is True, "transformers 5.13.0 — доступен (граница)")
        fake_version.value = "5.12.1"
        check(QwenASREngine.is_available() is False, "transformers 5.12.1 — НЕ доступен (нет qwen3_asr)")
        fake_version.value = "4.57.6"
        check(QwenASREngine.is_available() is False, "transformers 4.57.6 — НЕ доступен")
    finally:
        md.version = original
        QwenASREngine.__mro__[1].is_available = parent_available

    from app.asr.base import engine_packages

    pkgs = engine_packages("qwen-asr")
    check("transformers>=5.13" in pkgs, f"пин в engine_packages: {pkgs}")


def test_error_hints() -> None:
    print("== подсказки в сообщениях об ошибках ==")
    exc = OSError(
        "Can't load feature extractor for 'models/qwen3-asr-1.7b'. make sure "
        "'models/qwen3-asr-1.7b' is the correct path to a directory containing "
        "a preprocessor_config.json file"
    )
    hint = missing_files_hint(exc)
    check("не полностью" in hint and "Удалить" in hint, "ошибка про preprocessor_config -> подсказка переустановить")

    exc2 = ValueError("Unrecognized configuration class <class 'x'> for this kind of AutoModel")
    hint2 = transformers_version_hint(exc2, "Qwen3-ASR", "5.13")
    check("transformers>=5.13" in hint2, "неизвестная модель -> подсказка обновить transformers")

    check(missing_files_hint(ValueError("обычная ошибка")) == "", "обычная ошибка без ложных подсказок")


def test_transformers_cache_reset() -> None:
    print("== сброс lru-кэшей transformers (регрессия «requires accelerate») ==")
    try:
        import transformers.utils.import_utils as iu
    except ImportError:
        skip("нет transformers")
        return
    from app.asr.torch_engines import reset_transformers_caches

    fn = iu.is_accelerate_available
    original = iu._is_package_available

    def fake_missing(name, *args, **kwargs):
        # симулируем «accelerate ещё не установлен» — как в момент, когда
        # GigaAM грузился до установки пакетов для Qwen
        if name == "accelerate":
            if kwargs.get("return_version"):
                return False, "0.0.0"
            return False
        return original(name, *args, **kwargs)

    # 1) кэшируем устаревшее «accelerate нет»
    fn.cache_clear()
    iu._is_package_available = fake_missing
    try:
        stale = fn()
    finally:
        iu._is_package_available = original
    check(stale is False, "до сброса transformers кэшировал «accelerate нет» (False)")

    # 2) пакет «установлен»: повторный вызов без сброса вернул бы то же False,
    #    а после reset_transformers_caches проверка выполняется заново
    still_stale = fn()
    check(still_stale is False, "без сброса кэш продолжает отдавать устаревшее False")
    cleared = reset_transformers_caches()
    check(cleared >= 1, f"reset_transformers_caches сбросил кэши ({cleared})")
    fresh = fn()
    check(fresh is True,
          "после сброса transformers «увидел» свежеустановленный accelerate (True)")


def test_load_cascade_accelerate() -> None:
    print("== каскад загрузки: device_map -> сброс кэшей -> путь без accelerate ==")
    engine = QwenASREngine.__new__(QwenASREngine)
    calls: list = []

    class FakeModel:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            return self

    class FakeAuto:
        @staticmethod
        def from_pretrained(path, dtype=None, torch_dtype=None, device_map=None):
            calls.append(("from_pretrained", device_map))
            if device_map is not None:
                # ровно та ошибка, что случилась у пользователя: accelerate
                # уже установлен, но transformers кэшировал его отсутствие
                raise ValueError(
                    "Using a `device_map`, `tp_plan`, `torch.device` context manager "
                    "or setting `torch.set_default_device(device)` requires `accelerate`. "
                    "You can install it with `pip install accelerate`"
                )
            return FakeModel()

    model = engine._load_qwen_model(FakeAuto, "fake/path", None, "cuda")
    check(calls.count(("from_pretrained", "cuda")) == 2,
          "две попытки с device_map (исходная и после сброса кэшей)")
    check(("from_pretrained", None) in calls, "откат: загрузка без device_map")
    check(("to", "cuda") in calls, "откат завершается .to(device)")
    check(model is not None, "модель в итоге загружена несмотря на кэш")

    # accelerate реально отсутствует -> сразу путь без device_map
    import importlib.util as iu

    original = iu.find_spec
    iu.find_spec = lambda name, *a, **k: None if name == "accelerate" else original(name, *a, **k)
    try:
        calls.clear()
        engine._load_qwen_model(FakeAuto, "fake/path", None, "cpu")
        check(all(dm is None for _, dm in calls if _ == "from_pretrained"),
              "без accelerate device_map даже не пробуется")
        check(("to", "cpu") in calls, "без accelerate модель переносится через .to()")
    finally:
        iu.find_spec = original


def test_check_and_set_device_map_recovery() -> None:
    print("== полная цепочка бага: device_map -> кэш -> сброс -> работает ==")
    try:
        import torch  # noqa: F401
        import transformers.utils.import_utils as iu
        from transformers.integrations.accelerate import check_and_set_device_map
    except ImportError as exc:
        skip(f"нет torch/transformers ({exc})")
        return
    from app.asr.torch_engines import reset_transformers_caches

    original = iu._is_package_available

    def fake_missing(name, *args, **kwargs):
        if name == "accelerate":
            if kwargs.get("return_version"):
                return False, "0.0.0"
            return False
        return original(name, *args, **kwargs)

    # 1) «GigaAM загрузился до установки accelerate» — кэшируем отсутствие
    iu.is_accelerate_available.cache_clear()
    iu._is_package_available = fake_missing
    try:
        iu.is_accelerate_available()
    finally:
        iu._is_package_available = original

    # 2) ровно та ошибка пользователя: device_map + устаревший кэш
    try:
        check_and_set_device_map("cuda")
        check(False, "с устаревшим кэшем device_map должен падать (как у пользователя)")
    except ValueError as exc:
        check("accelerate" in str(exc), f"воспроизведён исходный баг: {str(exc)[:60]}…")

    # 3) сброс кэшей (что делает программа после установки пакетов) — и тот же
    #    вызов проходит: accelerate «увиден» без перезапуска
    reset_transformers_caches()
    try:
        result = check_and_set_device_map("cuda")
        check(True, f"после сброса кэшей device_map работает: {result}")
    except ValueError as exc:
        check(False, f"после сброса не должно падать: {exc}")


def main() -> int:
    test_audio_format_accepted_by_transformers()
    test_qwen_inputs_contract()
    test_oom_bisection()
    test_is_available_pin()
    test_error_hints()
    test_transformers_cache_reset()
    test_load_cascade_accelerate()
    test_check_and_set_device_map_recovery()
    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("РЕГРЕССИОННЫЕ ТЕСТЫ QWEN3-ASR ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
