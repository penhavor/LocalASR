# -*- coding: utf-8 -*-
"""Qwen3-ASR engine regression tests — no real weights needed.

Covers:
  * the audio passing format passes the transformers validator
    (make_list_of_audio);
  * the legacy format (tuple) is rejected — the test catches a regression;
  * _qwen_inputs calls apply_transcription_request with audio=<ndarray>;
  * the TypeError fallback cascade (signatures without prompt/language);
  * OOM bisection: a chunk is halved and the texts are joined;
  * is_available(): the transformers>=5.13 pin via metadata (no import);
  * error hints: "incomplete files" and "old transformers";
  * the "requires accelerate" regression: the transformers lru cache that
    remembered the absence of accelerate is reset, and the whole
    check_and_set_device_map(device_map) chain works WITHOUT a restart;
  * the weight-loading cascade: device_map -> cache reset -> the
    accelerate-free path (load + .to(device)) when accelerate is absent.

Only transformers is required (torch for the OOM branch, accelerate for the
full "installed while running" coverage; without them those steps skip).

Run:  cd <project root> && python tests/test_qwen_asr_engine.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("LOCALASR_QUIET", "1")

from app import i18n  # noqa: E402

i18n.init("ru")  # stable language for message assertions

import numpy as np  # noqa: E402

from app.asr.torch_engines import (  # noqa: E402
    QwenASREngine,
    missing_files_hint,
    transformers_version_hint,
)
from app.constants import SAMPLE_RATE  # noqa: E402

failures: list[str] = []

#: The "package installed WHILE the app runs" steps can only be verified for
#: real when accelerate is actually present (the app installs it together with
#: the engine). Without it those steps SKIP instead of failing the module.
HAS_ACCELERATE = importlib.util.find_spec("accelerate") is not None


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        failures.append(message)


def skip(message: str) -> None:
    print(f"  - SKIP: {message}")


class FakeProcessor:
    """Processor mirroring the Qwen3ASRProcessor contract (input validation)."""

    def __init__(self, mode: str = "full") -> None:
        self.mode = mode          # full | no_prompt | no_language
        self.calls: list[dict] = []

    def apply_transcription_request(self, audio=None, language=None, prompt=None, **kwargs):
        if self.mode == "no_prompt" and prompt is not None:
            raise TypeError("apply_transcription_request() got an unexpected keyword argument 'prompt'")
        if self.mode == "no_language" and language is not None:
            raise TypeError("apply_transcription_request() got an unexpected keyword argument 'language'")
        # mirror the real transformers 5.x validation
        from transformers.audio_utils import make_list_of_audio_chat_template

        items = list(make_list_of_audio_chat_template(audio))
        self.calls.append({"audio": items[0], "language": language, "prompt": prompt})
        return {"input_ids": items, "ok": True}

    def decode(self, ids, return_format="raw", **kwargs):
        return [{"transcription": "распознанный текст", "language": "Russian"}]


def test_audio_format_accepted_by_transformers() -> None:
    print("== audio format passes the transformers validator ==")
    try:
        from transformers.audio_utils import make_list_of_audio
    except ImportError:
        skip("no transformers")
        return
    chunk = np.zeros(SAMPLE_RATE, dtype=np.float32)
    try:
        make_list_of_audio(chunk)
        check(True, "ndarray 16 kHz is accepted by make_list_of_audio")
    except ValueError as exc:
        check(False, f"ndarray must be accepted: {exc}")
    try:
        make_list_of_audio((chunk, SAMPLE_RATE))
        check(False, "tuple (array, sr) must NOT be accepted (legacy regression)")
    except ValueError:
        check(True, "tuple (array, sr) is rejected — regression guarded")


def test_qwen_inputs_contract() -> None:
    print("== _qwen_inputs: audio=array, fallback cascade ==")
    engine = QwenASREngine.__new__(QwenASREngine)   # no __init__/torch
    chunk = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

    proc = FakeProcessor("full")
    engine._processor = proc
    engine._qwen_inputs(chunk, "ru", "термины")
    call = proc.calls[-1]
    check(isinstance(call["audio"], np.ndarray), "the processor gets an ndarray, not a tuple")
    check(call["language"] == "ru" and call["prompt"] == "термины", "language/prompt passed")

    proc2 = FakeProcessor("no_prompt")
    engine._processor = proc2
    engine._qwen_inputs(chunk, "ru", "термины")
    check(proc2.calls and proc2.calls[-1]["prompt"] is None, "fallback: call without prompt")

    proc3 = FakeProcessor("no_language")
    engine._processor = proc3
    engine._qwen_inputs(chunk, "ru", None)
    check(proc3.calls and proc3.calls[-1]["language"] is None, "fallback: call without language")

    engine._processor = object()
    try:
        engine._qwen_inputs(chunk, None, None)
        check(False, "without apply_transcription_request there must be a clear error")
    except Exception as exc:
        check("transformers" in str(exc), f"old transformers -> update hint: {str(exc)[:60]}…")


def test_oom_bisection() -> None:
    print("== OOM bisection of a long chunk ==")
    try:
        import torch  # noqa: F401
    except ImportError:
        skip("no torch")
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
        # a "long" 45 s chunk -> must be split down to <=20-second pieces
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

        # patch model.generate through an engine attribute
        engine._ensure_model = lambda: _FakeGen()
        text, lang = engine._infer_qwen(engine._ensure_model(), chunk, "ru", None, 0.0)
        check(calls["n"] >= 3, f"long chunk split (calls: {calls['n']})")
        check("кусок текста" in text, f"half texts joined: {text[:60]!r}")
        check(lang == "Russian", "language detected")
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
    # the class checks metadata only: torch/accelerate may be absent —
    # patch the base-module check
    parent_available = QwenASREngine.__mro__[1].is_available
    QwenASREngine.__mro__[1].is_available = classmethod(lambda cls: True)
    try:
        md.version = fake_version
        fake_version.value = "5.20.2"
        check(QwenASREngine.is_available() is True, "transformers 5.20.2 — available")
        fake_version.value = "5.13.0"
        check(QwenASREngine.is_available() is True, "transformers 5.13.0 — available (boundary)")
        fake_version.value = "5.12.1"
        check(QwenASREngine.is_available() is False, "transformers 5.12.1 — NOT available (no qwen3_asr)")
        fake_version.value = "4.57.6"
        check(QwenASREngine.is_available() is False, "transformers 4.57.6 — NOT available")
    finally:
        md.version = original
        QwenASREngine.__mro__[1].is_available = parent_available

    from app.asr.base import engine_packages

    pkgs = engine_packages("qwen-asr")
    check("transformers>=5.13" in pkgs, f"pin in engine_packages: {pkgs}")


def test_error_hints() -> None:
    print("== hints inside error messages ==")
    exc = OSError(
        "Can't load feature extractor for 'models/qwen3-asr-1.7b'. make sure "
        "'models/qwen3-asr-1.7b' is the correct path to a directory containing "
        "a preprocessor_config.json file"
    )
    hint = missing_files_hint(exc)
    check("не полностью" in hint and "Удалить" in hint, "preprocessor_config error -> reinstall hint")

    exc2 = ValueError("Unrecognized configuration class <class 'x'> for this kind of AutoModel")
    hint2 = transformers_version_hint(exc2, "Qwen3-ASR", "5.13")
    check("transformers>=5.13" in hint2, "unknown model -> transformers update hint")

    check(missing_files_hint(ValueError("обычная ошибка")) == "", "plain error without false hints")


def test_transformers_cache_reset() -> None:
    print("== transformers lru cache reset (the 'requires accelerate' regression) ==")
    try:
        import transformers.utils.import_utils as iu
    except ImportError:
        skip("no transformers")
        return
    from app.asr.torch_engines import reset_transformers_caches

    fn = iu.is_accelerate_available
    original = iu._is_package_available

    def fake_missing(name, *args, **kwargs):
        # simulate "accelerate not installed yet" — as in the moment when
        # GigaAM was loaded before the Qwen packages were installed
        if name == "accelerate":
            if kwargs.get("return_version"):
                return False, "0.0.0"
            return False
        return original(name, *args, **kwargs)

    # 1) cache the stale "accelerate absent"
    fn.cache_clear()
    iu._is_package_available = fake_missing
    try:
        stale = fn()
    finally:
        iu._is_package_available = original
    check(stale is False, "before the reset transformers cached 'no accelerate' (False)")

    # 2) the package is "installed": without a reset the same False returns;
    #    after reset_transformers_caches the check runs again
    still_stale = fn()
    check(still_stale is False, "without a reset the cache keeps returning the stale False")
    cleared = reset_transformers_caches()
    check(cleared >= 1, f"reset_transformers_caches cleared caches ({cleared})")
    if HAS_ACCELERATE:
        fresh = fn()
        check(fresh is True,
              "after the reset transformers 'sees' the freshly installed accelerate (True)")
    else:
        skip("accelerate is not installed — the 'freshly installed' transition is not verified")


def test_load_cascade_accelerate() -> None:
    print("== load cascade: device_map -> cache reset -> accelerate-free path ==")
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
                raise ValueError(
                    "Using a `device_map`, `tp_plan`, `torch.device` context manager "
                    "or setting `torch.set_default_device(device)` requires `accelerate`. "
                    "You can install it with `pip install accelerate`"
                )
            return FakeModel()

    model = engine._load_qwen_model(FakeAuto, "fake/path", None, "cuda")
    if HAS_ACCELERATE:
        check(calls.count(("from_pretrained", "cuda")) == 2,
              "two attempts with device_map (initial and after the cache reset)")
    else:
        check(calls.count(("from_pretrained", "cuda")) == 0,
              "without accelerate device_map is not tried at all")
        skip("accelerate is not installed — the device_map retry cascade is not verified")
    check(("from_pretrained", None) in calls, "fallback: load without device_map")
    check(("to", "cuda") in calls, "the fallback ends with .to(device)")
    check(model is not None, "the model eventually loads despite the cache")

    # accelerate truly absent -> straight to the device_map-free path
    import importlib.util as iu

    original = iu.find_spec
    iu.find_spec = lambda name, *a, **k: None if name == "accelerate" else original(name, *a, **k)
    try:
        calls.clear()
        engine._load_qwen_model(FakeAuto, "fake/path", None, "cpu")
        check(all(dm is None for _, dm in calls if _ == "from_pretrained"),
              "without accelerate device_map is not even tried")
        check(("to", "cpu") in calls, "without accelerate the model moves via .to()")
    finally:
        iu.find_spec = original


def test_check_and_set_device_map_recovery() -> None:
    print("== the full chain: device_map -> stale cache -> reset -> works ==")
    try:
        import torch  # noqa: F401
        import transformers.utils.import_utils as iu
        from transformers.integrations.accelerate import check_and_set_device_map
    except ImportError as exc:
        skip(f"no torch/transformers ({exc})")
        return
    from app.asr.torch_engines import reset_transformers_caches

    original = iu._is_package_available

    def fake_missing(name, *args, **kwargs):
        if name == "accelerate":
            if kwargs.get("return_version"):
                return False, "0.0.0"
            return False
        return original(name, *args, **kwargs)

    # 1) "GigaAM loaded before accelerate was installed" — cache the absence
    iu.is_accelerate_available.cache_clear()
    iu._is_package_available = fake_missing
    try:
        iu.is_accelerate_available()
    finally:
        iu._is_package_available = original

    # 2) device_map + stale cache must fail
    try:
        check_and_set_device_map("cuda")
        check(False, "with a stale cache device_map must fail")
    except ValueError as exc:
        check("accelerate" in str(exc), f"the stale-cache failure reproduced: {str(exc)[:60]}…")

    # 3) reset the caches (what the app does after installing packages) — and
    #    the same call passes: accelerate is "seen" without a restart
    reset_transformers_caches()
    if not HAS_ACCELERATE:
        skip("accelerate is not installed — the post-reset recovery is not verified")
        return
    try:
        result = check_and_set_device_map("cuda")
        check(True, f"after the cache reset device_map works: {result}")
    except ValueError as exc:
        check(False, f"must not fail after the reset: {exc}")


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
        print(f"FAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("QWEN3-ASR REGRESSION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
