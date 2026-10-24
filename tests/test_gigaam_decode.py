# -*- coding: utf-8 -*-
"""Контрактный тест быстрого пути GigaAM без реальных весов.

Прогоняет GigaAMEngine._decode_group (CTC и RNNT) на фейковых объектах,
которые повторяют КОНТРАКТ modeling_gigaam.py ОБЕИХ поколений:
  * старое:  decoding.decode(...) -> List[str]
  * новое:   decoding.decode(...) -> List[Tuple[str, token_ids, token_frames]]
Если быстрый путь начнёт стринговать кортежи (баг 2026-10-24: в файл падало
«('текст', [45, 33, …])»), тест упадёт.

Запуск:  cd <корень проекта> && python tests/test_gigaam_decode.py
Нужен только CPU-torch:  pip install torch --index-url https://download.pytorch.org/whl/cpu
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from app.asr.torch_engines import GigaAMEngine, decode_item_text  # noqa: E402

ENC_DIM = 4
JOINT_DIM = 8
NUM_CLASSES = 5          # 4 символа + blank
BLANK_ID = NUM_CLASSES - 1
FRAMES = 12


# --------------------------------------------------------------------------- #
#  Феки, повторяющие структуру modeling_gigaam.py
# --------------------------------------------------------------------------- #


class FakeTokenizer:
    """Словарь из 4 «символов»; decode склеивает без разделителей."""

    VOCAB = ["а", "б", "в", "г"]

    def __len__(self) -> int:
        return len(self.VOCAB)

    def decode(self, ids) -> str:
        return "".join(self.VOCAB[int(i)] for i in ids)


class FakeCTCDecoding:
    """CTC-декод в двух контрактах: строки (старый) и кортежи (новый)."""

    def __init__(self, contract: str) -> None:
        self.contract = contract
        self.tokenizer = FakeTokenizer()
        self.blank_id = BLANK_ID
        self._n = 0   # сквозной счётчик кусков (в каждом encoded batch=1)

    @torch.inference_mode()
    def decode(self, head, encoded, lengths):
        batch = encoded.shape[0]
        items = [self._n + i for i in range(batch)]
        self._n += batch
        if self.contract == "tuple":
            return [(f"фраза кусок {i}", [1, 2, 3], [0, 4, 8]) for i in items]
        return [f"фраза кусок {i}" for i in items]


class FakeRNNTDecoding:
    tokenizer = FakeTokenizer()
    blank_id = BLANK_ID
    max_symbols = 4


class TinyJoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(ENC_DIM, JOINT_DIM)
        self.pred = nn.Linear(ENC_DIM, JOINT_DIM)
        self.joint_net = nn.Sequential(nn.ReLU(), nn.Linear(JOINT_DIM, NUM_CLASSES))

    def joint(self, enc_out, dec_out):
        return self.joint_net(self.enc(enc_out) + self.pred(dec_out)).log_softmax(-1)


class TinyDecoder(nn.Module):
    """RNNTDecoder: embed + lstm + predict(x, state, batch_size=1)."""

    def __init__(self) -> None:
        super().__init__()
        self.pred_hidden = ENC_DIM
        self.embed = nn.Embedding(NUM_CLASSES, ENC_DIM, padding_idx=BLANK_ID)
        self.lstm = nn.LSTM(ENC_DIM, ENC_DIM, 1)

    def predict(self, x, state, batch_size: int = 1):
        emb = self.embed(x) if x is not None else torch.zeros(batch_size, 1, self.pred_hidden)
        out, hid = self.lstm(emb.transpose(0, 1), state)
        return out.transpose(0, 1), hid


class FakeRNNTHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = TinyDecoder()
        self.joint = TinyJoint()


class FakeCTCHead:
    """Для CTC head только передаётся в decode — содержимое не важно."""


def make_inner(kind: str, contract: str = "tuple"):
    class Inner:
        pass

    inner = Inner()
    if kind == "ctc":
        inner.decoding = FakeCTCDecoding(contract)
        inner.head = FakeCTCHead()
    else:
        inner.decoding = FakeRNNTDecoding()
        inner.head = FakeRNNTHead()
    return inner


def encoded_items(batch: int = 2):
    """Энкодер GigaAM отдаёт (B, D, T) — channels-first, как в modeling."""
    items = []
    for _ in range(batch):
        encoded = torch.randn(1, ENC_DIM, FRAMES)
        lengths = torch.tensor([FRAMES])
        items.append((encoded, lengths))
    return items


# --------------------------------------------------------------------------- #
#  Тесты
# --------------------------------------------------------------------------- #


def test_decode_item_text_contract() -> None:
    assert decode_item_text(("текст", [1, 2], [0, 1])) == "текст"
    assert decode_item_text(("текст", [1, 2])) == "текст"
    assert decode_item_text("текст") == "текст"
    assert decode_item_text(None) == ""
    assert decode_item_text([]) == ""


def test_ctc_new_contract_tuples() -> None:
    """Новое поколение: кортежи -> в текст идёт ТОЛЬКО строковая часть."""
    engine = GigaAMEngine()
    texts = engine._decode_group(make_inner("ctc", "tuple"), "ctc", encoded_items())
    assert texts == ["фраза кусок 0", "фраза кусок 1"], texts
    for text in texts:
        assert "(" not in text and "[" not in text, f"repr протёк в текст: {text!r}"


def test_ctc_old_contract_strings() -> None:
    """Старое поколение: голые строки по-прежнему работают."""
    engine = GigaAMEngine()
    texts = engine._decode_group(make_inner("ctc", "str"), "ctc", encoded_items())
    assert texts == ["фраза кусок 0", "фраза кусок 1"], texts


def test_rnnt_fast_path_runs() -> None:
    """RNNT: быстрый CPU-декод на фейках новой структуры даёт строки символов."""
    torch.manual_seed(0)
    engine = GigaAMEngine()
    texts = engine._decode_group(make_inner("rnnt"), "rnnt", encoded_items(batch=2))
    assert len(texts) == 2
    for text in texts:
        assert isinstance(text, str)
        assert "(" not in text and "[" not in text, f"repr протёк в текст: {text!r}"
        assert all(ch in FakeTokenizer.VOCAB for ch in text), text


def test_emit_segments_clean() -> None:
    """Сегменты, уходящие в постобработку/файл, не содержат repr-мусора."""
    from app.core.postprocess import Segment

    engine = GigaAMEngine()
    segments = []
    for (start, end), text in zip(((0, 16000), (16000, 32000)),
                                  engine._decode_group(make_inner("ctc", "tuple"), "ctc", encoded_items())):
        engine._emit(segments, start, end, text, None)
    assert len(segments) == 2
    for segment in segments:
        assert isinstance(segment, Segment)
        assert segment.text.startswith("фраза кусок"), segment.text


if __name__ == "__main__":
    test_decode_item_text_contract()
    print("  ✓ decode_item_text: кортежи и строки")
    test_ctc_new_contract_tuples()
    print("  ✓ CTC (новый контракт, кортежи) -> чистый текст")
    test_ctc_old_contract_strings()
    print("  ✓ CTC (старый контракт, строки) -> чистый текст")
    test_rnnt_fast_path_runs()
    print("  ✓ RNNT быстрый путь -> строки без repr-мусора")
    test_emit_segments_clean()
    print("  ✓ сегменты в постобработку чистые")
    print("\nКОНТРАКТНЫЙ ТЕСТ GIGAM ПРОЙДЕН")
