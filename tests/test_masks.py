"""마스크 생성 정확성 테스트."""
import torch

from config import PAD_IDX
from transformer.model.transformer import (
    Transformer,
    make_causal_mask,
    make_pad_mask,
)


def test_pad_mask_marks_padding_false():
    seq = torch.tensor([[5, 6, PAD_IDX, PAD_IDX]])
    mask = make_pad_mask(seq)  # (1,1,1,4)
    assert mask.shape == (1, 1, 1, 4)
    assert mask[0, 0, 0].tolist() == [True, True, False, False]


def test_causal_mask_is_lower_triangular():
    mask = make_causal_mask(4, torch.device("cpu"))[0, 0]
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(mask, expected)


def test_tgt_mask_combines_pad_and_causal():
    model = Transformer(10, 10, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_len=8)
    tgt = torch.tensor([[1, 5, 6, PAD_IDX]])
    mask = model.make_tgt_mask(tgt)  # (1,1,4,4)
    # 첫 행(위치0)은 자기 자신만 본다 + 패딩 열은 항상 False
    assert mask[0, 0, 0].tolist() == [True, False, False, False]
    # 마지막 열(패딩 토큰)은 모든 행에서 False
    assert mask[0, 0, :, 3].tolist() == [False, False, False, False]
