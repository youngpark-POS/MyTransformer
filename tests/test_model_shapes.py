"""모델 forward / 디코딩 shape 테스트."""
import torch

from transformer.model.transformer import Transformer


def _tiny_model():
    return Transformer(
        src_vocab_size=30, tgt_vocab_size=40,
        d_model=32, n_heads=4, n_layers=2, d_ff=64, max_len=20,
    )


def test_forward_output_shape():
    model = _tiny_model()
    src = torch.randint(4, 30, (3, 7))
    tgt = torch.randint(4, 40, (3, 5))
    logits = model(src, tgt)
    assert logits.shape == (3, 5, 40)  # (batch, tgt_len, tgt_vocab)


def test_encode_decode_separately():
    model = _tiny_model()
    src = torch.randint(4, 30, (2, 6))
    memory = model.encode(src)
    assert memory.shape == (2, 6, 32)
    tgt = torch.randint(4, 40, (2, 4))
    src_mask = model.make_src_mask(src)
    logits = model.decode(tgt, memory, src_mask=src_mask)
    assert logits.shape == (2, 4, 40)


def test_d_model_must_divide_n_heads():
    import pytest
    with pytest.raises(AssertionError):
        Transformer(10, 10, d_model=30, n_heads=4)  # 30 % 4 != 0
