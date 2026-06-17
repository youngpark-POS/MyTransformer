"""Scaled dot-product / multi-head attention 단위 테스트."""
import torch

from transformer.model.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)


def test_attention_output_shape():
    b, h, q, k, d = 2, 4, 5, 6, 8
    query = torch.randn(b, h, q, d)
    key = torch.randn(b, h, k, d)
    value = torch.randn(b, h, k, d)
    out, attn = scaled_dot_product_attention(query, key, value)
    assert out.shape == (b, h, q, d)
    assert attn.shape == (b, h, q, k)


def test_attention_weights_sum_to_one():
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 4, 6, 8)
    value = torch.randn(2, 4, 6, 8)
    _, attn = scaled_dot_product_attention(query, key, value)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_masked_positions_get_zero_weight():
    query = torch.randn(1, 1, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)
    # 마지막 key 위치를 차단
    mask = torch.tensor([[[[1, 1, 0]]]], dtype=torch.bool)
    _, attn = scaled_dot_product_attention(query, key, value, mask=mask)
    assert torch.allclose(attn[..., 2], torch.zeros_like(attn[..., 2]))


def test_multihead_attention_preserves_shape():
    mha = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
    x = torch.randn(2, 7, 32)
    out = mha(x, x, x)
    assert out.shape == (2, 7, 32)
