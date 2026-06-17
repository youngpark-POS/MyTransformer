"""Scaled dot-product attention과 multi-head attention을 직접 구현한다.

nn.MultiheadAttention을 쓰지 않고 Q/K/V projection, head 분할/병합,
스케일링, 마스킹, softmax를 모두 손으로 작성한다.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[Tensor, Tensor]:
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.

    Shapes:
        query/key/value: (batch, n_heads, seq_len, d_k)
        mask: (batch, 1, q_len, k_len) 또는 broadcast 가능한 형태.
              True=유지, False=마스킹(-inf로 채워 softmax에서 0이 됨)
    Returns:
        output: (batch, n_heads, q_len, d_k)
        attn:   (batch, n_heads, q_len, k_len)  # 시각화/검증용 가중치
    """
    d_k = query.size(-1)
    # QK^T: (batch, n_heads, q_len, k_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    output = torch.matmul(attn, value)
    return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention.

    d_model을 n_heads개로 쪼개 각 head에서 독립적으로 attention을 수행한 뒤
    다시 concat하여 출력 projection을 통과시킨다.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V, 출력에 각각 별도의 선형변환
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn: Optional[Tensor] = None  # 디버깅/검증용으로 보관

    def _split_heads(self, x: Tensor) -> Tensor:
        """(batch, seq, d_model) -> (batch, n_heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(batch, n_heads, seq, d_k) -> (batch, seq, d_model)"""
        batch, _, seq, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # 선형 projection 후 head 분할
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        out, attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        self.last_attn = attn.detach()

        # head 병합 후 출력 projection
        out = self._merge_heads(out)
        return self.w_o(out)
