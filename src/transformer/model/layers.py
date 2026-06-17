"""인코더/디코더를 구성하는 단일 레이어.

각 서브레이어는 'residual 연결 + LayerNorm'으로 감싼다. 여기서는 원논문의
post-norm 변형 대신, 학습 안정성이 좋은 pre-norm(LayerNorm을 서브레이어
입력에 먼저 적용) 방식을 사용한다: x + Sublayer(LayerNorm(x)).
"""
from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """self-attention + feed-forward."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_mask: Optional[Tensor]) -> Tensor:
        # pre-norm self-attention
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, src_mask))
        # pre-norm feed-forward
        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x


class DecoderLayer(nn.Module):
    """masked self-attention + cross-attention + feed-forward."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        src_mask: Optional[Tensor],
    ) -> Tensor:
        # 1) causal self-attention (미래 토큰 차단)
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, tgt_mask))
        # 2) cross-attention: query=디코더, key/value=인코더 출력(memory)
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, src_mask))
        # 3) feed-forward
        h = self.norm3(x)
        x = x + self.dropout(self.ff(h))
        return x
