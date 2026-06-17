"""토큰 임베딩 + positional encoding + N개의 EncoderLayer 스택."""
from __future__ import annotations

import math
from typing import Optional

from torch import Tensor, nn

from .layers import EncoderLayer
from .positional import PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)  # pre-norm 스택의 최종 정규화

    def forward(self, src: Tensor, src_mask: Optional[Tensor]) -> Tensor:
        # 임베딩에 sqrt(d_model) 스케일 적용(원논문 3.4절)
        x = self.embed(src) * math.sqrt(self.d_model)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
