"""토큰 임베딩 + positional encoding + N개의 DecoderLayer + 출력 projection."""
from __future__ import annotations

import math
from typing import Optional

from torch import Tensor, nn

from .layers import DecoderLayer
from .positional import PositionalEncoding


class Decoder(nn.Module):
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
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)  # 어휘 logits 생성

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        src_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.embed(tgt) * math.sqrt(self.d_model)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        x = self.norm(x)
        return self.out_proj(x)  # (batch, tgt_len, vocab_size)
