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
        tie_embeddings: bool = False,
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
        if tie_embeddings:
            # 입력 임베딩과 출력 projection 가중치를 공유(weight tying).
            # 두 행렬 모두 (vocab_size, d_model) 형상이라 그대로 묶을 수 있다.
            # 파라미터(약 vocab*d_model개)를 줄이고 작은 데이터셋에서 정규화 효과를 준다.
            self.out_proj.weight = self.embed.weight

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
