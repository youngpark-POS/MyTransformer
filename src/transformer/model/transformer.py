"""인코더-디코더 Transformer 전체 조립 + 마스크 생성.

마스크 규약(scaled_dot_product_attention와 일치):
    True/1  = 유지(attend 허용)
    False/0 = 차단(-inf로 채워 softmax에서 0)
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from config import PAD_IDX
from .decoder import Decoder
from .encoder import Encoder


def make_pad_mask(seq: Tensor, pad_idx: int = PAD_IDX) -> Tensor:
    """패딩 위치를 차단하는 마스크.

    seq: (batch, seq_len) -> (batch, 1, 1, seq_len)
    key 차원(마지막)에 broadcast되어 패딩 토큰에 대한 attention을 막는다.
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device: torch.device) -> Tensor:
    """미래 토큰을 차단하는 하삼각(lower-triangular) 마스크.

    반환: (1, 1, size, size). [i, j]가 True면 위치 i가 j(<=i)를 볼 수 있음.
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_idx: int = PAD_IDX,
        tie_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(
            src_vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len,
            tie_embeddings=tie_embeddings,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier 균등 초기화(원논문 관행). 다차원 가중치에만 적용."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # --- 마스크 ---------------------------------------------------------
    def make_src_mask(self, src: Tensor) -> Tensor:
        return make_pad_mask(src, self.pad_idx)

    def make_tgt_mask(self, tgt: Tensor) -> Tensor:
        """타깃은 pad 마스크 ∧ causal 마스크를 결합한다."""
        pad = make_pad_mask(tgt, self.pad_idx)  # (batch,1,1,tgt_len)
        causal = make_causal_mask(tgt.size(1), tgt.device)  # (1,1,tgt_len,tgt_len)
        return pad & causal

    # --- forward --------------------------------------------------------
    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        return self.decoder(tgt, memory, tgt_mask, src_mask)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """src:(batch,src_len), tgt:(batch,tgt_len) -> logits:(batch,tgt_len,vocab)."""
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encoder(src, src_mask)
        return self.decoder(tgt, memory, tgt_mask, src_mask)
