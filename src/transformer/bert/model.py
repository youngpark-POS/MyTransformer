"""BERT(encoder-only) 모델 — 번역용 인코더 부품을 재사용해 조립한다.

설계 원칙: '백본(BertModel)'과 '태스크 헤드(BertForMaskedLM 등)'를 분리한다.
- BertModel: 임베딩 + EncoderLayer 스택 + 최종 norm + [CLS] pooler.
- BertForMaskedLM: BertModel + MLM 헤드.

이렇게 두면 풀 BERT(NSP)나 분류 fine-tuning은 '같은 BertModel + 다른 헤드'로
끝난다. 또한 BertEmbeddings는 처음부터 segment(token-type) 임베딩을 포함하되
MLM 단계에선 segment_ids=0으로만 쓰므로, 문장쌍 입력으로의 확장이 무변경이다.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from config import PAD_IDX
from ..model.layers import EncoderLayer
from ..model.positional import PositionalEncoding
from ..model.transformer import make_pad_mask


class BertEmbeddings(nn.Module):
    """토큰 + 위치 + segment 임베딩의 합.

    위치 인코딩은 번역용 sinusoidal PositionalEncoding을 재사용한다(드롭아웃 포함).
    segment 임베딩은 지금은 전부 0번 세그먼트로만 쓰이지만, 자리(슬롯)를 미리
    두어 풀 BERT의 문장쌍(segment 0/1) 입력으로 그대로 확장된다.
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float, n_segments: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.segment = nn.Embedding(n_segments, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, input_ids: Tensor, segment_ids: Optional[Tensor] = None) -> Tensor:
        # 토큰 임베딩에 sqrt(d_model) 스케일(원논문 3.4절) — 번역 인코더와 동일 관행
        x = self.token(input_ids) * math.sqrt(self.d_model)
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)  # MLM 단계: 단일 세그먼트
        x = x + self.segment(segment_ids)
        return self.pos(x)  # 위치 인코딩 가산 + dropout


class BertModel(nn.Module):
    """BERT 인코더 백본. 양방향 self-attention(causal 마스크 없음)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embeddings = BertEmbeddings(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)  # pre-norm 스택의 최종 정규화
        # [CLS] pooler: 풀 BERT의 NSP / 분류 fine-tuning에서 문장 표현으로 사용.
        self.pooler = nn.Linear(d_model, d_model)

    def make_mask(self, input_ids: Tensor) -> Tensor:
        """패딩만 차단하는 양방향 마스크 (B,1,1,L). 미래 차단(causal)은 쓰지 않는다."""
        return make_pad_mask(input_ids, self.pad_idx)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if attention_mask is None:
            attention_mask = self.make_mask(input_ids)
        x = self.embeddings(input_ids, segment_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        pooled = torch.tanh(self.pooler(x[:, 0]))  # [CLS] 위치 표현
        return x, pooled


class BertMLMHead(nn.Module):
    """MLM 출력 헤드 (BERT 원논문 구조): dense → GELU → LayerNorm → vocab projection.

    마지막 projection 가중치는 토큰 임베딩과 weight tying할 수 있다(파라미터 절감 + 정규화).
    """

    def __init__(self, d_model: int, vocab_size: int, token_embedding: Optional[nn.Embedding] = None) -> None:
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
        if token_embedding is not None:
            self.decoder.weight = token_embedding.weight  # (vocab_size, d_model) 공유

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.act(self.dense(x)))
        return self.decoder(x)  # (B, L, vocab_size)


class BertForMaskedLM(nn.Module):
    """BERT 백본 + MLM 헤드. MLM 사전학습용 최상위 모듈."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_idx: int = PAD_IDX,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.bert = BertModel(
            vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len, pad_idx
        )
        token_embed = self.bert.embeddings.token if tie_embeddings else None
        self.mlm_head = BertMLMHead(d_model, vocab_size, token_embed)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier 균등 초기화(번역 Transformer와 동일 관행). 다차원 가중치에만 적용."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
    ) -> Tensor:
        sequence_output, _ = self.bert(input_ids, attention_mask, segment_ids)
        return self.mlm_head(sequence_output)  # (B, L, vocab_size)
