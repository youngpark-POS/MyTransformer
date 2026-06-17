"""Sinusoidal positional encoding (원논문 3.5절).

학습 파라미터 없이 위치마다 고정된 사인/코사인 패턴을 더해 순서 정보를 주입한다.
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # 10000^(2i/d_model)을 로그공간에서 안정적으로 계산
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) — 배치 broadcast용
        # 파라미터가 아닌 buffer로 등록(저장은 되지만 학습되지 않음)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
