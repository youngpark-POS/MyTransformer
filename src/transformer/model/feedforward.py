"""Position-wise Feed-Forward Network (원논문 3.3절).

FFN(x) = max(0, x W1 + b1) W2 + b2
각 위치에 독립적으로 동일하게 적용되는 2-layer MLP.
"""
from __future__ import annotations

from torch import Tensor, nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.linear1(x).relu()))
