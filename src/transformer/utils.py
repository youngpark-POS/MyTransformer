"""학습/추론 공용 유틸: 시드 고정, 디바이스 선택, Noam LR 스케줄, 체크포인트."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str) -> torch.device:
    """요청한 디바이스가 불가하면 cpu로 안전하게 대체."""
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


class NoamLR:
    """원논문 5.3절 학습률 스케줄.

    lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    warmup 동안 선형 증가 후 step^-0.5에 비례해 감소한다.
    옵티마이저의 lr을 매 step 직접 갱신하는 경량 래퍼.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        lr = self.d_model ** -0.5 * min(
            self._step ** -0.5, self._step * self.warmup_steps ** -1.5
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
