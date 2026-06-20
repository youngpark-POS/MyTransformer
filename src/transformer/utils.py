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

    def state_dict(self) -> Dict[str, Any]:
        """재개를 위해 step 카운터만 저장(d_model/warmup은 cfg에서 재구성)."""
        return {"_step": self._step}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._step = state["_step"]


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def save_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamLR,
    epoch: int,
    best_val: float,
    history: list,
    extra: Dict[str, Any],
) -> None:
    """매 epoch 호출되는 전체 학습 상태 저장(last.pt 덮어쓰기).

    재개에 필요한 model/optimizer/scheduler 상태와 진행 정보(epoch/best_val/history)를
    모두 담는다. `extra`에는 경로별 메타(cfg, vocab itos 등)를 넣어 추론 복원에도 쓸 수 있다.
    """
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "history": history,
        **extra,
    }
    save_checkpoint(path, payload)


def maybe_resume(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamLR,
    device: str | torch.device,
) -> tuple[int, float, list]:
    """`path`에 저장된 학습 상태가 있으면 복원하고 (start_epoch, best_val, history) 반환.

    없으면 (1, inf, [])을 돌려 처음부터 시작한다. model/optimizer는 반드시 `.to(device)`와
    `lr=0`으로 생성된 뒤에 호출해야 state_dict 모양이 맞는다.
    """
    if not path.exists():
        return 1, float("inf"), []
    ckpt = load_checkpoint(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    return start_epoch, ckpt["best_val"], ckpt["history"]
