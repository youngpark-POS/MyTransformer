"""wikitext MLM 사전학습 스크립트.

실행: python -m transformer.bert.train --epochs 10
      python -m transformer.bert.train --epochs 1 --device cpu   # 빠른 점검

번역 train.py의 구조(run_epoch 분기, NoamLR+Adam(lr=0), best 체크포인트 저장)를
그대로 따른다. 차이는 손실 대상이 teacher-forcing 시프트가 아니라 마스킹된 위치라는 점.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BertConfig, PAD_IDX
from ..data.vocab import Vocab
from ..utils import NoamLR, resolve_device, save_checkpoint, set_seed
from .data import build_mlm_dataloaders
from .model import BertForMaskedLM


def build_model(cfg: BertConfig, vocab: Vocab) -> BertForMaskedLM:
    m = cfg.model
    return BertForMaskedLM(
        vocab_size=len(vocab),
        d_model=m.d_model,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_len=m.max_len,
        pad_idx=PAD_IDX,
        tie_embeddings=m.tie_embeddings,
    )


def run_mlm_epoch(
    model: BertForMaskedLM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: NoamLR | None = None,
    grad_clip: float = 1.0,
    desc: str = "",
) -> tuple[float, float]:
    """한 epoch 수행. optimizer가 주어지면 학습, 아니면 평가 모드.

    배치는 (input_ids, labels) — labels는 마스킹된 위치만 원본 토큰, 나머지는 PAD_IDX.
    반환: (평균 loss, 마스킹 토큰 단위 정확도) — 둘 다 예측 대상(labels!=PAD)만 집계.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_tokens, total_correct = 0.0, 0, 0

    for input_ids, labels in tqdm(loader, desc=desc, leave=False):
        input_ids, labels = input_ids.to(device), labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(input_ids)  # (B, L, vocab)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # 예측 대상(마스킹된) 위치만으로 loss/정확도 집계
        predicted = labels != PAD_IDX
        n_tokens = predicted.sum().item()
        preds = logits.argmax(dim=-1)
        total_correct += ((preds == labels) & predicted).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    denom = max(total_tokens, 1)
    return total_loss / denom, total_correct / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = BertConfig()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.device is not None:
        cfg.train.device = args.device

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    print(f"device: {device}")

    train_loader, val_loader, vocab = build_mlm_dataloaders(cfg)
    print(f"vocab: {len(vocab)}  train_batches: {len(train_loader)}")

    model = build_model(cfg, vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    # 마스킹된 위치만 학습(labels의 나머지는 PAD_IDX = ignore_index)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # AdamW: 2D 가중치에만 weight decay 적용, LayerNorm/bias(1D)는 제외(BERT 관행).
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() == 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=0, betas=cfg.train.betas, eps=cfg.train.eps,
    )
    scheduler = NoamLR(optimizer, cfg.model.d_model, cfg.train.warmup_steps)

    ckpt_path = Path(cfg.checkpoint_dir) / "bert_best.pt"
    history_path = Path(cfg.checkpoint_dir) / "bert_history.json"
    best_val = float("inf")
    history: List[dict] = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, train_acc = run_mlm_epoch(
            model, train_loader, criterion, device, optimizer, scheduler,
            cfg.train.grad_clip, desc=f"train {epoch}",
        )
        val_loss, val_acc = run_mlm_epoch(
            model, val_loader, criterion, device, desc=f"val {epoch}"
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpt_path,
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "itos": vocab.itos,
                },
            )
            print(f"  saved best -> {ckpt_path}")


if __name__ == "__main__":
    main()
