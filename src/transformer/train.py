"""Multi30k(EN->DE) Transformer 학습 스크립트.

실행: python -m transformer.train --epochs 10
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

from config import BOS_IDX, EOS_IDX, PAD_IDX, Config
from .data.dataset import build_dataloaders
from .data.vocab import Vocab
from .model.transformer import Transformer
from .utils import (
    NoamLR,
    load_checkpoint,
    maybe_resume,
    resolve_device,
    save_checkpoint,
    save_training_state,
    set_seed,
)


def build_model(cfg: Config, src_vocab: Vocab, tgt_vocab: Vocab) -> Transformer:
    m = cfg.model
    return Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=m.d_model,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_len=m.max_len,
        pad_idx=PAD_IDX,
        tie_embeddings=m.tie_embeddings,
    )


def run_epoch(
    model: Transformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: NoamLR | None = None,
    grad_clip: float = 1.0,
    desc: str = "",
) -> tuple[float, float]:
    """한 epoch 수행. optimizer가 주어지면 학습, 아니면 평가 모드.

    teacher forcing: 디코더 입력은 tgt[:, :-1], 정답은 tgt[:, 1:].
    반환: (평균 loss, 토큰 단위 정확도) — 둘 다 PAD 토큰을 제외하고 집계.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_tokens, total_correct = 0.0, 0, 0

    for src, tgt in tqdm(loader, desc=desc, leave=False):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in)  # (B, T, vocab)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
            )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # PAD를 제외한 토큰 단위로 loss와 정확도(teacher forcing argmax 일치)를 집계
        non_pad = tgt_out != PAD_IDX
        n_tokens = non_pad.sum().item()
        preds = logits.argmax(dim=-1)  # (B, T)
        total_correct += ((preds == tgt_out) & non_pad).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    denom = max(total_tokens, 1)
    return total_loss / denom, total_correct / denom


@torch.no_grad()
def evaluate_bleu(
    model: Transformer,
    loader: DataLoader,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int = 128,
) -> float:
    """greedy 디코딩으로 corpus BLEU를 계산(sacrebleu)."""
    from .translate import greedy_decode  # 순환 import 방지를 위해 지연 로드
    import sacrebleu

    model.eval()
    hyps: List[str] = []
    refs: List[str] = []
    from .data.tokenizer import detokenize

    for src, tgt in loader:
        src = src.to(device)
        for i in range(src.size(0)):
            pred_ids = greedy_decode(model, src[i : i + 1], device, max_len)
            hyps.append(detokenize(tgt_vocab.decode(pred_ids)))
            refs.append(detokenize(tgt_vocab.decode(tgt[i].tolist())))
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-bleu", action="store_true", help="epoch마다 BLEU 측정(느림)")
    parser.add_argument("--fresh", action="store_true",
                        help="last.pt 체크포인트를 무시하고 처음(epoch 1)부터 학습")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.device is not None:
        cfg.train.device = args.device

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    print(f"device: {device}")

    train_loader, val_loader, _, src_vocab, tgt_vocab = build_dataloaders(cfg)
    print(f"vocab: src={len(src_vocab)} tgt={len(tgt_vocab)}")

    model = build_model(cfg, src_vocab, tgt_vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=cfg.train.label_smoothing
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0, betas=cfg.train.betas, eps=cfg.train.eps
    )
    scheduler = NoamLR(optimizer, cfg.model.d_model, cfg.train.warmup_steps)

    ckpt_path = Path(cfg.checkpoint_dir) / "best.pt"
    last_path = Path(cfg.checkpoint_dir) / "last.pt"
    history_path = Path(cfg.checkpoint_dir) / "history.json"

    # 자동 재개: last.pt가 있으면 model/optimizer/scheduler 상태를 복원하고 다음 epoch부터 이어간다.
    if args.fresh:
        start_epoch, best_val, history = 1, float("inf"), []
    else:
        start_epoch, best_val, history = maybe_resume(
            last_path, model=model, optimizer=optimizer,
            scheduler=scheduler, device=device,
        )
    if start_epoch > 1:
        print(f"resuming from epoch {start_epoch} (best_val={best_val:.4f})")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler,
            cfg.train.grad_clip, desc=f"train {epoch}",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, desc=f"val {epoch}"
        )
        msg = (
            f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        if args.eval_bleu:
            bleu = evaluate_bleu(model, val_loader, tgt_vocab, device, cfg.model.max_len)
            record["val_bleu"] = bleu
            msg += f" val_bleu={bleu:.2f}"
        print(msg)

        # epoch마다 누적 기록을 저장 — 학습이 중간에 끊겨도 곡선을 그릴 수 있다.
        history.append(record)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpt_path,
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "src_itos": src_vocab.itos,
                    "tgt_itos": tgt_vocab.itos,
                },
            )
            print(f"  saved best -> {ckpt_path}")

        # best 갱신까지 끝난 뒤 매 epoch 전체 학습 상태를 last.pt에 저장 — 자동 재개의 근거.
        save_training_state(
            last_path,
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_val=best_val, history=history,
            extra={"cfg": cfg, "src_itos": src_vocab.itos, "tgt_itos": tgt_vocab.itos},
        )


if __name__ == "__main__":
    main()
