"""Multi30k(EN->DE) Transformer 학습 스크립트.

실행: python -m transformer.train --epochs 10
"""
from __future__ import annotations

import argparse
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
from .utils import NoamLR, load_checkpoint, resolve_device, save_checkpoint, set_seed


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
) -> float:
    """한 epoch 수행. optimizer가 주어지면 학습, 아니면 평가 모드.

    teacher forcing: 디코더 입력은 tgt[:, :-1], 정답은 tgt[:, 1:].
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_tokens = 0.0, 0

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

        # PAD를 제외한 토큰 단위 평균 loss 집계
        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


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
    best_val = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler,
            cfg.train.grad_clip, desc=f"train {epoch}",
        )
        val_loss = run_epoch(model, val_loader, criterion, device, desc=f"val {epoch}")
        msg = f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}"

        if args.eval_bleu:
            bleu = evaluate_bleu(model, val_loader, tgt_vocab, device, cfg.model.max_len)
            msg += f" val_bleu={bleu:.2f}"
        print(msg)

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


if __name__ == "__main__":
    main()
