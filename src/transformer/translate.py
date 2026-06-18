"""학습된 모델로 자가회귀 디코딩(greedy / beam search)을 수행한다.

실행: python -m transformer.translate --text "a man is riding a bike"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch import Tensor

from config import BOS_IDX, EOS_IDX, Config
from .data.tokenizer import detokenize, tokenize
from .data.vocab import Vocab
from .model.transformer import Transformer
from .utils import load_checkpoint, resolve_device


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,
    device: torch.device,
    max_len: int = 128,
) -> List[int]:
    """매 step 가장 확률 높은 토큰 1개를 선택. src: (1, src_len)."""
    model.eval()
    src = src.to(device)
    src_mask = model.make_src_mask(src)
    memory = model.encoder(src, src_mask)

    ys = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys)
        out = model.decoder(ys, memory, tgt_mask, src_mask)  # (1, cur_len, vocab)
        next_token = out[:, -1].argmax(-1).item()  # 마지막 위치의 예측
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == EOS_IDX:
            break
    return ys.squeeze(0).tolist()


@torch.no_grad()
def beam_search_decode(
    model: Transformer,
    src: Tensor,
    device: torch.device,
    beam_size: int = 5,
    max_len: int = 128,
    length_penalty: float = 0.7,
) -> List[int]:
    """길이 정규화를 적용한 beam search. src: (1, src_len)."""
    model.eval()
    src = src.to(device)
    src_mask = model.make_src_mask(src)
    memory = model.encoder(src, src_mask)

    # 각 beam: (누적 로그확률, 토큰 리스트, 종료여부)
    beams = [(0.0, [BOS_IDX], False)]
    for _ in range(max_len - 1):
        if all(finished for _, _, finished in beams):
            break
        candidates = []
        for score, seq, finished in beams:
            if finished:
                candidates.append((score, seq, True))
                continue
            ys = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_mask = model.make_tgt_mask(ys)
            out = model.decoder(ys, memory, tgt_mask, src_mask)
            log_probs = torch.log_softmax(out[:, -1], dim=-1).squeeze(0)
            topk = torch.topk(log_probs, beam_size)
            for log_p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                new_seq = seq + [idx]
                candidates.append((score + log_p, new_seq, idx == EOS_IDX))
        # 길이 정규화 후 상위 beam_size개만 유지
        candidates.sort(
            key=lambda c: c[0] / (len(c[1]) ** length_penalty), reverse=True
        )
        beams = candidates[:beam_size]

    best_seq = max(beams, key=lambda c: c[0] / (len(c[1]) ** length_penalty))[1]
    return best_seq


def load_for_inference(ckpt_path: Path, device: torch.device):
    """체크포인트에서 모델 + vocab 복원."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg: Config = ckpt["cfg"]
    src_vocab = Vocab(ckpt["src_itos"])
    tgt_vocab = Vocab(ckpt["tgt_itos"])
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_len=cfg.model.max_len,
        tie_embeddings=getattr(cfg.model, "tie_embeddings", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, src_vocab, tgt_vocab, cfg


def translate(
    text: str,
    model: Transformer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    beam_size: int = 1,
    max_len: int = 128,
) -> str:
    src_ids = src_vocab.encode(tokenize(text))
    src = torch.tensor([src_ids], dtype=torch.long)
    if beam_size > 1:
        out_ids = beam_search_decode(model, src, device, beam_size, max_len)
    else:
        out_ids = greedy_decode(model, src, device, max_len)
    return detokenize(tgt_vocab.decode(out_ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--beam", type=int, default=1, help="1=greedy, >1=beam search")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Config().checkpoint_dir / "best.pt"
    model, src_vocab, tgt_vocab, cfg = load_for_inference(ckpt_path, device)

    result = translate(
        args.text, model, src_vocab, tgt_vocab, device, args.beam, cfg.model.max_len
    )
    print(f"SRC: {args.text}")
    print(f"OUT: {result}")


if __name__ == "__main__":
    main()
