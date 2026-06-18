"""통합 테스트: 합성 MLM 태스크를 과적합시켜 BERT 파이프라인이 실제로
학습되는지(loss 감소 + 마스킹 토큰 복원) 검증한다.

번역의 test_overfit.py와 동일한 철학 — 외부 다운로드 없이 수 초 내 실행된다.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import CLS_IDX, MASK_IDX, PAD_IDX, SEP_IDX, SPECIAL_TOKENS, MASK_TOKEN
from transformer.bert.data import MLMCollator
from transformer.bert.model import BertForMaskedLM
from transformer.bert.train import run_mlm_epoch
from transformer.data.vocab import Vocab
from transformer.utils import NoamLR, set_seed

VOCAB_SIZE = 24


def _make_vocab() -> Vocab:
    """[<pad>,<bos>,<eos>,<unk>, <mask>, w0..wN] 형태의 작은 합성 vocab."""
    itos = list(SPECIAL_TOKENS) + [MASK_TOKEN] + [f"w{i}" for i in range(VOCAB_SIZE - 5)]
    return Vocab(itos)  # __init__의 0~3 특수토큰 assert를 통과해야 함


def _make_mlm_data(n: int):
    """[CLS] body... [SEP] 형태의 합성 시퀀스.

    body는 '연속 증가' 규칙(token[i+1] = token[i]+1, 실제 단어 범위 내 순환)을 따른다.
    이렇게 학습 가능한 국소 구조를 주면 MLM이 마스킹된 토큰을 이웃으로부터 예측할 수
    있어 loss가 실제로 내려간다(순수 랜덤이면 예측 신호가 없어 사전분포에서 정체됨).
    """
    real = VOCAB_SIZE - (MASK_IDX + 1)  # 실제 단어 개수
    data = []
    for _ in range(n):
        length = torch.randint(4, 8, (1,)).item()
        start = torch.randint(0, real, (1,)).item()
        body = [MASK_IDX + 1 + (start + i) % real for i in range(length)]
        seq = torch.tensor([CLS_IDX] + body + [SEP_IDX], dtype=torch.long)
        data.append(seq)
    return data


def test_overfit_mlm_task():
    set_seed(0)
    device = torch.device("cpu")
    data = _make_mlm_data(96)
    collator = MLMCollator(VOCAB_SIZE, mask_prob=0.15)
    loader = DataLoader(data, batch_size=16, shuffle=True, collate_fn=collator)

    model = BertForMaskedLM(
        VOCAB_SIZE, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_len=20
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=64, warmup_steps=100)

    first_loss, _ = run_mlm_epoch(model, loader, criterion, device, optimizer, scheduler)
    for _ in range(59):
        last_loss, _ = run_mlm_epoch(model, loader, criterion, device, optimizer, scheduler)

    # 1) loss가 유의하게 감소했는가
    assert last_loss < first_loss * 0.5, f"loss did not drop: {first_loss}->{last_loss}"

    # 2) 학습한 시퀀스의 한 위치를 마스킹하면 원본 토큰을 복원하는가
    model.eval()
    seq = data[0].clone()
    pos = 2  # CLS(0) 다음의 본문 위치
    original = seq[pos].item()
    seq[pos] = MASK_IDX
    with torch.no_grad():
        logits = model(seq.unsqueeze(0))
    pred = logits[0, pos].argmax().item()
    assert pred == original, f"mask predict mismatch: {pred} != {original}"
