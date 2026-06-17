"""통합 테스트: 합성 복사(copy) 태스크를 과적합시켜 전체 파이프라인이
실제로 학습되는지(loss 감소 + greedy 디코딩 정확) 검증한다.

외부 다운로드 없이 수 초 내 실행된다.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import BOS_IDX, EOS_IDX, PAD_IDX
from transformer.data.dataset import collate_batch
from transformer.model.transformer import Transformer
from transformer.train import run_epoch
from transformer.translate import greedy_decode
from transformer.utils import NoamLR, set_seed


def _make_copy_data(n: int):
    """tgt == src 인 복사 태스크 샘플 생성."""
    data = []
    for _ in range(n):
        length = torch.randint(3, 6, (1,)).item()
        body = torch.randint(4, 15, (length,))
        seq = torch.cat([torch.tensor([BOS_IDX]), body, torch.tensor([EOS_IDX])])
        data.append((seq, seq.clone()))
    return data


def test_overfit_copy_task():
    set_seed(0)
    device = torch.device("cpu")
    data = _make_copy_data(96)
    loader = DataLoader(data, batch_size=16, shuffle=True, collate_fn=collate_batch)

    model = Transformer(15, 15, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_len=20)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=64, warmup_steps=100)

    first_loss = run_epoch(model, loader, criterion, device, optimizer, scheduler)
    for _ in range(29):
        last_loss = run_epoch(model, loader, criterion, device, optimizer, scheduler)

    # 1) loss가 유의하게 감소했는가
    assert last_loss < first_loss * 0.5, f"loss did not drop: {first_loss}->{last_loss}"

    # 2) 학습한 복사 태스크를 greedy 디코딩이 정확히 재현하는가
    src = data[0][0].unsqueeze(0)
    decoded = greedy_decode(model, src, device, max_len=20)
    assert decoded == data[0][0].tolist(), f"copy mismatch: {decoded}"
