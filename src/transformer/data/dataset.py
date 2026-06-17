"""Multi30k(EN->DE) 로드, 토큰화, vocab 구축, 배치 패딩까지의 데이터 파이프라인.

torchtext가 유지보수 중단 상태이므로 HuggingFace `datasets`의
`bentrevett/multi30k`(en/de 평행 코퍼스, train ~29k)를 사용한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import Config, PAD_IDX
from .tokenizer import tokenize
from .vocab import Vocab

HF_DATASET = "bentrevett/multi30k"


class TranslationDataset(Dataset):
    """(src_ids, tgt_ids) 정수 텐서 쌍을 제공하는 Dataset.

    토큰화/인코딩은 생성 시점에 미리 수행해 둔다(데이터가 작아 메모리에 충분).
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
    ) -> None:
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for src_text, tgt_text in pairs:
            src_ids = src_vocab.encode(tokenize(src_text))
            tgt_ids = tgt_vocab.encode(tokenize(tgt_text))
            self.examples.append(
                (torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long))
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


def collate_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """가변 길이 시퀀스를 PAD_IDX로 패딩해 (batch, seq_len) 텐서로 묶는다."""
    src_seqs, tgt_seqs = zip(*batch)
    src = pad_sequence(list(src_seqs), batch_first=True, padding_value=PAD_IDX)
    tgt = pad_sequence(list(tgt_seqs), batch_first=True, padding_value=PAD_IDX)
    return src, tgt


def _load_pairs(split: str, src_lang: str, tgt_lang: str) -> List[Tuple[str, str]]:
    """HuggingFace에서 한 split을 받아 (src, tgt) 문자열 쌍 리스트로 변환."""
    from datasets import load_dataset  # 지연 import: 테스트가 무거운 의존성을 강제하지 않도록

    ds = load_dataset(HF_DATASET, split=split)
    return [(ex[src_lang], ex[tgt_lang]) for ex in ds]


def build_dataloaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocab, Vocab]:
    """train/val/test DataLoader와 src/tgt vocab을 구성해 반환한다.

    vocab은 학습셋 기준으로만 구축하고, 한번 만들면 디스크에 캐시한다.
    """
    train_pairs = _load_pairs("train", cfg.src_lang, cfg.tgt_lang)
    val_pairs = _load_pairs("validation", cfg.src_lang, cfg.tgt_lang)
    test_pairs = _load_pairs("test", cfg.src_lang, cfg.tgt_lang)

    src_vocab, tgt_vocab = _get_or_build_vocabs(cfg, train_pairs)

    def make_loader(pairs: List[Tuple[str, str]], shuffle: bool) -> DataLoader:
        dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
        return DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            collate_fn=collate_batch,
        )

    return (
        make_loader(train_pairs, shuffle=True),
        make_loader(val_pairs, shuffle=False),
        make_loader(test_pairs, shuffle=False),
        src_vocab,
        tgt_vocab,
    )


def _get_or_build_vocabs(
    cfg: Config, train_pairs: List[Tuple[str, str]]
) -> Tuple[Vocab, Vocab]:
    """캐시가 있으면 로드, 없으면 학습셋에서 vocab을 구축해 저장."""
    src_path = Path(cfg.data_dir) / f"vocab_{cfg.src_lang}.json"
    tgt_path = Path(cfg.data_dir) / f"vocab_{cfg.tgt_lang}.json"

    if src_path.exists() and tgt_path.exists():
        return Vocab.load(src_path), Vocab.load(tgt_path)

    min_freq = cfg.train.min_freq
    src_vocab = Vocab.build((tokenize(s) for s, _ in train_pairs), min_freq=min_freq)
    tgt_vocab = Vocab.build((tokenize(t) for _, t in train_pairs), min_freq=min_freq)
    src_vocab.save(src_path)
    tgt_vocab.save(tgt_path)
    return src_vocab, tgt_vocab
