"""MLM 사전학습용 데이터 파이프라인: wikitext 로드 → vocab → 동적 마스킹.

번역 파이프라인(data/)의 부품을 최대한 재사용한다:
- tokenize: 정규식 토크나이저 그대로
- Vocab: build/encode/save/load 그대로. encode()가 [<bos>]..[<eos>]를 붙이는데,
  BERT에선 이것이 곧 [CLS]..[SEP]이므로 추가 래핑 없이 그대로 쓴다.
- pad_sequence: PAD_IDX 패딩 그대로

새로 추가되는 건 <mask> 토큰 예약과 동적 마스킹(MLMCollator)뿐이다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import (
    BertConfig,
    CLS_IDX,
    MASK_IDX,
    MASK_TOKEN,
    PAD_IDX,
    SEP_IDX,
    UNK_IDX,
)
from ..data.tokenizer import tokenize
from ..data.vocab import Vocab

# wikitext-103(raw): 약 100M 토큰의 영어 단일언어 코퍼스(BERT급 학습용).
# 최신 huggingface_hub는 repo id가 'namespace/name' 형식이어야 하므로 정식 ID(Salesforce/wikitext)를 쓴다.
WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1")


def build_mlm_vocab(
    token_sequences: Iterable[List[str]],
    min_freq: int = 3,
    max_size: int | None = None,
) -> Vocab:
    """번역용 Vocab.build를 재사용하되 <mask>를 인덱스 4에 끼워 넣는다.

    base.itos = [<pad>,<bos>,<eos>,<unk>, 단어(빈도 내림차순)...] 이므로 앞 4개
    (특수토큰) 뒤, 실제 단어들 앞(인덱스 4)에 <mask>를 삽입한다. 그러면 0~3
    인덱스 약속이 유지되어 Vocab.__init__의 특수토큰 assert를 그대로 통과한다.

    max_size를 주면 빈도 상위 단어만 남겨 최종 vocab 크기를 max_size로 제한한다
    (wikitext-103처럼 어휘가 수십만인 코퍼스에서 임베딩/softmax 비대화를 막는다).
    """
    base = Vocab.build(token_sequences, min_freq=min_freq)
    itos = base.itos
    if max_size is not None and len(itos) >= max_size:
        itos = itos[: max_size - 1]  # <mask> 1칸을 더해 최종이 정확히 max_size가 되도록
    n_special = UNK_IDX + 1  # 4
    itos = itos[:n_special] + [MASK_TOKEN] + itos[n_special:]
    return Vocab(itos)


def load_wikitext_lines(split: str) -> List[str]:
    """wikitext 한 split을 받아 학습에 쓸 텍스트 줄 리스트로 정리한다.

    wikitext는 빈 줄과 ' = 제목 = ' 형태의 헤더 줄이 많아 걸러낸다.
    """
    from datasets import load_dataset  # 지연 import: 테스트가 무거운 의존성을 강제하지 않도록

    ds = load_dataset(*WIKITEXT, split=split)
    lines: List[str] = []
    for ex in ds:
        text = ex["text"].strip()
        if not text or text.startswith("="):
            continue
        lines.append(text)
    return lines


class MLMDataset(Dataset):
    """한 줄 = 한 학습 예제. [CLS] tok... [SEP] 정수 시퀀스를 미리 인코딩해 둔다.

    마스킹은 여기서 하지 않고 collator에서 매 배치 동적으로 적용한다
    (RoBERTa식 dynamic masking — 같은 문장도 epoch마다 다른 위치가 마스킹됨).
    """

    def __init__(self, lines: List[str], vocab: Vocab, max_len: int = 128, min_tokens: int = 4) -> None:
        self.examples: List[torch.Tensor] = []
        body_len = max_len - 2  # [CLS], [SEP] 자리 확보
        for text in lines:
            # 본문 토큰만(CLS/SEP 없이) 얻은 뒤, max_len을 넘으면 잘라 버리지 않고
            # body_len 단위로 쪼개 여러 예제로 만든다(truncation 대신 chunking).
            body = vocab.encode(tokenize(text), add_bos_eos=False)
            for start in range(0, len(body), body_len):
                chunk = body[start : start + body_len]
                ids = [CLS_IDX] + chunk + [SEP_IDX]
                if len(ids) < min_tokens:  # 자투리 너무 짧은 청크는 버림
                    continue
                self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.examples[idx]


class MLMCollator:
    """배치를 패딩하고 BERT의 80/10/10 마스킹을 동적으로 적용한다.

    반환: (input_ids, labels)
      - input_ids: 마스킹이 적용된 입력 (일부 위치가 <mask>/랜덤/원본)
      - labels: 예측 대상 위치만 원본 토큰, 나머지는 PAD_IDX(=ignore_index)

    특수 토큰(PAD/CLS/SEP)은 절대 마스킹 대상에 넣지 않는다.
    """

    def __init__(self, vocab_size: int, mask_prob: float = 0.15) -> None:
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob

    def __call__(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)
        labels = input_ids.clone()

        # 1) 마스킹 후보 선정: 특수 토큰 제외 위치에서 mask_prob 확률로 선택
        special = (input_ids == PAD_IDX) | (input_ids == CLS_IDX) | (input_ids == SEP_IDX)
        probs = torch.full(input_ids.shape, self.mask_prob)
        probs[special] = 0.0
        masked = torch.bernoulli(probs).bool()

        # 2) 선택되지 않은 위치는 손실에서 무시(PAD_IDX = ignore_index)
        labels[~masked] = PAD_IDX

        # 3) 선택된 위치를 80% <mask> / 10% 랜덤 / 10% 원본 유지로 치환
        replace = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked
        input_ids[replace] = MASK_IDX
        # 남은 20% 중 절반(=전체의 10%)을 랜덤 토큰으로 (특수토큰 0~4는 제외)
        rand = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked & ~replace
        random_tokens = torch.randint(MASK_IDX + 1, self.vocab_size, input_ids.shape)
        input_ids[rand] = random_tokens[rand]
        # 나머지 10%는 원본 그대로 둔다(아무 작업 없음)

        return input_ids, labels


def _get_or_build_mlm_vocab(cfg: BertConfig, train_lines: List[str]) -> Vocab:
    """캐시가 있으면 로드, 없으면 학습셋에서 MLM vocab을 구축해 저장.

    캐시 파일명에 코퍼스 이름을 넣어 wikitext-2/103 전환 시 옛 캐시를 재사용하지 않는다.
    """
    path = Path(cfg.data_dir) / f"vocab_mlm_{WIKITEXT[1]}.json"
    if path.exists():
        return Vocab.load(path)
    vocab = build_mlm_vocab(
        (tokenize(line) for line in train_lines),
        min_freq=cfg.train.min_freq,
        max_size=cfg.train.max_vocab_size,
    )
    vocab.save(path)
    return vocab


def build_mlm_dataloaders(cfg: BertConfig) -> Tuple[DataLoader, DataLoader, Vocab]:
    """train/validation DataLoader와 vocab을 구성해 반환한다."""
    train_lines = load_wikitext_lines("train")
    val_lines = load_wikitext_lines("validation")
    vocab = _get_or_build_mlm_vocab(cfg, train_lines)
    collator = MLMCollator(len(vocab), cfg.train.mask_prob)

    def make_loader(lines: List[str], shuffle: bool) -> DataLoader:
        dataset = MLMDataset(lines, vocab, cfg.model.max_len)
        return DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            collate_fn=collator,
        )

    return make_loader(train_lines, True), make_loader(val_lines, False), vocab
