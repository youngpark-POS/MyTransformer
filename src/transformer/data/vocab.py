"""학습 코퍼스에서 구축하는 단어 사전(Vocabulary).

특수 토큰(<pad>,<bos>,<eos>,<unk>)을 config에 고정된 인덱스(0~3)로 예약하고,
나머지 단어를 빈도 내림차순으로 추가한다. 빈도가 min_freq 미만인 단어는
<unk>으로 처리한다.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from config import (
    BOS_IDX,
    BOS_TOKEN,
    EOS_IDX,
    EOS_TOKEN,
    PAD_IDX,
    SPECIAL_TOKENS,
    UNK_IDX,
    UNK_TOKEN,
)


class Vocab:
    """토큰 <-> 정수 인덱스 양방향 매핑."""

    def __init__(self, itos: List[str]) -> None:
        self.itos: List[str] = itos
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(itos)}
        # 특수 토큰 인덱스가 config 약속과 일치하는지 검증
        assert self.stoi[SPECIAL_TOKENS[PAD_IDX]] == PAD_IDX
        assert self.stoi[SPECIAL_TOKENS[BOS_IDX]] == BOS_IDX
        assert self.stoi[SPECIAL_TOKENS[EOS_IDX]] == EOS_IDX
        assert self.stoi[SPECIAL_TOKENS[UNK_IDX]] == UNK_IDX

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, token_sequences: Iterable[List[str]], min_freq: int = 2) -> "Vocab":
        """토큰화된 문장들의 반복자로부터 vocab을 만든다."""
        counter: Counter[str] = Counter()
        for tokens in token_sequences:
            counter.update(tokens)
        # 특수 토큰을 먼저, 그 뒤로 빈도>=min_freq 단어를 빈도 내림차순으로
        itos = list(SPECIAL_TOKENS)
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in SPECIAL_TOKENS:
                itos.append(token)
        return cls(itos)

    def encode(self, tokens: List[str], add_bos_eos: bool = True) -> List[int]:
        """토큰 리스트 -> 인덱스 리스트. 미등록 단어는 <unk>."""
        ids = [self.stoi.get(tok, UNK_IDX) for tok in tokens]
        if add_bos_eos:
            ids = [BOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> List[str]:
        """인덱스 리스트 -> 토큰 리스트. 특수 토큰은 기본적으로 제거."""
        special = {PAD_IDX, BOS_IDX, EOS_IDX} if strip_special else set()
        return [self.itos[i] for i in ids if i not in special]

    # --- 영속화 ---------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        itos = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(itos)
