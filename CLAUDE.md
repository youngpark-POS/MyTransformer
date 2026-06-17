# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

From-scratch PyTorch implementation of the "Attention is All You Need" encoder-decoder
Transformer for **EN→DE machine translation** on the Multi30k dataset. The core blocks
(multi-head attention, positional encoding, masking) are implemented by hand — `nn.Transformer`
and `nn.MultiheadAttention` are intentionally **not** used. Tokenizer and vocabulary are also
hand-built (no spaCy/torchtext).

## Commands

```bash
# 환경 (src 레이아웃 — 설치하거나 PYTHONPATH로 src를 잡아야 import됨)
pip install -r requirements.txt
pip install -e .                          # 또는: export PYTHONPATH=src

# 테스트 (외부 다운로드 불필요, 수 초)
pytest -q                                 # 전체
pytest tests/test_attention.py            # 한 파일
pytest tests/test_overfit.py::test_overfit_copy_task   # 단일 테스트

# 학습 (첫 실행 시 HuggingFace에서 Multi30k 다운로드 + vocab 캐시 생성)
python -m transformer.train --epochs 10
python -m transformer.train --epochs 1 --device cpu     # 빠른 점검
python -m transformer.train --epochs 10 --eval-bleu     # epoch마다 BLEU(느림)

# 추론 (checkpoints/best.pt 필요)
python -m transformer.translate --text "a man is riding a bike"
python -m transformer.translate --text "..." --beam 5   # beam search
```

`pyproject.toml`이 `pytest`의 `pythonpath`에 `src`,`.`을 추가하므로 테스트는 설치 없이도 돈다.
직접 스크립트를 돌릴 때는 `pip install -e .` 또는 `PYTHONPATH=src`가 필요하다.

## Architecture

데이터 흐름: **dataset → model → train(teacher forcing) → translate(autoregressive)**.

- **설정의 단일 출처는 `config.py`** (`ModelConfig`/`TrainConfig` dataclass). 특수 토큰 인덱스
  `PAD_IDX=0, BOS_IDX=1, EOS_IDX=2, UNK_IDX=3`이 전역 상수로 고정돼 있고, 데이터·모델·loss·
  디코딩이 모두 이 값을 공유한다. 토큰 추가/변경 시 여기부터 본다.

- **데이터 (`src/transformer/data/`)**: `tokenizer.py`(정규식) → `vocab.py`(학습셋에서 빈도순
  구축, `min_freq` 컷오프, JSON 캐시) → `dataset.py`(HuggingFace `bentrevett/multi30k` 로드,
  `collate_batch`가 `PAD_IDX`로 패딩). vocab은 **학습셋 기준으로만** 만들고 `data/`에 캐시한다.

- **모델 (`src/transformer/model/`)**: `attention.py`(scaled dot-product + MultiHeadAttention) →
  `positional.py`(sinusoidal, buffer) → `feedforward.py` → `layers.py`(Encoder/DecoderLayer) →
  `encoder.py`/`decoder.py` → `transformer.py`(조립 + 마스크 생성). **Pre-norm** residual 구조
  (`x + Sublayer(LayerNorm(x))`)를 쓴다 — 원논문 post-norm과 다르므로 레이어 수정 시 유의.

- **마스크 규약 (`transformer.py`)**: `True=유지, False=차단`. `scaled_dot_product_attention`이
  `mask==0` 위치를 `-inf`로 채운다. `make_pad_mask`(패딩 차단)와 `make_causal_mask`(미래 차단)를
  `make_tgt_mask`에서 `&`로 결합한다. **마스크 규약을 바꾸면 attention의 `masked_fill`도 함께 바꿔야 한다.**

- **학습 (`train.py`)**: `run_epoch`가 학습/평가 겸용(`optimizer` 유무로 분기). Teacher forcing은
  디코더 입력 `tgt[:, :-1]`, 정답 `tgt[:, 1:]`. Loss는 `CrossEntropyLoss(ignore_index=PAD_IDX,
  label_smoothing=0.1)`. LR은 `utils.NoamLR`(warmup 스케줄)이 매 step Adam의 lr을 덮어쓰므로
  **Adam은 반드시 `lr=0`으로 생성**한다. best 체크포인트에 모델+vocab(`itos`)+cfg를 함께 저장.

- **추론 (`translate.py`)**: 인코더를 한 번만 돌려 `memory`를 캐싱하고 `<bos>`부터 자가회귀 디코딩.
  `greedy_decode`(argmax)와 `beam_search_decode`(길이 정규화 `score/len^0.7`). 체크포인트에
  vocab이 들어 있어 추론은 별도 vocab 파일 없이 복원된다.

## Conventions / gotchas

- `config.device` 기본값은 `"cuda"`지만 `utils.resolve_device`가 CUDA 미가용 시 CPU로 대체한다.
- 시퀀스 텐서는 모두 **batch-first** `(batch, seq_len)`. 마스크는 `(batch, 1, q_len, k_len)`로 broadcast.
- 의존성 최소화 원칙: 새 기능에 spaCy/torchtext 등을 끌어오지 말 것. `datasets`는 `dataset.py`
  안에서 **지연 import**해 토크나이저/모델 테스트가 무거운 의존성을 강제하지 않게 한다.
- 모델/데이터 변경의 1차 검증은 항상 `pytest`(특히 `test_overfit.py`)로 — 다운로드 없이 회귀를 잡는다.
