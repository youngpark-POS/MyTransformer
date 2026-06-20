# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

From-scratch PyTorch implementation of the "Attention is All You Need" encoder-decoder
Transformer for **EN→DE machine translation** on the Multi30k dataset. The core blocks
(multi-head attention, positional encoding, masking) are implemented by hand — `nn.Transformer`
and `nn.MultiheadAttention` are intentionally **not** used. Tokenizer and vocabulary are also
hand-built (no spaCy/torchtext).

추가로 같은 인코더 부품을 재사용한 **encoder-only BERT(MLM 사전학습)**가 `src/transformer/bert/`에
들어 있다(wikitext-2 코퍼스). MLM 전용이지만 풀 BERT(NSP)·fine-tuning으로 확장 가능한 골격이다.

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
# 매 epoch 전체 상태를 checkpoints/last.pt에 저장. 재실행 시 자동으로 이어서 학습한다.
python -m transformer.train --epochs 10 --fresh         # last.pt 무시하고 처음부터

# 추론 (checkpoints/best.pt 필요)
python -m transformer.translate --text "a man is riding a bike"
python -m transformer.translate --text "..." --beam 5   # beam search

# BERT MLM 사전학습 (첫 실행 시 wikitext-2 다운로드 + data/vocab_mlm.json 캐시)
python -m transformer.bert.train --epochs 10
python -m transformer.bert.train --epochs 1 --device cpu   # 빠른 점검
python -m transformer.bert.train --epochs 10 --fresh       # bert_last.pt 무시하고 처음부터
pytest tests/test_bert_overfit.py                          # 다운로드 없는 MLM 회귀 검증
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
  **Adam은 반드시 `lr=0`으로 생성**한다. best 체크포인트(`best.pt`)에 모델+vocab(`itos`)+cfg를
  함께 저장. 또한 **매 epoch `last.pt`에 전체 학습 상태**(model/optimizer/scheduler `_step`/epoch/
  best_val/history)를 덮어써 저장하고, 재실행 시 `utils.maybe_resume`가 이를 불러와 자동으로 이어 학습한다
  (`--fresh`로 무시). BERT도 `bert_last.pt`로 동일.

- **추론 (`translate.py`)**: 인코더를 한 번만 돌려 `memory`를 캐싱하고 `<bos>`부터 자가회귀 디코딩.
  `greedy_decode`(argmax)와 `beam_search_decode`(길이 정규화 `score/len^0.7`). 체크포인트에
  vocab이 들어 있어 추론은 별도 vocab 파일 없이 복원된다.

## BERT (`src/transformer/bert/`)

encoder-only BERT를 MLM으로 사전학습한다. **번역용 인코더 부품을 그대로 재사용**한다:
`EncoderLayer`(pre-norm self-attn+FFN) 스택, `PositionalEncoding`(sinusoidal),
`make_pad_mask`(causal 없이 양방향), 그리고 데이터 쪽 `tokenize`/`Vocab`/`pad_sequence`.

- **특수 토큰**: `[CLS]=<bos>(1)`, `[SEP]=<eos>(2)`로 기존 토큰을 의미만 바꿔 재사용하고,
  새로 필요한 `<mask>`만 `MASK_IDX=4`에 예약한다(`config.py`). `SPECIAL_TOKENS`와 번역 vocab
  캐시는 건드리지 않아 번역 경로는 무영향. `Vocab.encode()`가 붙이는 `[<bos>]..[<eos>]`가 곧
  `[CLS]..[SEP]`라 추가 래핑 없이 그대로 쓴다.

- **설정**: `config.py`의 `BertModelConfig`/`BertTrainConfig`/`BertConfig`. 번역과 별도 dataclass다.

- **데이터 (`bert/data.py`)**: `load_wikitext_lines`(HuggingFace `wikitext-2-raw-v1`, 지연 import) →
  `build_mlm_vocab`(번역 `Vocab.build` 후 `<mask>`를 idx 4에 삽입) → `MLMDataset`(문장당 한 예제) →
  `MLMCollator`(매 배치 **동적 80/10/10 마스킹**: 80% `<mask>`, 10% 랜덤, 10% 유지). 마스킹 결과
  `labels`는 예측 대상 위치만 원본 토큰, 나머지는 `PAD_IDX`라 `ignore_index`로 자동 제외된다.
  PAD/CLS/SEP은 절대 마스킹하지 않는다. vocab은 `data/vocab_mlm.json`에 캐시.

- **모델 (`bert/model.py`)**: `BertEmbeddings`(token + position + **segment**) → `BertModel`(백본:
  `EncoderLayer` 스택 + 최종 norm + `[CLS]` pooler) → `BertMLMHead`(dense→GELU→LayerNorm→vocab,
  토큰 임베딩과 weight tying) → `BertForMaskedLM`. **백본과 헤드를 분리**하고 segment 임베딩 슬롯과
  pooler를 미리 둔 것이 확장 포인트다 — 풀 BERT(NSP)·분류 fine-tuning은 *같은 `BertModel` + 다른 헤드*,
  문장쌍 입력은 `segment_ids`만 채우면 된다(백본·vocab 무변경).

- **학습 (`bert/train.py`)**: `run_mlm_epoch`는 번역 `run_epoch`와 같은 구조(`optimizer` 유무로 분기,
  `NoamLR`+`Adam(lr=0)`)지만 손실/정확도를 **마스킹된 위치(`labels!=PAD_IDX`)에서만** 집계한다.
  체크포인트는 `checkpoints/bert_best.pt`(번역 `best.pt`와 분리), 곡선은 `bert_history.json`에 기록 →
  기존 `plot_history.py`로 그대로 시각화. Colab은 `colab_bert.ipynb`(사전학습+곡선+fill-mask 데모).

## Conventions / gotchas

- `config.device` 기본값은 `"cuda"`지만 `utils.resolve_device`가 CUDA 미가용 시 CPU로 대체한다.
- 시퀀스 텐서는 모두 **batch-first** `(batch, seq_len)`. 마스크는 `(batch, 1, q_len, k_len)`로 broadcast.
- 의존성 최소화 원칙: 새 기능에 spaCy/torchtext 등을 끌어오지 말 것. `datasets`는 `dataset.py`
  안에서 **지연 import**해 토크나이저/모델 테스트가 무거운 의존성을 강제하지 않게 한다.
- 모델/데이터 변경의 1차 검증은 항상 `pytest`로 — 다운로드 없이 회귀를 잡는다. 번역은
  `test_overfit.py`(복사 태스크), BERT는 `test_bert_overfit.py`(MLM 마스크 복원 태스크)가 담당한다.
