# Transformer (from scratch, PyTorch)

"Attention is All You Need"(Vaswani et al., 2017)의 Encoder-Decoder Transformer를
PyTorch로 직접 구현한 EN→DE 기계번역 프로젝트입니다. Multi-head attention, positional
encoding, masking을 `nn.Transformer` 없이 구현했습니다.

## 구조

```
config.py                  # 하이퍼파라미터/특수토큰 인덱스(단일 출처)
src/transformer/
  data/   tokenizer.py · vocab.py · dataset.py    # 전처리 + Multi30k 로드
  model/  attention · positional · feedforward    # 핵심 블록 (직접 구현)
          layers · encoder · decoder · transformer
  train.py        # 학습 (teacher forcing, Noam LR, label smoothing, BLEU)
  translate.py    # 추론 (greedy / beam search)
  utils.py        # 시드·디바이스·LR스케줄·체크포인트
tests/            # 단위 + 과적합 통합 테스트 (다운로드 불필요)
```

## 설치

```bash
pip install -r requirements.txt
pip install -e .          # src 레이아웃 — 또는 PYTHONPATH=src
```

## 사용

```bash
# 테스트 (수 초, 외부 다운로드 없음)
pytest -q

# 학습 (첫 실행 시 HuggingFace에서 Multi30k 자동 다운로드)
python -m transformer.train --epochs 10

# 추론
python -m transformer.translate --text "a man is riding a bike"
python -m transformer.translate --text "a man is riding a bike" --beam 5
```
