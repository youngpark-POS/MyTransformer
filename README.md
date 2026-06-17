# Transformer (from scratch, PyTorch)

"Attention is All You Need"(Vaswani et al., 2017)의 **인코더-디코더 Transformer**를
PyTorch로 직접 구현한 EN→DE 기계번역 프로젝트입니다. Multi-head attention, positional
encoding, masking을 `nn.Transformer` 없이 손으로 구현했고, 토크나이저/단어사전도 직접
만들었습니다(spaCy/torchtext 미사용).

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

## 설계 요점

- **From scratch**: scaled dot-product attention, multi-head 분할/병합, sinusoidal PE,
  pad/causal 마스크를 직접 구현.
- **Pre-norm** residual(`x + Sublayer(LayerNorm(x))`)로 학습 안정성 확보.
- **Teacher forcing** 학습 + **자가회귀** 추론(greedy/beam, beam은 길이 정규화 적용).
- **Noam warmup** LR 스케줄 + label smoothing(0.1).
- 데이터셋은 `datasets`의 `bentrevett/multi30k`를 사용(torchtext 회피).

자세한 아키텍처/관례는 [CLAUDE.md](CLAUDE.md) 참고.
