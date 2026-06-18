"""중앙 하이퍼파라미터/경로 설정.

원논문(Vaswani et al., 2017)의 base 모델보다 축소된 기본값을 사용해
CPU/단일 GPU에서도 빠르게 학습이 돌도록 한다. 모든 값은 dataclass 필드라
스크립트에서 손쉽게 덮어쓸 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 프로젝트 루트 기준 경로
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"


@dataclass
class ModelConfig:
    """Transformer 아키텍처 하이퍼파라미터."""
    d_model: int = 512          # 임베딩/모델 차원
    n_heads: int = 8            # multi-head attention head 개수 (d_model % n_heads == 0)
    n_layers: int = 6           # 인코더/디코더 스택 깊이
    d_ff: int = 2048            # position-wise FFN 내부 차원 (관행상 4*d_model)
    dropout: float = 0.3        # Multi30k(~29k쌍) 과적합 방지를 위해 base보다 높게
    max_len: int = 128          # positional encoding 최대 길이
    tie_embeddings: bool = True # 디코더 입력 임베딩과 출력 projection 가중치 공유(정규화+파라미터 절감)

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model은 n_heads로 나누어떨어져야 합니다."


@dataclass
class TrainConfig:
    """학습 루프 하이퍼파라미터."""
    batch_size: int = 128
    epochs: int = 30            # 더 큰 모델은 수렴이 느림 — val_loss 곡선 보며 조정
    warmup_steps: int = 1500    # Noam LR warmup. Multi30k(~226 step/epoch)에 맞춰 축소
                                # (4000=~18epoch 워밍업이라 과대 → 1500=~6.6epoch에 LR 정점)
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    seed: int = 42
    min_freq: int = 2           # vocab 빈도 컷오프
    device: str = "cuda"        # 런타임에 가용성 확인 후 cpu로 대체될 수 있음


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    src_lang: str = "en"
    tgt_lang: str = "de"
    data_dir: Path = DATA_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR


# 특수 토큰 (vocab 인덱스 고정)
PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<bos>", "<eos>", "<unk>"
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
