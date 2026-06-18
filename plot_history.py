"""학습 히스토리(checkpoints/history.json) 시각화.

train.py가 epoch마다 저장한 메트릭을 읽어 loss/accuracy(있으면 BLEU) 곡선을
그린다. CLI에서는 PNG로 저장하고, Colab/노트북에서는 plot_history()를 직접
호출해 plt.show()로 인라인 표시할 수 있다.

사용법:
    python plot_history.py                       # checkpoints/history.json -> checkpoints/history.png
    python plot_history.py --history path.json --out curves.png
    python plot_history.py --show                # 창으로 표시(데스크톱)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


def load_history(path: Path) -> List[dict]:
    """history.json(epoch별 메트릭 dict의 리스트)을 읽는다."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_history(history: List[dict], out: Optional[Path] = None, show: bool = False):
    """loss/acc(있으면 BLEU) 곡선을 그린다.

    BLEU(--eval-bleu)는 기록에 있을 때만 세 번째 패널로 추가된다.
    out이 주어지면 PNG로 저장, show=True면 화면에 표시.
    """
    if not history:
        raise ValueError("history가 비어 있습니다. 먼저 학습을 1 epoch 이상 돌리세요.")

    epochs = [h["epoch"] for h in history]
    has_bleu = any("val_bleu" in h for h in history)
    n_panels = 3 if has_bleu else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))

    # 1) Loss
    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], "s-", label="val")
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Accuracy (토큰 단위)
    ax = axes[1]
    ax.plot(epochs, [h["train_acc"] for h in history], "o-", label="train")
    ax.plot(epochs, [h["val_acc"] for h in history], "s-", label="val")
    ax.set_title("Token accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) BLEU (기록이 있을 때만)
    if has_bleu:
        ax = axes[2]
        bleu_epochs = [h["epoch"] for h in history if "val_bleu" in h]
        bleu_vals = [h["val_bleu"] for h in history if "val_bleu" in h]
        ax.plot(bleu_epochs, bleu_vals, "^-", color="tab:green", label="val BLEU")
        ax.set_title("Validation BLEU")
        ax.set_xlabel("epoch")
        ax.set_ylabel("BLEU")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        print(f"saved -> {out}")
    if show:
        plt.show()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=Path("checkpoints/history.json"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/history.png"))
    parser.add_argument("--show", action="store_true", help="창으로 표시(데스크톱)")
    args = parser.parse_args()

    history = load_history(args.history)
    plot_history(history, out=args.out, show=args.show)


if __name__ == "__main__":
    main()
