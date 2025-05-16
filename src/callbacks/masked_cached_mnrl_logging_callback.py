import io

import numpy as np
import wandb
from matplotlib import pyplot as plt

from transformers import TrainerCallback, TrainingArguments, TrainerState
from wandb.sdk.wandb_run import Run

from src.losses.MaskedCachedMultipleNegativesRankingLoss import MaskedCachedMultipleNegativesRankingLoss


class MaskLoggingCallback(TrainerCallback):
    """Streams per-batch stats and per-epoch heat-maps to W&B."""

    def __init__(self, loss_module: MaskedCachedMultipleNegativesRankingLoss, run: Run):
        self.loss_module = loss_module
        self.run = run

    # ────────────────────────────────────────── step-level
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        stats = self.loss_module.pop_batch_metrics()
        if stats:
            self.run.log(stats)
        self.log_heatmap()

    # ────────────────────────────────────────── epoch-level
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        self.log_heatmap()

    def log_heatmap(self):
        heatmap_dict = self.loss_module.pop_epoch_heatmaps()
        for scenario, counts in heatmap_dict.items():
            if not counts:
                continue

            anchor_labels = sorted({a for a, _ in counts})
            negative_labels = sorted({n for _, n in counts})

            matrix = np.zeros((len(anchor_labels), len(negative_labels)), dtype=int)
            for (a_lbl, n_lbl), cnt in counts.items():
                matrix[anchor_labels.index(a_lbl), negative_labels.index(n_lbl)] = cnt

            # Matplotlib heat-map → PNG bytes → W&B Image
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(matrix, cmap="viridis")
            ax.set_xticks(np.arange(len(negative_labels)), labels=negative_labels, rotation=90)
            ax.set_yticks(np.arange(len(anchor_labels)), labels=anchor_labels)
            ax.set_title(f"Label overlap – {scenario}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=120)
            plt.close(fig)
            buf.seek(0)
            self.run.log({f"{scenario}/label_overlap_heatmap": wandb.Image(buf)})
