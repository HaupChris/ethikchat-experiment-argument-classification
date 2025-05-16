import io

import numpy as np
import pandas as pd
import wandb
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

    # ────────────────────────────────────────── epoch-level
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        self.log_heatmap(state.epoch)

    def log_heatmap(self, epoch: int):
        heatmap_dict = self.loss_module.pop_epoch_heatmaps()
        for scenario, counts in heatmap_dict.items():
            if not counts:
                continue

            anchor_labels = sorted({a for a, _ in counts})
            negative_labels = sorted({n for _, n in counts})

            matrix = np.zeros((len(anchor_labels), len(negative_labels)), dtype=int)
            for (anchor_label, negative_label), count in counts.items():
                matrix[anchor_labels.index(anchor_label), negative_labels.index(negative_label)] = count

            result_df = pd.DataFrame(matrix, index=anchor_labels, columns=negative_labels)

            self.run.log({f"{scenario}/label_overlap_heatmap_{epoch}": wandb.Table(dataframe=result_df)})
