import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer
from torch.utils.data import BatchSampler

from src.batch_sampling.no_duplicate_labels_sampler import NoDuplicateLabelsBatchSampler


class CustomSentenceTransformerTrainer(SentenceTransformerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_batch_sampler(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
    ) -> BatchSampler | None:
        return NoDuplicateLabelsBatchSampler(dataset, batch_size, shuffle=True, drop_last=drop_last)
