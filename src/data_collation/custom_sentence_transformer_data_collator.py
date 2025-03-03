from typing import Any

import torch
from sentence_transformers.data_collator import SentenceTransformerDataCollator


class CustomSentenceTransformerDataCollator(SentenceTransformerDataCollator):
    def __init__(self, *args, skip_columns: list[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_columns = skip_columns or []


    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        # 2. Process skip_columns first:
        for skip_col in self.skip_columns:
            if skip_col in column_names:
                # If you want to keep them in the batch, do it here:
                batch[skip_col] = [row[skip_col] for row in features]
                # Remove from the list so they won't be tokenized
                column_names.remove(skip_col)


        for column_name in column_names:
            # If the prompt length has been set, we should add it to the batch
            if column_name.endswith("_prompt_length") and column_name[: -len("_prompt_length")] in column_names:
                batch[column_name] = torch.tensor([row[column_name] for row in features], dtype=torch.int)
                continue

            tokenized = self.tokenize_fn([row[column_name] for row in features])
            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value

        return batch