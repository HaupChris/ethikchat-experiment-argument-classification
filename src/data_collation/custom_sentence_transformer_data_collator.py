from dataclasses import dataclass
from typing import Any

import torch
from sentence_transformers.data_collator import SentenceTransformerDataCollator


@dataclass
class CustomSentenceTransformerDataCollator(SentenceTransformerDataCollator):
    """Collator for a SentenceTransformers model with support for label-aware masking."""

    # Add a new parameter but with default value of False for backward compatibility
    handle_specialized_label_columns: bool = False


    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the regular label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        # NEW: Process specialized label columns when handling is enabled
        if self.handle_specialized_label_columns:
            # Find all specialized label columns (ending with _label)
            specialized_label_columns = [col for col in column_names if col.endswith("_label")]

            for label_column in specialized_label_columns:
                # Get all label values from this column
                label_data = [row[label_column] for row in features]
                batch[label_column] = label_data


        # Process remaining columns for tokenization
        for column_name in column_names:
            # Skip specialized label columns as they've already been processed
            if self.handle_specialized_label_columns and column_name.endswith("_label"):
                continue

            # Handle prompt length columns
            if column_name.endswith("_prompt_length") and column_name[: -len("_prompt_length")] in column_names:
                batch[column_name] = torch.tensor([row[column_name] for row in features], dtype=torch.int)
                continue

            # Tokenize text columns
            # gibt vermutlich lists aus, wenn in features daten dabei sind, die nicht in einen tensor convertiert werden können.
            # lösung: daten, die nicht convertiert werden können aus features entfernen und anderweitig zum batch hinzufügen.
            tokenized = self.tokenize_fn([row[column_name] for row in features])
            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value

        return batch
