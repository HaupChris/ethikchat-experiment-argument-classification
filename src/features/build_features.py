import random

from datasets import Dataset, DatasetDict, load_from_disk
from typing import Optional


def build_contrastive_dataset(hf_dataset: DatasetDict, max_ratio_negatives_to_positives: Optional[int] = None) -> DatasetDict:
    """
    Transform your dataset from:
      {
        "utterance": str,
        "context": str,
        "positive_passages": List[str],
        "negative_passages": List[str]
      }
    into:
      {
        "anchor": str,
        "passage": str,
        "label": int  # 1 = positive, 0 = negative
      }

    :param hf_dataset: A DatasetDict with splits (train, validation, test).
    :param max_ratio_negatives_to_positives: The maximum ratio of negative to positives examples to keep. Since the negatives
        are usually much more than positives, we'll randomly sample a number of negatives depending on this ratio to the positives.
    :return: A new DatasetDict in contrastive pair format.
    """
    def to_contrastive_rows(example):
        # We'll store expanded data in lists
        anchors = []
        passages = []
        labels = []

        # Anchor is always the utterance
        anchor_text = example["utterance"]

        num_positives = len(example["positive_passages"])
        if max_ratio_negatives_to_positives is not None:
            num_negatives = min(len(example["negative_passages"]), max_ratio_negatives_to_positives * num_positives)
        else:
            num_negatives = len(example["negative_passages"])

        # Expand positives
        for pos_passage in example["positive_passages"]:
            anchors.append(anchor_text)
            passages.append(pos_passage)
            labels.append(1)

        # Expand negatives
        negative_passages = random.sample(example["negative_passages"], num_negatives)
        for neg_passage in negative_passages:
            anchors.append(anchor_text)
            passages.append(neg_passage)
            labels.append(0)

        return {"anchor": anchors, "passage": passages, "label": labels}

    # Convert each split
    new_splits = {}
    for split_name in ["train", "validation", "test"]:
        # "map" will transform each example in the split into multiple examples
        new_dataset = hf_dataset[split_name].map(
            to_contrastive_rows,
            batched=True,
            remove_columns=hf_dataset[split_name].column_names
        )
        new_splits[split_name] = new_dataset

    return DatasetDict(new_splits)



if __name__ =="__main__":
    hf_dataset = load_from_disk("../data/dummy_dataset")
    contrastive_dataset = build_contrastive_dataset(hf_dataset, max_ratio_negatives_to_positives=None)
    print(contrastive_dataset)