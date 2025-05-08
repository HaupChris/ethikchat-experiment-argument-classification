from collections import defaultdict

import pandas as pd
from datasets import DatasetDict


def get_split_key_count_per_label_for_corpus_dataset(split_dataset: DatasetDict, split_key: str = "queries", labels_key: str = "labels") -> pd.DataFrame:
    """
    Counts the number of queries accross all occurring (scenario, label) combinations in the split_dataset
    Args:
        split_dataset (): A dataset dict, containing "train", "validation", "test" and for each of them the types of texts, like
                            "queries", "passages"
        split_key (): the type of text to count the occurences, e.g. "queries" or "passages"
        labels_key (): "label" or "labels

    Returns:

    """
    # Count labels per scenario and split for queries
    label_counts = defaultdict(lambda: defaultdict(int))

    for split_key_val in split_dataset[split_key]:
        scenario = split_key_val["discussion_scenario"]
        if labels_key == "labels":
            for label in split_key_val[labels_key]:
                label_counts[scenario][label] += 1
        elif labels_key == "label":
            label_counts[scenario][split_key_val["label"]] += 1
        else:
            raise ValueError(f"labels_key '{labels_key}' not possible!")


    # Build dataframe
    records = []
    for scenario, labels in label_counts.items():
        for label, count in labels.items():
            records.append({
                "scenario": scenario,
                "label": label,
                "count": count
            })

    return pd.DataFrame(records)