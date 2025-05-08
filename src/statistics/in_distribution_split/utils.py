import copy
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from datasets import DatasetDict
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from src.features.build_features import filter_queries_for_few_shot_setting


def get_query_difference_between_num_shots(original_dataset: DatasetDict, num_shots_large: int, num_shots_small:int) -> List[Dict[str, Any]]:
    """
    filters a queries split for two few shot settings and shows the queries

    """
    ds_all = original_dataset.copy()
    ds_all["queries"] = copy.deepcopy(original_dataset["queries"])
    ds_all["passages"] = copy.deepcopy(original_dataset["passages"])
    ds_all["queries_relevant_passages_mapping"] = copy.deepcopy(original_dataset["queries_relevant_passages_mapping"])
    ds_all["queries_trivial_passages_mapping"] = copy.deepcopy(original_dataset["queries_trivial_passages_mapping"])


    # Create filtered datasets
    dataset_large = filter_queries_for_few_shot_setting(ds_all, num_shots_large, True)
    dataset_small = filter_queries_for_few_shot_setting(original_dataset, num_shots_small, True)

    # Extract query IDs from each
    query_ids_large = set(q["id"] for q in dataset_large["queries"])
    query_ids_small = set(q["id"] for q in dataset_small["queries"])

    # Compute the difference
    extra_ids = query_ids_large - query_ids_small

    # Extract full query dicts from the original dataset
    extra_queries = [q for q in ds_all["queries"] if q["id"] in extra_ids]

    return extra_queries


def get_passage_difference_between_num_shots(original_dataset: DatasetDict, num_shots_large: int, num_shots_small: int) -> List[Dict[str, Any]]:
    """
    Returns the passages that are included in the -1 setting but not in the 512 setting.
    """
    # Filtered datasets
    ds_all = original_dataset.copy()
    ds_all["queries"] = copy.deepcopy(original_dataset["queries"])
    ds_all["passages"] = copy.deepcopy(original_dataset["passages"])
    ds_all["queries_relevant_passages_mapping"] = copy.deepcopy(original_dataset["queries_relevant_passages_mapping"])
    ds_all["queries_trivial_passages_mapping"] = copy.deepcopy(original_dataset["queries_trivial_passages_mapping"])


    dataset_large = filter_queries_for_few_shot_setting(ds_all, num_shots_large, True)
    dataset_small = filter_queries_for_few_shot_setting(original_dataset, num_shots_small, True)

    # Extract passage IDs from each
    passage_ids_large = set(p["id"] for p in dataset_large["passages"])
    passage_ids_small = set(p["id"] for p in dataset_small["passages"])

    # Passages that are only in the -1 setting
    extra_ids = passage_ids_large - passage_ids_small

    # Extract full passage dicts from the dataset
    extra_passages = [p for p in ds_all["passages"] if p["id"] in extra_ids]

    return extra_passages


def plot_query_count_per_scenario_label(query_count_per_scenario_label: pd.DataFrame, save_as_eps:bool = False, save_dir: str = ".") -> None:
    """
    Creates a plot for each scenario with number of queries each label occurs. X-axis has label ids, y-axis the count.
    Scale is symlog, linear between 0 and 1 to see zero counts, logarithmic with base 2 at values larger than 1.
    Args:
        query_count_per_scenario_label ():
        save_as_eps ():
        save_dir ():

    Returns:

    """

    # Plot stacked bar chart per scenario
    scenarios = query_count_per_scenario_label["scenario"].unique()

    # --- for each scenario ---
    for scenario in scenarios:
        # pivot as before
        df_scenario = (
            query_count_per_scenario_label
            # @scenario refers to a python variable in the current scope
            .query("scenario == @scenario")
            .pivot(index="label", columns="split", values="count")
            .fillna(0)
            .sort_values("train", ascending=False)
        )

        # compute y-ticks: 0, 1, then powers of 2 up to max
        max_count = int(df_scenario.values.max())
        if max_count < 2:
            yticks = [0, 1]
        else:
            max_pow = int(np.floor(np.log2(max_count)))
            yticks = [0, 1] + [2 ** i for i in range(1, max_pow + 1)]
        yticks = sorted(set(yticks))

        fig, ax = plt.subplots(figsize=(12, 8))
        df_scenario.plot(
            kind="line",
            stacked=False,
            color=["blue", "yellow", "red"],
            ax=ax,
            linewidth=2,
        )

        # titles and labels
        ax.set_title(
            f"Query-label distribution in {scenario} (train / val / test)",
            fontsize=16,
        )
        ax.set_xlabel("Label", fontsize=16)
        ax.set_ylabel("Number of queries", fontsize=16)

        # xticks: one per label
        ax.set_xticks(range(len(df_scenario.index)))
        ax.set_xticklabels(df_scenario.index, rotation=90, ha="right", fontsize=16)

        # yscale symlog: linear 0-1, log₂ beyond, base=2
        ax.set_yscale("symlog", linthresh=1, base=2)
        ax.set_ylim(bottom=0)  # cut off negatives

        # custom y-ticks
        ax.set_yticks(yticks)
        ax.get_yaxis().set_major_formatter(ScalarFormatter())  # show raw numbers

        # all tick labels fontsize
        ax.tick_params(axis="y", labelsize=16)

        plt.tight_layout()
        plt.show()


def find_uncovered_labels(
        splits: Dict[str, Any],
        *,
        train_split: str = "train",
        val_split: str = "validation",
        test_split: str = "test",
        scenario_field: str = "discussion_scenario",
        labels_field: str = "labels",
) -> List[Dict[str, int]]:
    """
    Identify any (scenario, label) that appears in validation or test
    with 0 occurrences in train.

    Parameters
    ----------
    splits
        A DatasetDict-like mapping with keys train/validation/test, each
        having a 'queries' table of dict-like examples.
    train_split, val_split, test_split
        Names of the splits to inspect.
    scenario_field
        Name of the field holding the scenario in each query.
    labels_field
        Name of the field holding the list of labels in each query.

    Returns
    -------
    A list of dicts, each with keys:
      - 'scenario'
      - 'label'
      - 'validation'  : count in validation split
      - 'test'        : count in test split
      - 'train'       : count in train split (will be zero for all entries)
    """
    # count occurrences per split
    counts = defaultdict(lambda: defaultdict(lambda: {train_split: 0, val_split: 0, test_split: 0}))
    for split_name in (train_split, val_split, test_split):
        for q in splits[split_name]["queries"]:
            scen = q[scenario_field]
            for lab in q[labels_field]:
                counts[scen][lab][split_name] += 1

    # collect uncovered
    uncovered = []
    for scen, label_map in counts.items():
        for lab, cnts in label_map.items():
            if (cnts[val_split] > 0 or cnts[test_split] > 0) and cnts[train_split] == 0:
                uncovered.append({
                    "scenario": scen,
                    "label": lab,
                    train_split: cnts[train_split],
                    val_split: cnts[val_split],
                    test_split: cnts[test_split],
                })
    return uncovered


def get_max_counts_per_scenario(df: pd.DataFrame) -> dict:
    """
    Returns the maximum count value per scenario from a DataFrame with multiple splits and labels.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame with at least the columns ['scenario', 'count'].

    Returns:
    -------
    dict
        A dictionary mapping each scenario to the maximum value in the 'count' column.
        Format: {scenario: max_count}
    """
    return df.groupby('scenario')['count'].max().to_dict()


def get_split_key_count_per_scenario_label(split_dataset: DatasetDict, split_key: str = "queries", labels_key: str = "labels") -> pd.DataFrame:
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
    label_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    if hasattr(split_dataset, "keys"):
        splits = list(split_dataset.keys())
    else:
        splits = list(split_dataset.column_names)
    print(f"Counting '{split_key}' labels for splits '{splits}'")

    for split in splits:
        for split_key_val in split_dataset[split][split_key]:
            scenario = split_key_val["discussion_scenario"]
            if labels_key == "labels":
                for label in split_key_val[labels_key]:
                    label_counts[split][scenario][label] += 1
            elif labels_key == "label":
                label_counts[split][scenario][split_key_val["label"]] += 1
            else:
                raise ValueError(f"labels_key '{labels_key}' not possible!")

    # Build dataframe
    records = []
    for split, split_dict in label_counts.items():
        for scenario, labels in split_dict.items():
            for label, count in labels.items():
                records.append({
                    "scenario": scenario,
                    "label": label,
                    "split": split,
                    "count": count
                })

    return pd.DataFrame(records)
