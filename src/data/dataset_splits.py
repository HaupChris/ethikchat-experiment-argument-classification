import json
import os
import random
from collections import defaultdict
from typing import Optional, Union, Dict
from copy import deepcopy

import datasets
from datasets import DatasetDict, Dataset, load_from_disk
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario

from src.data.create_corpus_dataset import DatasetSplitType, create_dataset, DatasetConfig, UtteranceType


# --- Helper Functions ---
def subset_by_query_ids(
        dataset: Dataset,
        valid_query_ids: set,
        is_mapping: bool = False
) -> Dataset:
    """
    Filters a dataset of queries or queries_relevant_passages_mapping to only include the specified query IDs.

    If `is_mapping=True`, filters by "query_id" in that column.
    Otherwise, filters by "id" in the "queries" dataset.

    Passages are left as-is (we do not filter them further by default).
    """
    if is_mapping:
        return dataset.filter(lambda x: x["query_id"] in valid_query_ids)
    else:
        # "queries" case
        return dataset.filter(lambda x: x["id"] in valid_query_ids)


def create_datasetdict_for_query_ids(corpus_dataset: DatasetDict, query_ids: list) -> DatasetDict:
    """
    Given a list of query IDs, create a DatasetDict containing:
      - "queries" subset
      - "passages" (unfiltered — keep them all)
      - "queries_relevant_passages_mapping" subset for only those query IDs
    """
    query_ids_set = set(query_ids)

    # Subset queries
    sub_queries = subset_by_query_ids(corpus_dataset["queries"], query_ids_set, is_mapping=False)

    # Keep all passages (standard IR scenario). If you want to keep only scenario-matching passages,
    # adapt this line:
    sub_passages = deepcopy(corpus_dataset["passages"])

    # Subset queries_relevant_passages_mapping
    sub_mapping_relevants = subset_by_query_ids(corpus_dataset["queries_relevant_passages_mapping"], query_ids_set,
                                                True)

    # Subset queries_trivial_passages_mapping
    sub_mapping_trivial = subset_by_query_ids(corpus_dataset["queries_trivial_passages_mapping"], query_ids_set, True)

    return DatasetDict({
        "queries": sub_queries,
        "passages": sub_passages,
        "queries_relevant_passages_mapping": sub_mapping_relevants,
        "queries_trivial_passages_mapping": sub_mapping_trivial
    })


def load_splits_from_disk(save_path) -> DatasetDict:
    # get splits from dataset_dict.json
    json_path = os.path.join(save_path, "dataset_dict.json")
    with open(json_path, "r") as f:
        dataset_dict = json.load(f)
    # load datasets
    dataset_dict = {k: load_from_disk(os.path.join(save_path, k)) for k in dataset_dict["splits"]}
    return DatasetDict(dataset_dict)


def create_splits_from_corpus_dataset(
        corpus_dataset: DatasetDict,
        dataset_split_type: DatasetSplitType,
        test_scenario: Optional[DiscussionSzenario] = None,
        save_folder: Optional[str] = None,
        dataset_save_name: Optional[str] = None,
        k: int = 5,
        seed: int = 42
) -> Union[DatasetDict, Dict[str, DatasetDict]]:
    """
    Splits a corpus_dataset (with splits "queries", "passages", "queries_relevant_passages_mapping")
    according to the chosen DatasetSplitType.

    Parameters
    ----------
    corpus_dataset : DatasetDict
        A huggingface DatasetDict with:
          - "queries": has columns ["id", "text", "labels", "discussion_scenario"]
          - "passages": has columns ["id", "text", "label", "discussion_scenario"]
          - "queries_relevant_passages_mapping": has columns ["query_id", "passages_ids"]
          - "queries_trivial_passages_mapping": has columns ["query_id", "passages_ids"]
    dataset_split_type : DatasetSplitType
        The type of split to create: InDistribution, OutOfDistributionSimple, OutOfDistributionHard.
    test_scenario : Optional[DiscussionSzenario], optional
        Required if dataset_split_type == ByDiscussionSzenario;
        the scenario that should go into the test split.
    save_path : Optional[str], if provided, saves the splits to this path.
    k : int, optional
        Number of folds for kFold splitting (default=5).
    seed : int, optional
        Random seed for reproducible shuffles.

    Returns
    -------
    Union[DatasetDict, Dict[str, DatasetDict]]
        - If Simple or ByDiscussionSzenario: returns a DatasetDict with keys ["train", "validation", "test"].
        - If kFold: returns a dictionary of k folds,
          each fold is a DatasetDict with keys ["train", "test"].
    """
    # if dataset already exists, load it and return it. Otherwise create it.
    save_path = None
    if save_folder and dataset_save_name:
        save_path = os.path.join(save_folder, dataset_save_name)
        if os.path.exists(save_path):
            print(f"Dataset already exists at {save_path}. Loading it.")
            return load_splits_from_disk(save_path)

    if dataset_split_type == DatasetSplitType.InDistribution:
        splitted_dataset = create_in_distribution_splits(corpus_dataset,
                                                         train_ratio=0.70,
                                                         val_ratio=0.15,
                                                         seed=seed)

    elif dataset_split_type == DatasetSplitType.OutOfDistributionSimple:
        splitted_dataset = create_out_of_distribution_simple_splits(
            corpus_dataset,
            fraction_unseen=0.15,
            train_ratio=0.70,
            seed=seed
        )

    elif dataset_split_type == DatasetSplitType.OutOfDistributionHard:
        if test_scenario is None:
            raise ValueError(
                "When using DatasetSplitType.ByDiscussionSzenario, you must provide `test_scenario`."
            )
        splitted_dataset = create_out_of_distribution_hard_splits(corpus_dataset, test_scenario)

    elif dataset_split_type == DatasetSplitType.kFold:
        # Return: Dict[str, DatasetDict] with fold_i => { "train": ..., "test": ... }
        splitted_dataset = create_k_fold_splits(corpus_dataset, k)
    else:
        raise ValueError(f"Unknown dataset_split_type: {dataset_split_type}")

    if save_path:
        splitted_dataset.save_to_disk(save_path)

    return splitted_dataset


def create_k_fold_splits(corpus_dataset: DatasetDict, k: int) -> Dict[str, DatasetDict]:
    indices = list(range(len(corpus_dataset["queries"])))
    num_queries = len(indices)
    all_queries = corpus_dataset["queries"].to_list()
    random.shuffle(indices)
    fold_size = num_queries // k
    results = {}
    for fold_idx in range(k):
        # Test fold
        start = fold_idx * fold_size
        end = start + fold_size if fold_idx < k - 1 else num_queries

        test_indices_fold = indices[start:end]
        train_indices_fold = indices[:start] + indices[end:]

        test_ids = [all_queries[i]["id"] for i in test_indices_fold]
        train_ids = [all_queries[i]["id"] for i in train_indices_fold]

        fold_train = create_datasetdict_for_query_ids(corpus_dataset, train_ids)
        fold_test = create_datasetdict_for_query_ids(corpus_dataset, test_ids)

        results[f"fold_{fold_idx}"] = DatasetDict({
            "train": fold_train,
            "test": fold_test
        })
    return results


def create_out_of_distribution_hard_splits(corpus_dataset: DatasetDict, test_scenario: DiscussionSzenario) -> DatasetDict:
    # Convert HF dataset to python list for easier filtering
    all_queries_list = corpus_dataset["queries"].to_list()
    # test split: queries with the given scenario
    test_queries = [q for q in all_queries_list if q["discussion_scenario"] == test_scenario.value]
    # train+val: the rest
    train_val_queries = [q for q in all_queries_list if q["discussion_scenario"] != test_scenario.value]
    # Now do an 80:20 split on the train_val queries for train/validation
    random.shuffle(train_val_queries)
    tv_cut = int(0.8 * len(train_val_queries))
    train_queries = train_val_queries[:tv_cut]
    val_queries = train_val_queries[tv_cut:]
    train_ids = [q["id"] for q in train_queries]
    val_ids = [q["id"] for q in val_queries]
    test_ids = [q["id"] for q in test_queries]
    ds_train = create_datasetdict_for_query_ids(corpus_dataset, train_ids)
    ds_val = create_datasetdict_for_query_ids(corpus_dataset, val_ids)
    ds_test = create_datasetdict_for_query_ids(corpus_dataset, test_ids)
    hf_dataset = DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test
    })
    return hf_dataset


def create_in_distribution_splits(corpus_dataset: DatasetDict,
                                  train_ratio=0.7,
                                  val_ratio=0.15,
                                  seed=42) -> DatasetDict:
    """
    Splits anchors into Train/Eval/Test, ensuring every label in the test or validation set is seen in training.

    Parameters
    ----------
    corpus_dataset : DatasetDict, with keys ["queries", "passages", "queries_relevant_passages_mapping", "queries_trivial_passages_mapping"]
    train_ratio : float, ratio of training data from the whole dataset
    val_ratio : float, ratio of validation data from the whole dataset, (1 - train_ratio - val_ratio) is the test ratio
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DatasetDict, with keys ["train", "validation", "test"]
    """
    random.seed(seed)

    # 1) Shuffle anchors globally
    indices = list(range(len(corpus_dataset["queries"])))
    anchor_label_sets = [set(q["labels"]) for q in corpus_dataset["queries"]]
    random.shuffle(indices)

    # 2) For each label, ensure coverage in train
    train_indices_forced = set()

    # Map label -> all anchor indices that contain that label
    label2anchor = defaultdict(list)
    for i in indices:
        for lab in anchor_label_sets[i]:
            label2anchor[lab].append(i)

    # For each label, pick an anchor to ensure coverage in training
    for lab, anchor_list in label2anchor.items():
        chosen_idx = anchor_list[0]  # pick the first, or pick randomly
        train_indices_forced.add(chosen_idx)

    # 3) Remove forced anchors from the main pool
    remaining_indices = [i for i in indices if i not in train_indices_forced]

    # 4) Split the REMAINING anchors by ratio
    total_num_indices = len(indices)
    num_train_indices = int(train_ratio * total_num_indices)
    num_val_indices = int(val_ratio * total_num_indices)

    num_unforced_train_indices = num_train_indices - len(train_indices_forced)

    train_part = remaining_indices[:num_unforced_train_indices]
    val_indices = remaining_indices[num_unforced_train_indices:num_unforced_train_indices + num_val_indices]
    test_indices = remaining_indices[num_unforced_train_indices + num_val_indices:]

    # Union forced-train with newly assigned train
    train_indices_final = list(train_indices_forced.union(set(train_part)))

    ds_train = create_datasetdict_for_query_ids(corpus_dataset, train_indices_final)
    ds_val = create_datasetdict_for_query_ids(corpus_dataset, val_indices)
    ds_test = create_datasetdict_for_query_ids(corpus_dataset, test_indices)

    return DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test
    })

def create_out_of_distribution_simple_splits(
        corpus_dataset: DatasetDict,
        fraction_unseen: float = 0.15,
        train_ratio: float = 0.70,
        seed: int = 42
) -> DatasetDict:
    """
    Creates an Out-of-Distribution (Simple) split by withholding a subset of labels
    (chosen from top, middle, and bottom frequency ranges) entirely from training.

    1) Determine label frequencies over all queries.
    2) Sort labels by frequency descending.
    3) Partition into top-third, middle-third, bottom-third.
    4) Randomly pick some fraction of labels from each partition to be "unseen".
    5) All queries with those unseen labels go into the final test set.
    6) Remaining queries (with only "seen" labels) are split 80:20.

    Parameters
    ----------
    corpus_dataset : DatasetDict, with keys ["queries", "passages", "queries_relevant_passages_mapping", "queries_trivial_passages_mapping"]
    fraction_unseen : float, optional
        Fraction of labels (in each frequency partition) to mark as unseen (default=0.20).
    train_ratio : float, optional
        Train portion for the queries (default=0.70).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DatasetDict
      With three splits: "train", "validation", "test"
    """
    random.seed(seed)

    # -- STEP A: Gather all queries + their labels
    all_queries = corpus_dataset["queries"]  # huggingface Dataset
    anchors = all_queries.to_list()  # convert to python list for easier manip

    # anchors[i]["labels"] is the list of labels for the i-th query
    # anchors[i]["id"] is the query ID

    # -- STEP B: Count frequencies for each label for each scenario
    scenario_label_counts = {"MEDAI": {}, "AUTOAI": {}, "JURAI": {}, "REFAI": {}}
    for item in anchors:
        for lab in item["labels"]:
            if lab not in scenario_label_counts[item["discussion_scenario"]]:
                scenario_label_counts[item["discussion_scenario"]][lab] = 1
            else:
                scenario_label_counts[item["discussion_scenario"]][lab] += 1

    scenario_sorted_labels = {}
    num_labels_per_scenario = {}
    for scn, label_counts in scenario_label_counts.items():
        sorted_labels = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)
        scenario_sorted_labels[scn] = sorted_labels
        num_labels_per_scenario[scn] = len(sorted_labels)

    for scn, count in num_labels_per_scenario.items():
        if count < 3:
            raise ValueError(f"Scenario '{scn}' has fewer than 3 labels, cannot partition top/middle/bottom.")

    # --- Helper function to pick fraction_unseen from each partition
    def pick_unseen_from_partition(labels_partition):
        # e.g. 20% of that partition's labels
        k = int(len(labels_partition) * fraction_unseen)
        random.shuffle(labels_partition)
        return set(labels_partition[:k])

    # -- STEP B: For each scenario, partition & pick unseen labels
    scenario_unseen_labels = {"MEDAI": set(), "AUTOAI": set(), "JURAI": set(), "REFAI": set()}

    for scn, sorted_labels in scenario_sorted_labels.items():
        num_labels = len(sorted_labels)
        one_third = num_labels // 3

        top_third = sorted_labels[:one_third]
        mid_third = sorted_labels[one_third:2 * one_third]
        bot_third = sorted_labels[2 * one_third:]  # possibly bigger if not multiple of 3

        unseen_top = pick_unseen_from_partition(top_third)
        unseen_mid = pick_unseen_from_partition(mid_third)
        unseen_bot = pick_unseen_from_partition(bot_third)

        scenario_unseen_labels[scn] = unseen_top.union(unseen_mid).union(unseen_bot)

    # -- STEP C: Separate queries: unseen vs. seen
    #   "unseen" = queries that contain at least one unseen label for *their* scenario
    unseen_label_query_ids = []
    seen_label_query_ids = []
    for item in anchors:
        scn = item["discussion_scenario"]
        qid = item["id"]
        labs = set(item["labels"])
        # If labs intersect scenario_unseen_labels[scn], it's an unseen query
        if labs.intersection(scenario_unseen_labels[scn]):
            unseen_label_query_ids.append(qid)
        else:
            seen_label_query_ids.append(qid)

    # -- STEP D: The "unseen" queries go into test. The "seen" queries get 70:15:15
    random.shuffle(seen_label_query_ids)
    total_seen = len(seen_label_query_ids)

    test_size = len(unseen_label_query_ids)

    # determine if the test_size is roughly 20% of the total dataset
    if test_size < (fraction_unseen * 0.8) * (total_seen + test_size):
        raise ValueError(
            f"Test set should be at least {(fraction_unseen * 0.8)} of the total dataset. but is {test_size / (total_seen + test_size)}. Choose a different seed.")
    elif test_size > (fraction_unseen * 1.2) * (total_seen + test_size):
        raise ValueError(
            f"Test set should be at most {(fraction_unseen * 1.2)} of the total dataset. but is {test_size / (total_seen + test_size)}. Choose a different seed.")

    seen_fraction = total_seen / (total_seen + test_size)
    train_fraction_of_seen = train_ratio / seen_fraction
    n_train = int(train_fraction_of_seen * total_seen)

    train_ids = seen_label_query_ids[:n_train]
    val_ids = seen_label_query_ids[n_train:]
    test_ids = unseen_label_query_ids

    # check if all data has been used
    if len(train_ids) + len(val_ids) + test_size != total_seen + test_size:
        raise ValueError("Not all data has been used. Choose a different seed.")

    # -- STEP E: Build final splits with your existing helper
    ds_train = create_datasetdict_for_query_ids(corpus_dataset, train_ids)
    ds_val = create_datasetdict_for_query_ids(corpus_dataset, val_ids)
    ds_test = create_datasetdict_for_query_ids(corpus_dataset, test_ids)

    # Return a DatasetDict with top-level "train", "validation", "test"
    return DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test
    })


if __name__ == "__main__":

    dataset_folder = "../../data/processed/"
    dataset_path = os.path.join(dataset_folder, "corpus_dataset_v1")

    if not os.path.exists(dataset_path):
        # Beispiel zum Erstellen eines Datensatzes. Mögliche Optionen von DatasetConfig sind im DocString beschrieben.
        create_dataset(
            DatasetConfig(
                dataset_path=dataset_path,
                project_dir="../../",
                num_previous_turns=3,
                include_role=True,
                sep_token="[SEP]",
                utterance_type=UtteranceType.User,
                eval_size=0.5,
                validation_test_ratio=0.5
            )
        )

    # Beispiel zum Laden des Datensatzes + collate_function des DataLoaders um dynamisch ein Subset der negative passages zu laden.
    loaded_dataset = load_from_disk(dataset_path)
    dataset_name = "dataset_split_in_distribution"
    save_path = os.path.join(dataset_folder, dataset_name)
    in_distribution_split = create_splits_from_corpus_dataset(corpus_dataset=loaded_dataset,
                                                              dataset_split_type=DatasetSplitType.InDistribution,
                                                              save_folder=dataset_folder,
                                                              dataset_save_name=dataset_name)

    # Create an Out-of-Distribution (Simple) split
    ood_splits = create_splits_from_corpus_dataset(
        loaded_dataset,
        dataset_split_type=DatasetSplitType.OutOfDistributionSimple,
        save_folder=dataset_folder,
        dataset_save_name="dataset_split_out_of_distribution_simple",
        seed=420
    )

    print("Number of queries in train:", ood_splits["train"]["queries"].num_rows)
    print("Number of queries in val:", ood_splits["validation"]["queries"].num_rows)
    print("Number of queries in test:", ood_splits["test"]["queries"].num_rows)

    print("Done.")

    test_scenario = DiscussionSzenario.JURAI
    split_by_scenario = create_splits_from_corpus_dataset(loaded_dataset,
                                                          DatasetSplitType.OutOfDistributionHard,
                                                          save_folder=dataset_folder,
                                                          dataset_save_name=f"dataset_split_by_scenario_{test_scenario}",
                                                          test_scenario=test_scenario)
    # kfold_split = create_splits_from_corpus_dataset(hf_dataset, DatasetSplitType.kFold, None, 5)
    print("done")
