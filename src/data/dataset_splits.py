import json
import os
import random
import warnings
from collections import defaultdict
from typing import Optional, Union, Dict, Set, List, Tuple
from copy import deepcopy

from datasets import DatasetDict, Dataset, load_from_disk
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario

from src.data.create_corpus_dataset import DatasetSplitType, create_dataset, DatasetConfig, UtteranceType, Query


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


def check_splits_for_contamination(train_split: DatasetDict, val_split: DatasetDict, test_split: DatasetDict) -> None:
    """
    Check if there is any overlap between the splits. If there is, the splits are contaminated.
    """
    train_ids = set(train_split["queries"]["id"])
    val_ids = set(val_split["queries"]["id"])
    test_ids = set(test_split["queries"]["id"])

    id_train_val_contamination = train_ids.intersection(val_ids)
    id_train_test_contamination = train_ids.intersection(test_ids)

    if len(id_train_val_contamination) > 0:
        raise ValueError(f"Train-Validation contamination: {len(id_train_val_contamination)} overlapping IDs.\n"
                         f"Example IDs: {list(id_train_val_contamination)}")

    if len(id_train_test_contamination) > 0:
        raise ValueError(f"Train-Test contamination: {len(id_train_test_contamination)} overlapping IDs.\n"
                         f"Example IDs: {list(id_train_test_contamination)}")

    train_texts = set(train_split["queries"]["text"])
    val_texts = set(val_split["queries"]["text"])
    test_texts = set(test_split["queries"]["text"])

    text_train_val_contamination = train_texts.intersection(val_texts)
    text_train_test_contamination = train_texts.intersection(test_texts)

    if len(text_train_val_contamination) > 0:
        warnings.warn(
            f"Overlapping texts between train and validation (but no overlapping query ids, so theses are not the same anchors): {len(text_train_val_contamination)}\n"
            f"Example texts: {list(text_train_val_contamination)}")

    if len(text_train_test_contamination) > 0:
        warnings.warn(
            f"Overlapping texts between train and test (but no overlapping query ids, so theses are not the same queries): {len(text_train_test_contamination)}\n"
            f"Example texts: {list(text_train_test_contamination)}")


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
    save_folder:
    dataset_save_name:
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
            splitted_dataset = load_splits_from_disk(save_path)
            check_splits_for_contamination(splitted_dataset["train"],
                                           splitted_dataset["validation"],
                                           splitted_dataset["test"])
            return splitted_dataset

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

    check_splits_for_contamination(splitted_dataset["train"], splitted_dataset["validation"], splitted_dataset["test"])

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


def create_out_of_distribution_hard_splits(corpus_dataset: DatasetDict,
                                           test_scenario: DiscussionSzenario) -> DatasetDict:
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

    def put_queries_forced_by_labels_into_split(queries: List[Query]) -> Tuple[Dict[DiscussionSzenario, Set[str]], List[Query], List[Query]]:
        """
        Creates a set of
        """
        forced_queries = []
        rest_queries = []
        covered_labels = defaultdict(set)  # scenario -> labels covered

        for query in queries:
            label_set = set(query.labels)
            if not label_set.issubset(covered_labels[query.discussion_scenario]):
                forced_queries.append(query)
                covered_labels[query.discussion_scenario].update(label_set)
            else:
                rest_queries.append(query)
        return covered_labels, forced_queries, rest_queries

    def check_label_coverage_of_split(queries: List[Query], split_covered_labels: Dict[DiscussionSzenario, Set[str]]) -> None:
        """
        Checks if the labels of a split cover all labels that are possible to ensure in-distribution testing
        """

        # get all labels that exist per scenario in the whole query split
        all_available_labels_per_scenario = defaultdict(set)
        for query in queries:
            label_set = set(query.labels)
            all_available_labels_per_scenario[query.discussion_scenario].update(label_set)

        # check for each scenario if the covered labels by a split completely contain
        for scenario, available_labels in all_available_labels_per_scenario.items():
            if available_labels != split_covered_labels[scenario]:
                missing_labels = split_covered_labels[scenario].difference(available_labels)
                raise ValueError(f"Not all labels are covered for scenario {scenario}.\n"
                                 f"These labels are missing: {missing_labels}")

    random.seed(seed)

    # 1) Shuffle anchors globally
    all_queries = [
        Query(
            id=entry["id"],
            text=entry["text"],
            labels=entry["labels"],
            discussion_scenario=entry["discussion_scenario"],
            context=entry["context"],
            scenario_description=entry["scenario_description"],
            scenario_question=entry["scenario_question"]
        )
        for entry in corpus_dataset["queries"]
    ]

    random.shuffle(all_queries)

    # Ensure all labels where an anchor is available are in the training set
    train_labels_covered, train_queries_forced, remaining_queries_after_train_selection = put_queries_forced_by_labels_into_split(all_queries)

    # check if all labels are covered in train
    check_label_coverage_of_split(all_queries, train_labels_covered)

    # all labels where still an anchor is available need to be in the test set to ensure in-distribution testing
    test_labels_covered, test_queries_forced, remaining_queries_after_test_selection = put_queries_forced_by_labels_into_split(remaining_queries_after_train_selection)

    # check if all remaining labels (left after train forced assignment) are covered
    check_label_coverage_of_split(remaining_queries_after_train_selection, test_labels_covered)


    # calculate split sizes. test size is the remaining part, after train and validation are selected.
    # since the data is shuffled, the selection from the remaining indices corresponds to sampling from the distribution of the dataset.
    dataset_size = len(all_queries)
    train_size = int(dataset_size * train_ratio)
    validation_size = int(dataset_size * val_ratio)
    test_size = dataset_size - (train_size + validation_size)

    train_forced_size = len(train_queries_forced)
    train_unforced_size = train_size - train_forced_size

    test_forced_size = len(test_queries_forced)
    test_unforced_size = test_size - test_forced_size

    train_queries_unforced = remaining_queries_after_test_selection[:train_unforced_size]
    train_queries = train_queries_forced + train_queries_unforced

    validation_queries = remaining_queries_after_test_selection[train_unforced_size: train_unforced_size + validation_size]
    test_queries_unforced = remaining_queries_after_test_selection[train_unforced_size + validation_size:]
    test_queries = test_queries_forced + test_queries_unforced

    train_query_ids = [query.id for query in train_queries]
    validation_query_ids = [query.id for query in validation_queries]
    test_query_ids = [query.id for query in test_queries]

    ds_train = create_datasetdict_for_query_ids(corpus_dataset, train_query_ids)
    ds_val = create_datasetdict_for_query_ids(corpus_dataset, validation_query_ids)
    ds_test = create_datasetdict_for_query_ids(corpus_dataset, test_query_ids)

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

    # add labels to unseen that appear with as unseen already selected labels to make sure all labels of multilabel queries remain unseen
    for anchor in anchors:
        scenario = anchor["discussion_scenario"]
        labels = set(anchor["labels"])
        print(labels)
        print(scenario)
        print(scenario_unseen_labels[scenario])
        print()
        if labels.intersection(scenario_unseen_labels[scenario]):
            scenario_unseen_labels[scenario].update(labels)

    # -- STEP C: Separate queries: unseen vs. seen
    #   "unseen" = queries that contain at least one unseen label for *their* scenario
    unseen_label_query_ids = []
    seen_label_query_ids = []
    for item in anchors:
        scn = item["discussion_scenario"]
        qid = item["id"]
        labs = set(item["labels"])
        # If all labels are unseen, it's an unseen query
        if labs.issubset(scenario_unseen_labels[scn]):
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

    dataset_folder = "../../data/processed/with_context"
    dataset_path = os.path.join(dataset_folder, "corpus_dataset_v2")

    if not os.path.exists(dataset_path):
        # Beispiel zum Erstellen eines Datensatzes. Mögliche Optionen von DatasetConfig sind im DocString beschrieben.
        create_dataset(
            DatasetConfig(
                dataset_path=dataset_path,
                project_dir="../../",
                utterance_type=UtteranceType.User,
                eval_size=0.5,
                validation_test_ratio=0.5
            )
        )

    # Beispiel zum Laden des Datensatzes + collate_function des DataLoaders um dynamisch ein Subset der negative passages zu laden.
    loaded_dataset = load_from_disk(dataset_path)
    # dataset_name = "dataset_split_in_distribution"
    # save_path = os.path.join(dataset_folder, dataset_name)
    # in_distribution_split = create_splits_from_corpus_dataset(corpus_dataset=loaded_dataset,
    #                                                           dataset_split_type=DatasetSplitType.InDistribution,
    #                                                           save_folder=dataset_folder,
    #                                                           dataset_save_name=dataset_name)

    # in_distribution_split_2 = create_splits_from_corpus_dataset(corpus_dataset=loaded_dataset,
    #                                                             dataset_split_type=DatasetSplitType.InDistribution,
    #                                                             save_folder=dataset_path,
    #                                                             dataset_save_name=dataset_name)

    # Create an Out-of-Distribution (Simple) split
    # ood_splits = create_splits_from_corpus_dataset(
    #     loaded_dataset,
    #     dataset_split_type=DatasetSplitType.OutOfDistributionSimple,
    #     save_folder=dataset_folder,
    #     dataset_save_name="dataset_split_out_of_distribution_simple",
    #     seed=420
    # )

    # print("Number of queries in train:", ood_splits["train"]["queries"].num_rows)
    # print("Number of queries in val:", ood_splits["validation"]["queries"].num_rows)
    # print("Number of queries in test:", ood_splits["test"]["queries"].num_rows)
    #
    # print("Done.")

    test_scenario = DiscussionSzenario.MEDAI
    split_by_scenario = create_splits_from_corpus_dataset(loaded_dataset,
                                                          DatasetSplitType.InDistribution,
                                                          save_folder=dataset_folder,
                                                          dataset_save_name=f"dataset_in_distribution",
                                                          test_scenario=test_scenario)
    # kfold_split = create_splits_from_corpus_dataset(hf_dataset, DatasetSplitType.kFold, None, 5)
    print("done")
