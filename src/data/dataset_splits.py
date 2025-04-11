import json
import os
import random
import warnings
from collections import defaultdict
from typing import Optional, Union, Dict, List, Tuple, Set, Any
from copy import deepcopy

import pandas as pd
from datasets import DatasetDict, Dataset, load_from_disk
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario
from sklearn.model_selection import train_test_split

from src.data.create_corpus_dataset import DatasetSplitType, create_dataset, DatasetConfig, UtteranceType, Query, \
    Passage


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
    # if dataset already exists, load it and return it. Otherwise, create it.
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
        splitted_dataset = create_out_of_distribution_label_split(
            corpus_dataset,
            heldout_label_fraction=0.3,
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


def compute_label_frequencies(queries: List[Query]) -> Dict[str, Dict[str, int]]:
    frequencies = defaultdict(lambda: defaultdict(int))
    for query in queries:
        for label in query.labels:
            frequencies[query.discussion_scenario][label] += 1
    return frequencies


def bucketize(items: List[Any], num_buckets: int) -> List[List[Any]]:
    k, r = divmod(len(items), num_buckets)
    return [items[i * k + min(i, r):(i + 1) * k + min(i + 1, r)] for i in range(num_buckets)]


def select_heldout_labels(label_frequencies: Dict[str, Dict[str, int]],
                          fraction: float,
                          num_buckets: int) -> set[Tuple[str, str]]:
    heldout_labels = set()
    scenario_buckets = {}
    for scenario, label_freqs in label_frequencies.items():
        sorted_labels = sorted(label_freqs.items(), key=lambda x: x[1], reverse=True)
        buckets = bucketize(sorted_labels, num_buckets)
        scenario_buckets[scenario] = buckets
        for bucket in buckets:
            random.shuffle(bucket)
            num_to_select = max(1, int(len(bucket) * fraction))
            heldout_labels.update([(scenario, label) for label, _ in bucket[:num_to_select]])
    return heldout_labels, scenario_buckets


def split_queries_by_label_presence(queries: List[Query],
                                    heldout_labels: set[Tuple[str, str]]) -> Tuple[List[Query], List[Query]]:
    with_heldout, with_only_train = [], []
    for query in queries:
        if any((query.discussion_scenario, label) in heldout_labels for label in query.labels):
            with_heldout.append(query)
        else:
            with_only_train.append(query)
    return with_heldout, with_only_train


def extract_primary_heldout_labels(queries: List[Query],
                                   heldout_labels: set[Tuple[str, str]]) -> Tuple[List[Query], List[Tuple[str, str]]]:
    filtered_queries = []
    primary_labels = []
    for query in queries:
        for label in query.labels:
            key = (query.discussion_scenario, label)
            if key in heldout_labels:
                primary_labels.append(key)
                filtered_queries.append(query)
                break
    return filtered_queries, primary_labels


def stratified_split(queries: List[Query],
                     labels: List[Tuple[str, str]],
                     seed: int) -> Tuple[List[Query], List[Query]]:
    label_support = defaultdict(int)
    for label in labels:
        label_support[label] += 1

    stratifiable_labels = {label for label, count in label_support.items() if count >= 2}

    strat_queries, strat_labels, non_strat_queries = [], [], []
    for query, label in zip(queries, labels):
        if label in stratifiable_labels:
            strat_queries.append(query)
            strat_labels.append(label)
        else:
            non_strat_queries.append(query)

    val, test = train_test_split(
        strat_queries, test_size=0.67, random_state=seed, stratify=strat_labels
    )

    random.shuffle(non_strat_queries)
    split_point = int(len(non_strat_queries) * 0.33)
    val += non_strat_queries[:split_point]
    test += non_strat_queries[split_point:]

    return val, test


def create_out_of_distribution_label_split(corpus_dataset: DatasetDict,
                                           heldout_label_fraction: float = 0.3,
                                           seed: int = 42) -> DatasetDict:
    random.seed(seed)
    num_buckets = 5

    all_queries = [
        Query(
            id=entry["id"],
            text=entry["text"],
            labels=entry["labels"],
            discussion_scenario=entry["discussion_scenario"],
            context=entry["context"],
            scenario_description=entry["scenario_description"],
            scenario_question=entry["scenario_question"]
        ) for entry in corpus_dataset["queries"]
    ]

    label_frequencies = compute_label_frequencies(all_queries)
    heldout_labels, scenario_buckets = select_heldout_labels(label_frequencies, heldout_label_fraction, num_buckets)

    queries_with_heldout, queries_with_only_train = split_queries_by_label_presence(all_queries, heldout_labels)
    selected_heldout_queries, primary_labels = extract_primary_heldout_labels(queries_with_heldout, heldout_labels)

    queries_validation, queries_test = stratified_split(selected_heldout_queries, primary_labels, seed)
    queries_train = queries_with_only_train

    ds_train = create_datasetdict_for_query_ids(corpus_dataset, [query.id for query in queries_train])
    ds_validation = create_datasetdict_for_query_ids(corpus_dataset, [query.id for query in queries_validation])
    ds_test = create_datasetdict_for_query_ids(corpus_dataset, [query.id for query in queries_test])

    ds_validation = clean_passages_mappings(ds_validation, heldout_labels)
    ds_test = clean_passages_mappings(ds_test, heldout_labels)

    rows = []
    for scenario, buckets in scenario_buckets.items():
        for bucket_id, bucket in enumerate(buckets):
            for label, freq in bucket:
                rows.append([scenario, bucket_id, label, freq])

    return DatasetDict({
        "train": ds_train,
        "validation": ds_validation,
        "test": ds_test,
        "buckets": Dataset.from_pandas(pd.DataFrame(rows, columns=["Scenario", "bucket", "label", "freq"]))
    })


def clean_passages_mappings(corpus_dataset: DatasetDict, held_out_labels: List[Tuple[str]]) -> DatasetDict:
    queries = [
        Query(
            id=entry["id"],
            text=entry["text"],
            labels=entry["labels"],
            discussion_scenario=entry["discussion_scenario"],
            context=entry["context"],
            scenario_description=entry["scenario_description"],
            scenario_question=entry["scenario_question"]
        ) for entry in corpus_dataset["queries"]
    ]

    passages = [Passage(
        id = entry["id"],
        text = entry["text"],
        label = entry["label"],
        discussion_scenario = entry["discussion_scenario"],
        passage_source = entry["passage_source"],
        retrieved_query_id = entry["retrieved_query_id"]
    ) for entry in corpus_dataset["passages"]]

    relevant_mapping = {
        entry["query_id"]: entry["passages_ids"] for entry in  corpus_dataset["queries_relevant_passages_mapping"]
    }
    trivial_mapping = {
        entry["query_id"]: entry["passages_ids"] for entry in corpus_dataset["queries_trivial_passages_mapping"]
    }

    for query in queries:
        relevant_passages = [passage for passage in passages if passage.id in relevant_mapping[query.id]]
        trivial_passages = [passage for passage in passages if passage.id in trivial_mapping[query.id]]
        for label in query.labels:
            if (query.discussion_scenario, label) not in held_out_labels:
                # remove relevant passages with non held out labels
                print(len(relevant_passages))
                trivial_passages.extend([passage for passage in relevant_passages if passage.label == label])
                relevant_passages = [passage for passage in relevant_passages if passage.label != label]
                print(len(relevant_passages))
                # add trivial passages with held out labels
        relevant_mapping[query.id] = [passage.id for passage in relevant_passages]
        trivial_mapping[query.id] = [passage.id for passage in trivial_passages]

    return DatasetDict({
        **corpus_dataset,
        "queries_relevant_passages_mapping": Dataset.from_dict({
            "query_id": [idx for idx, _ in relevant_mapping.items()],
            "passages_ids": [ids for _, ids in relevant_mapping.items()]
        }),
        "queries_trivial_passages_mapping": Dataset.from_dict({
            "query_id": [idx for idx, _ in trivial_mapping.items()],
            "passages_ids": [ids for _, ids in trivial_mapping.items()]
        }),
    })



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
    test_val_queries = [q for q in all_queries_list if q["discussion_scenario"] == test_scenario.value]
    # train+val: the rest
    train_queries = [q for q in all_queries_list if q["discussion_scenario"] != test_scenario.value]
    random.shuffle(test_val_queries)
    test_validation_cut = int(0.5 * len(test_val_queries))

    test_queries = test_val_queries[:test_validation_cut]
    val_queries = test_val_queries[test_validation_cut:]

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

    def put_queries_forced_by_labels_into_split(queries: List[Query]) -> Tuple[
        Dict[DiscussionSzenario, Set[str]], List[Query], List[Query]]:
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

    def check_label_coverage_of_split(queries: List[Query],
                                      split_covered_labels: Dict[DiscussionSzenario, Set[str]]) -> None:
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
    train_labels_covered, train_queries_forced, remaining_queries_after_train_selection = put_queries_forced_by_labels_into_split(
        all_queries)

    # check if all labels are covered in train
    check_label_coverage_of_split(all_queries, train_labels_covered)

    # all labels where still an anchor is available need to be in the test set to ensure in-distribution testing
    test_labels_covered, test_queries_forced, remaining_queries_after_test_selection = put_queries_forced_by_labels_into_split(
        remaining_queries_after_train_selection)

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

    validation_queries = remaining_queries_after_test_selection[
                         train_unforced_size: train_unforced_size + validation_size]
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
                                                          DatasetSplitType.OutOfDistributionSimple,
                                                          save_folder=dataset_folder,
                                                          dataset_save_name=f"dataset_out_of_distribution_label",
                                                          test_scenario=test_scenario)

    # create_out_of_distribution_label_split(loaded_dataset, 0.15, 0.7, 42)
    # kfold_split = create_splits_from_corpus_dataset(hf_dataset, DatasetSplitType.kFold, None, 5)
    print("done")
