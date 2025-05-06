import json
import os
import warnings
from copy import deepcopy

from datasets import Dataset, DatasetDict, load_from_disk


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
