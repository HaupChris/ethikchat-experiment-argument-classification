import os
import random
from typing import Optional, Union, Dict
from copy import deepcopy
from datasets import DatasetDict, Dataset, load_from_disk
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario

from src.data.create_corpus_dataset import DatasetSplitType, create_dataset, DatasetConfig
from src.data.make_dataset import UtteranceType


def create_splits_from_corpus_dataset(
        corpus_dataset: DatasetDict,
        dataset_split_type: DatasetSplitType,
        test_scenario: Optional[DiscussionSzenario] = None,
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
          - "queries": has columns ["id", "text", "discussion_scenario"]
          - "passages": has columns ["id", "text", "label", "discussion_scenario"]
          - "queries_relevant_passages_mapping": has columns ["query_id", "passages_ids"]
    dataset_split_type : DatasetSplitType
        The type of split to create: Simple, ByDiscussionSzenario, or kFold.
    test_scenario : Optional[DiscussionSzenario], optional
        Required if dataset_split_type == ByDiscussionSzenario;
        the scenario that should go into the test split.
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

    def create_datasetdict_for_query_ids(query_ids: list) -> DatasetDict:
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
        sub_mapping_relevants = subset_by_query_ids(corpus_dataset["queries_relevant_passages_mapping"], query_ids_set, True)

        # Subset queries_trivial_passages_mapping
        sub_mapping_trivial = subset_by_query_ids(corpus_dataset["queries_trivial_passages_mapping"], query_ids_set, True)

        return DatasetDict({
            "queries": sub_queries,
            "passages": sub_passages,
            "queries_relevant_passages_mapping": sub_mapping_relevants,
            "queries_trivial_passages_mapping": sub_mapping_trivial
        })

    # Extract the relevant columns from the corpus
    all_queries = corpus_dataset["queries"]  # HF Dataset for queries
    num_queries = all_queries.num_rows

    random.seed(seed)

    # ========== 1) Simple split: 80/10/10 ==========
    if dataset_split_type == DatasetSplitType.Simple:
        indices = list(range(num_queries))
        random.shuffle(indices)

        train_cut = int(0.8 * num_queries)
        val_cut = int(0.9 * num_queries)

        train_ids = [all_queries[i]["id"] for i in indices[:train_cut]]
        val_ids = [all_queries[i]["id"] for i in indices[train_cut:val_cut]]
        test_ids = [all_queries[i]["id"] for i in indices[val_cut:]]

        ds_train = create_datasetdict_for_query_ids(train_ids)
        ds_val = create_datasetdict_for_query_ids(val_ids)
        ds_test = create_datasetdict_for_query_ids(test_ids)

        return DatasetDict({
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test
        })

    # ========== 2) ByDiscussionSzenario ==========
    elif dataset_split_type == DatasetSplitType.ByDiscussionSzenario:
        if test_scenario is None:
            raise ValueError(
                "When using DatasetSplitType.ByDiscussionSzenario, you must provide `test_scenario`."
            )

        # Convert HF dataset to python list for easier filtering
        all_queries_list = all_queries.to_list()

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

        ds_train = create_datasetdict_for_query_ids(train_ids)
        ds_val = create_datasetdict_for_query_ids(val_ids)
        ds_test = create_datasetdict_for_query_ids(test_ids)

        return DatasetDict({
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test
        })

    # ========== 3) kFold cross-validation ==========
    elif dataset_split_type == DatasetSplitType.kFold:
        # Return: Dict[str, DatasetDict] with fold_i => { "train": ..., "test": ... }
        indices = list(range(num_queries))
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

            fold_train = create_datasetdict_for_query_ids(train_ids)
            fold_test = create_datasetdict_for_query_ids(test_ids)

            results[f"fold_{fold_idx}"] = DatasetDict({
                "train": fold_train,
                "test": fold_test
            })

        return results

    else:
        raise ValueError(f"Unknown dataset_split_type: {dataset_split_type}")


if __name__ == "__main__":

    dataset_path = "../../data/processed/corpus_dataset_experiment_v0"

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
    hf_dataset = load_from_disk(dataset_path)
    simple_split = create_splits_from_corpus_dataset(hf_dataset, DatasetSplitType.Simple)
    split_by_scenario = create_splits_from_corpus_dataset(hf_dataset, DatasetSplitType.ByDiscussionSzenario, test_scenario=DiscussionSzenario.JURAI)
    kfold_split = create_splits_from_corpus_dataset(hf_dataset, DatasetSplitType.kFold, None, 5)
    print("done")



