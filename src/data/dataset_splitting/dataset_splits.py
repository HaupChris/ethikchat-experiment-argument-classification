import os
import random

from typing import Optional, Union, Dict

from datasets import DatasetDict, load_from_disk
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario

from src.data.create_corpus_dataset import create_dataset
from src.data.classes import UtteranceType, DatasetConfig, DatasetSplitType
from src.data.dataset_splitting.in_distribution_splitting import dialogue_level_in_distribution_split
from src.data.dataset_splitting.out_of_distribution_label_splitting import create_out_of_distribution_label_split
from src.data.dataset_splitting.out_of_distribution_topic_splitting import create_out_of_distribution_hard_splits
from src.data.dataset_splitting.utils import create_datasetdict_for_query_ids, load_splits_from_disk, \
    check_splits_for_contamination


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
        splitted_dataset = dialogue_level_in_distribution_split(corpus_dataset,
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



if __name__ == "__main__":

    dataset_folder = "../../../data/processed/with_context"
    dataset_path = os.path.join(dataset_folder, "corpus_dataset_v3")

    if not os.path.exists(dataset_path):
        # Beispiel zum Erstellen eines Datensatzes. Mögliche Optionen von DatasetConfig sind im DocString beschrieben.
        create_dataset(
            DatasetConfig(
                dataset_path=dataset_path,
                project_dir="../../../",
                utterance_type=UtteranceType.User,
                eval_size=0.5,
                validation_test_ratio=0.5
            )
        )

    # Beispiel zum Laden des Datensatzes + collate_function des DataLoaders um dynamisch ein Subset der negative passages zu laden.
    loaded_dataset = load_from_disk(dataset_path)
    dataset_name = "dataset_split_in_distribution_from_v3"
    save_path = os.path.join(dataset_folder, dataset_name)
    in_distribution_split = create_splits_from_corpus_dataset(corpus_dataset=loaded_dataset,
                                                              dataset_split_type=DatasetSplitType.InDistribution,
                                                              save_folder=dataset_folder,
                                                              dataset_save_name=dataset_name)

    # in_distribution_split_2 = create_splits_from_corpus_dataset(corpus_dataset=loaded_dataset,
    #                                                             dataset_split_type=DatasetSplitType.InDistribution,
    #                                                             save_folder=dataset_path,
    #                                                             dataset_save_name="dataset_split_in_distribution")
    print()
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

    # test_scenario = DiscussionSzenario.MEDAI
    # split_by_scenario = create_splits_from_corpus_dataset(loaded_dataset,
    #                                                       DatasetSplitType.OutOfDistributionSimple,
    #                                                       save_folder=dataset_folder,
    #                                                       dataset_save_name=f"dataset_out_of_distribution_label",
    #                                                       test_scenario=test_scenario)

    # create_out_of_distribution_label_split(loaded_dataset, 0.15, 0.7, 42)
    # kfold_split = create_splits_from_corpus_dataset(hf_dataset, DatasetSplitType.kFold, None, 5)
    print("done")
