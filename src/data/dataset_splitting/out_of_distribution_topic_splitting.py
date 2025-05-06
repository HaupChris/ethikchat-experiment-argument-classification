import random

from datasets import DatasetDict
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario

from src.data.dataset_splitting.utils import create_datasetdict_for_query_ids


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
