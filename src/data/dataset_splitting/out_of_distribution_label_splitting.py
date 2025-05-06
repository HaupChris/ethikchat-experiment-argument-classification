import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import pandas as pd
from datasets import DatasetDict, Dataset

from sklearn.model_selection import train_test_split

from src.data.classes import Query, Passage
from src.data.dataset_splitting.utils import create_datasetdict_for_query_ids


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
