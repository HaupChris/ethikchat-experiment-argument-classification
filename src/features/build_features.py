import random
import warnings
from collections import defaultdict
from typing import Optional, List, Dict, Tuple
from datasets import Dataset, DatasetDict, load_from_disk
from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection

from src.data.classes import PassageSource, Passage, Query, DatasetSplitType
from src.data.dataset_splitting.dataset_splits import create_splits_from_corpus_dataset
from src.features.find_n_cover import approximate_n_cover


def create_dataset_for_multiple_negatives_ranking_loss(
        split_dataset: DatasetDict,
        max_positives_per_query: Optional[int] = None
) -> Dataset:
    """
    Given a split_dataset with:
      - "queries": has columns ["id", "text", ...]
      - "passages": has columns ["id", "text", ...]
      - "queries_relevant_passages_mapping": columns ["query_id", "passages_ids"]

    Returns an HF Dataset with only (query, positive).

    This serves as input to the MultipleNegativesRankingLoss.
    According to the docs (https://www.sbert.net/docs/sentence_transformer/training_overview.html#dataset-format)
    the order of the columns is important: "anchor" is the query, "positive" is the positive passage. The names
    of the columns is not taken into account by the loss function, only the order.
    """

    queries = split_dataset["queries"].to_list()  # each item: {"id", "text", ...}
    passages = split_dataset["passages"].to_list()  # each item: {"id", "text", ...}
    mapping = split_dataset["queries_relevant_passages_mapping"].to_list()

    # Build lookups
    query_id_to_query = {q["id"]: q for q in queries}
    passage_id_to_text = {p["id"]: p["text"] for p in passages}

    examples = []
    for row in mapping:
        q_id = row["query_id"]
        relevant_pids = row["passages_ids"]
        # optionally limit the positives to reduce dataset size
        if max_positives_per_query is not None:
            random.shuffle(relevant_pids)
            relevant_pids = relevant_pids[:max_positives_per_query]

        for pid in relevant_pids:
            if pid not in passage_id_to_text:
                continue

            # add discussion scenario as prefix to scenario specific labels
            labels = []
            for label in query_id_to_query[q_id]["labels"]:
                if label.startswith("Z") or label.startswith("NZ") or label.startswith("FAQ"):
                    labels.append(f"{query_id_to_query[q_id]['discussion_scenario']}_{label}")
                else:
                    labels.append(label)

            examples.append({
                "query": query_id_to_query[q_id]["text"],
                "positive": passage_id_to_text[pid],
                # "labels": labels
            })

    return Dataset.from_list(examples)


def add_scenario_tokens_to_texts(split_dataset: DatasetDict,
                                 split_dataset_keys: List[str] = ["queries", "passages"]) -> DatasetDict:
    """
    Given a split_dataset where each split_datasets_key has a "text" and a "discussion_scenario" feature:

    Returns the same dataset with the "text" columns of each split_dataset_key modified to include a discussion scenario
    string as a prefix to the text.

    This is useful to have the scenario information as part of the text, which can be used by the model to learn
    scenario-specific patterns.
    """

    split_keys_with_scenario_token_texts = {}

    for split_dataset_key in split_dataset_keys:
        split_keys_with_scenario_token_texts[split_dataset_key] = split_dataset[split_dataset_key].map(
            lambda x: {**x, "text": f"[{x['discussion_scenario']}] {x['text']}"}
        )

    return DatasetDict({
        **split_dataset,
        **split_keys_with_scenario_token_texts
    })


def add_context_to_texts(split_dataset: DatasetDict, context_length: int, sep_token="[SEP]",
                         split_dataset_key: str = "queries") -> DatasetDict:
    """
    Args:
        split_dataset (DatasetDict):
        context_length (int): -1 for the whole conversation history up to the utterance, 0 for no context or other positive integers. If the specified context in longer than the available context, all context is used.
        sep_token (str): the sep_token of the model tokenizer
        split_dataset_key: e.g. "queries" or "noisy_queries"
    Returns:
        A DatsetDict with the modified queries and the all other parts unchanged.

    """

    def concatenate_context(example):
        # Extract the conversation history
        context = example['context']  # Assuming 'context' is a list of tuples [('Speaker', 'Utterance'), ...]

        # Determine the portion of context to include
        if context_length == -1:
            selected_context = context
        elif context_length == 0:
            selected_context = []
        elif context_length > 0:
            selected_context = context[-context_length:]
        else:
            raise ValueError(f"Context length of {context_length} is not allowed.")

        # Format the context
        formatted_context = f" {sep_token} ".join(
            [f"[{speaker.upper()}] {utterance}" for speaker, utterance in selected_context])

        # Prepend the formatted context to the main text
        example['text'] = f"{formatted_context} {sep_token} [USER] {example['text']}" if formatted_context else example[
            'text']

        return example

    split_dataset[split_dataset_key] = split_dataset[split_dataset_key].map(concatenate_context)

    return split_dataset


def filter_queries_for_few_shot_setting(split_dataset: DatasetDict, num_shots: int) -> DatasetDict:
    """
    Filters the queries in the dataset so that each label has about 'num_shots' queries.
    Due to the possibility of multiple labels for a query, it can happen that num_shots
    of some queries may differ slightly from the passed number of the parameter.
    If fewer than 'num_shots' passages exist
    for a label, all passages for that label are kept and a warning is issued.

    Args:
        split_dataset: Dictionary with keys of at least ["queries", "passages", "queries_relevant_passages_mapping", "queries_trivial_passages_mapping"].
        num_shots: Number of passages to keep per label.

    Returns:
        A filtered copy of the dataset with a reduced set of queries and updated mappings.
         Other keys of the dataset are untouched and kept in the output dataset dict.
    """

    def split_dataset_by_filter(dataset: Dataset, filter_fn) -> Tuple[Dataset, Dataset]:
        """
        Applies a filter function and returns two datasets:
        - The filtered (kept) dataset
        - The removed dataset
        """
        keep_indices = []
        drop_indices = []

        for idx, example in enumerate(dataset):
            if filter_fn(example):
                keep_indices.append(idx)
            else:
                drop_indices.append(idx)

        kept = dataset.select(keep_indices)
        removed = dataset.select(drop_indices)

        return kept, removed

    if num_shots < 0:
        warnings.warn("Parameter 'num_shots' < 0. All queries are kept in the dataset.")
        return split_dataset

    queries = [Query(
        id=entry["id"],
        text=entry["text"],
        labels=entry["labels"],
        discussion_scenario=entry["discussion_scenario"],
        context=entry["context"],
        scenario_description=entry["scenario_description"],
        scenario_question=entry["scenario_question"]
    ) for entry in split_dataset["queries"]]

    scenario_queries = defaultdict(list)
    for query in queries:
        scenario_queries[query.discussion_scenario].append(query)

    result_queries = defaultdict(list)
    for scenario, s_queries in scenario_queries.items():
        n_cover_queries = approximate_n_cover(s_queries, num_shots)
        if len(n_cover_queries) != len(s_queries):
            print(len(n_cover_queries), len(s_queries))
        result_queries[scenario] = n_cover_queries

    result = [query for _, s_queries in result_queries.items() for query in s_queries]
    result_query_ids = list(map(lambda q: q.id, result))

    split_dataset["queries"] = split_dataset["queries"].filter(lambda query: query["id"] in result_query_ids)

    # UserUtterance = "user_utterance"
    # ArgumentgraphSummary = "argumentgraph_summary"
    # ArgumentgraphFullText = "argumentgraph_full_text"
    # ArgumentgraphSample = "argumentgraph_sample"

    # remove all passages from the dataset that originate from queries that have been removed.
    split_dataset["passages"], removed = split_dataset_by_filter(split_dataset["passages"],
                                                                 lambda entry: entry[
                                                                                   "passage_source"] != PassageSource.UserUtterance.value or
                                                                               entry[
                                                                                   "retrieved_query_id"] in result_query_ids)

    split_dataset["queries_relevant_passages_mapping"] = split_dataset["queries_relevant_passages_mapping"].filter(
        lambda entry: entry["query_id"] in result_query_ids)
    split_dataset["queries_trivial_passages_mapping"] = split_dataset["queries_trivial_passages_mapping"].filter(
        lambda entry: entry["query_id"] in result_query_ids)

    return split_dataset


def filter_passages_for_few_shot_setting(
        split_dataset: dict,
        num_shots: int,
        prefered_passage_sources: List[PassageSource] = []
) -> DatasetDict:
    """
    Filters the passages in the dataset so that each label has at most 'num_shots' passages,
    optionally prioritizing certain passage sources first. If fewer than 'num_shots' passages exist
    for a label, all passages for that label are kept and a warning is issued.

    Args:
        split_dataset: Dictionary with keys ["queries", "passages", "queries_relevant_passages_mapping", "queries_trivial_passages_mapping"].
        num_shots: Number of passages to keep per label.
        prefered_passage_sources: If provided, list of sources to try first (in the given order).

    Returns:
        A filtered copy of the dataset with a reduced set of passages and updated mappings.
        Other keys of the dataset are untouched and kept in the output dataset dict.

    """

    if num_shots < 0:
        warnings.warn("Parameter 'num_shots' < 0. All passages are kept in the dataset.")
        return split_dataset

    # Convert the dictionary entries into Passage objects
    if len(prefered_passage_sources) == 0:
        prefered_passage_sources = [PassageSource.ArgumentgraphFullText,
                                    PassageSource.ArgumentgraphSummary,
                                    PassageSource.UserUtterance,
                                    PassageSource.ArgumentgraphSample]

    # Helper function to pick up to num_shots passages according to the source preference
    def pick_passages_for_label(passages_of_label: List[Passage], selected_label: str):
        total_available = len(passages_of_label)
        random.seed(42)
        random.shuffle(passages_of_label)

        # If there's not enough total to reach num_shots, keep everything and warn
        if total_available < num_shots:
            warnings.warn(
                f"Label '{selected_label}' has only {total_available} passages but {num_shots} requested; keeping all."
            )
            return passages_of_label

        # Otherwise, we attempt to pick in order from the prefered_passage_source
        if prefered_passage_sources:
            chosen = []
            # In preferred order
            for src in prefered_passage_sources:
                if len(chosen) >= num_shots:
                    break
                # Keep them in the order they appear, but filtered by src

                chosen.extend(
                    [p for p in passages_of_label if p.passage_source == src.value][: (num_shots - len(chosen))])
            # If we still don't have enough, pick from remaining sources (in order)
            if len(chosen) < num_shots:
                remaining_needed = num_shots - len(chosen)
                # All sources not in prefered_passage_source
                remaining_passages = [
                    p for p in passages_of_label if p.passage_source not in prefered_passage_sources
                ]
                chosen.extend(remaining_passages[:remaining_needed])
            return chosen
        else:
            # No preference given; just pick the first num_shots
            return passages_of_label[:num_shots]

    passages = [
        Passage(
            id=entry["id"],
            text=entry["text"],
            label=entry["label"],
            discussion_scenario=entry["discussion_scenario"],
            passage_source=entry["passage_source"],
            retrieved_query_id=entry["retrieved_query_id"]
        )
        for entry in split_dataset["passages"]
    ]

    # Group passages by label, preserving the original order
    scenario_label_to_passages = defaultdict(lambda: defaultdict(list))
    for passage in passages:
        scenario_label_to_passages[passage.discussion_scenario][passage.label].append(passage)

    # Select passages for each label
    filtered_passages_ids = []
    for scenario, label_to_passages in scenario_label_to_passages.items():
        for label, label_passages in label_to_passages.items():
            chosen_passages_for_label = pick_passages_for_label(label_passages, label)
            filtered_passages_ids.extend([p.id for p in chosen_passages_for_label])

    split_dataset["passages"] = split_dataset["passages"].filter(lambda passage: passage["id"] in filtered_passages_ids)

    def filter_passage_ids(entry):
        entry["passages_ids"] = [pid for pid in entry["passages_ids"] if pid in filtered_passages_ids]
        return entry

    split_dataset["queries_relevant_passages_mapping"] = split_dataset["queries_relevant_passages_mapping"].map(
        filter_passage_ids)

    split_dataset["queries_trivial_passages_mapping"] = split_dataset["queries_trivial_passages_mapping"].map(
        filter_passage_ids)

    return split_dataset


def get_similarity_score_for_passage_pair(passage_1: Passage, passage_2: Passage,
                                          argument_graphs: Dict[str, ResponseTemplateCollection]) -> float:
    """
    # scores:
    # 4.0 Fulltext und Summary texte vom selben label
    # 3.0 Selbes label, aber mind einer der Texte ist aus der Quelle user oder synthetic
    # 2.0 texte, deren templates eine kannte gemeinsam haben
    # 1.0 texte, die zur gleichen Gruppe gehören
    # 0.0 alle anderen
    Args:
        passage_1 ():
        passage_2 ():
        argument_graphs ():

    Returns:

    """
    same_source = {PassageSource.ArgumentgraphSummary, PassageSource.ArgumentgraphFullText}
    template_1 = argument_graphs[passage_1.discussion_scenario].get_template_for_label(passage_1.label)
    template_2 = argument_graphs[passage_2.discussion_scenario].get_template_for_label(passage_2.label)

    if passage_1.discussion_scenario != passage_2.discussion_scenario:
        return 0.0

    if passage_1.label == passage_2.label:
        if passage_1.passage_source in same_source and passage_2.passage_source in same_source:
            return 4.0
        return 3.0
    if template_1 in template_2.parent_templates or template_2 in template_1.parent_templates:
        return 2.0
    if template_1 in template_2.group_templates or template_2 in template_1.group_templates:
        return 1.0
    return 0.0


def create_textual_similarity_dataset(split_dataset: DatasetDict,
                                      argument_graphs: Dict[str, ResponseTemplateCollection]) -> List[
    Tuple[str, str, float]]:
    passages = Passage.get_passages_from_hf_dataset(split_dataset["train"]["passages"])
    queries = Query.get_queries_from_hf_dataset(split_dataset["train"]["queries"])

    user_passages = list(filter(lambda passage: passage.passage_source == PassageSource.UserUtterance, passages))
    non_user_passages = list(filter(lambda passage: passage.passage_source != PassageSource.UserUtterance, passages))

    query_ids = [query.id for query in queries]
    non_user_passages = list(filter(lambda passage: passage.retrieved_query_id not in query_ids, non_user_passages))

    train_passages = user_passages + non_user_passages

    similarity_texts = defaultdict(list)
    for passage1 in train_passages:
        for passage2 in train_passages:
            if passage1 == passage2:
                continue
            else:
                similarity_score = get_similarity_score_for_passage_pair(passage1, passage2, argument_graphs)
                similarity_score /= 4.
                similarity_texts[similarity_score].append((passage1.text, passage2.text, similarity_score))

    return similarity_texts


if __name__ == "__main__":
    from src.models.train_model_sweep import load_argument_graphs

    dataset_folder = "../../data/processed/with_context"
    corpus_ds = load_from_disk(f"{dataset_folder}/corpus_dataset_v2")
    argument_graphs = load_argument_graphs(
        "/home/christian/PycharmProjects/ethikchat-experiment-argument-classification", is_test_run=True)

    in_distribution_split = create_splits_from_corpus_dataset(corpus_dataset=corpus_ds,
                                                              dataset_split_type=DatasetSplitType.InDistribution,
                                                              save_folder=dataset_folder,
                                                              dataset_save_name="dataset_split_in_distribution")
    textual_similarity_dataset = create_textual_similarity_dataset(in_distribution_split, argument_graphs)

    ids_train = in_distribution_split["train"]
    # ids_train_empty = DatasetDict({
    #     "queries": in_distribution_split["train"]["queries"],
    #     "passages": in_distribution_split["train"]["passages"],
    # })
    in_distribution_split_with_context = add_context_to_texts(ids_train, -1, "[SEP]")
    in_distribution_split_with_scenario_tokens = add_scenario_tokens_to_texts(in_distribution_split_with_context)
    in_distribution_split_with_scenario_tokens_few_shot = filter_passages_for_few_shot_setting(
        in_distribution_split_with_scenario_tokens, 1)

    # pos_ds_train = create_dataset_for_multiple_negatives_ranking_loss(
    #     in_distribution_split_with_scenario_tokens["train"])
    #
    # # pos_ds_train = create_dataset_for_multiple_negatives_ranking_loss(
    # #     in_distribution_split_with_scenario_tokens["train"])
    # pos_ds_train = create_dataset_for_multiple_negatives_ranking_loss(
    #     in_distribution_split_with_context["train"])
    # print(pos_ds_train)
