import dataclasses
import random
import warnings
from collections import defaultdict
from typing import Optional, List
from datasets import Dataset, DatasetDict, load_from_disk

from src.data.create_corpus_dataset import DatasetSplitType, Passage, PassageSource, Query
from src.data.dataset_splits import create_splits_from_corpus_dataset


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


def add_scenario_tokens_to_texts(split_dataset: DatasetDict) -> DatasetDict:
    """
    Given a split_dataset with:
      - "queries": has columns ["id", "text", ...]
      - "passages": has columns ["id", "text", ...]
      - and other splits

    Returns the same dataset with the "text" columns of queries and passages modified to include the discussion scenario
    as a prefix to the text.

    This is useful to have the scenario information as part of the text, which can be used by the model to learn
    scenario-specific patterns.
    """

    queries = split_dataset["queries"].map(
        lambda x: {**x, "text": f"[{x['discussion_scenario']}] {x['text']}"}
    )
    passages = split_dataset["passages"].map(
        lambda x: {**x, "text": f"[{x['discussion_scenario']}] {x['text']}"}
    )

    return DatasetDict({
        **split_dataset,
        "queries": queries,
        "passages": passages,
    })


def add_context_to_texts(split_dataset: DatasetDict, context_length: int, sep_token="[SEP]") -> DatasetDict:
    """
    Args:
        split_dataset (DatasetDict):
        context_length (int): -1 for the whole conversation history up to the utterance, 0 for no context or other positive integers. If the specified context in longer than the available context, all context is used.
        sep_token (str): the sep_token of the model tokenizer
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
            raise ValueError(f"Context lenght of {context_length} is not allowed.")

        # Format the context
        formatted_context = f" {sep_token} ".join(
            [f"[{speaker.upper()}] {utterance}" for speaker, utterance in selected_context])

        # Prepend the formatted context to the main text
        example['text'] = f"{formatted_context} {sep_token} [USER] {example['text']}" if formatted_context else example[
            'text']

        return example

    split_dataset["queries"] = split_dataset["queries"].map(concatenate_context)

    return split_dataset


def filter_queries_for_few_shot_setting(split_dataset: DatasetDict, num_shots: int) -> DatasetDict:
    """

    Args:
        split_dataset ():
        num_shots ():

    Returns:

    """
    queries = [Query(
        id=entry["id"],
        text=entry["text"],
        labels=entry["labels"],
        discussion_scenario=entry["discussion_scenario"],
        context=entry["context"],
        scenario_description=entry["scenario_description"],
        scenario_question=entry["scenario_question"]
    ) for entry in split_dataset["queries"]]

    return assigned_queries


def filter_passages_for_few_shot_setting(
        split_dataset: dict,
        num_shots: int,
        prefered_passage_sources: Optional[List[PassageSource]] = [PassageSource.ArgumentgraphFullText,
                                                                   PassageSource.ArgumentgraphSummary,
                                                                   PassageSource.UserUtterance,
                                                                   PassageSource.ArgumentgraphSample]
) -> dict:
    """
    Filters the passages in the dataset so that each label has at most 'num_shots' passages,
    optionally prioritizing certain passage sources first. If fewer than 'num_shots' passages exist
    for a label, all passages for that label are kept and a warning is issued.

    Args:
        split_dataset: Dictionary with keys ["queries", "passages", "queries_relevant_passages_mapping", "queries_trivial_passages_mapping"].
        num_shots: Number of passages to keep per label.
        prefered_passage_sources: If provided, list of sources to try first (in the given order).

    Returns:
        A filtered copy of the dataset with a reduced set of passages (and updated mappings).
    """

    # Convert the dictionary entries into Passage objects
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
    label_to_passages = defaultdict(list)
    for passage in passages:
        label_to_passages[passage.label].append(passage)

    # Helper function to pick up to num_shots passages according to the source preference
    def pick_passages_for_label(label_passages: List[Passage], label: str):
        total_available = len(label_passages)
        random.seed(42)
        random.shuffle(label_passages)

        # If there's not enough total to reach num_shots, keep everything and warn
        if total_available < num_shots:
            warnings.warn(
                f"Label '{label}' has only {total_available} passages but {num_shots} requested; keeping all."
            )
            return label_passages

        # Otherwise, we attempt to pick in order from the prefered_passage_source
        if prefered_passage_sources:
            chosen = []
            # In preferred order
            for src in prefered_passage_sources:
                if len(chosen) >= num_shots:
                    break
                # Keep them in the order they appear, but filtered by src

                chosen.extend([p for p in label_passages if p.passage_source == src][: (num_shots - len(chosen))])
            # If we still don't have enough, pick from remaining sources (in order)
            if len(chosen) < num_shots:
                remaining_needed = num_shots - len(chosen)
                # All sources not in prefered_passage_source
                remaining_passages = [
                    p for p in label_passages if p.passage_source not in prefered_passage_sources
                ]
                chosen.extend(remaining_passages[:remaining_needed])
            return chosen
        else:
            # No preference given; just pick the first num_shots
            return label_passages[:num_shots]

    # Select passages for each label
    filtered_passages = []
    for label, label_passes in label_to_passages.items():
        chosen_for_label = pick_passages_for_label(label_passes, label)
        filtered_passages.extend(chosen_for_label)

    # Build a set of IDs we keep, so we can update relevant/trivial mappings
    kept_passage_ids = set(p.id for p in filtered_passages)

    # Filter the passages portion of the dataset
    new_passages = [dataclasses.asdict(p) for p in filtered_passages]

    # Helper to filter passage IDs in the mapping
    def filter_mapping(original_map):
        # Often the mapping is a dict of query_id -> list of passage_ids
        # Adjust if your mapping structure differs
        filtered_map = {}
        for entry in original_map:
            q_id = entry["query_id"]
            p_ids = entry["passages_ids"]
            retained = [pid for pid in p_ids if pid in kept_passage_ids]
            if retained:
                filtered_map[q_id] = retained
        return filtered_map

    # Update 'relevant_passages_mapping' and 'trivial_passages_mapping'
    new_relevant_map = filter_mapping(split_dataset["queries_relevant_passages_mapping"])
    new_trivial_map = filter_mapping(split_dataset["queries_trivial_passages_mapping"])

    # Return the filtered dataset with the same structure
    filtered_dataset = {
        **split_dataset,
        "passages": new_passages,
        "queries_relevant_passages_mapping": new_relevant_map,
        "queries_trivial_passages_mapping": new_trivial_map
    }

    return filtered_dataset


if __name__ == "__main__":
    dataset_folder = "../../data/processed/with_context"
    corpus_ds = load_from_disk(f"{dataset_folder}/corpus_dataset_v2")
    in_distribution_split = create_splits_from_corpus_dataset(corpus_dataset=corpus_ds,
                                                              dataset_split_type=DatasetSplitType.InDistribution,
                                                              save_folder=dataset_folder,
                                                              dataset_save_name="dataset_split_in_distribution")
    ids_train = in_distribution_split["train"]
    # ids_train_empty = DatasetDict({
    #     "queries": in_distribution_split["train"]["queries"],
    #     "passages": in_distribution_split["train"]["passages"],
    # })
    in_distribution_split_with_context = add_context_to_texts(ids_train, -1, "[SEP]")
    in_distribution_split_with_scenario_tokens = add_scenario_tokens_to_texts(in_distribution_split_with_context)
    in_distribution_split_with_scenario_tokens_few_shot = filter_passages_for_few_shot_setting(in_distribution_split_with_scenario_tokens, 1)

    # pos_ds_train = create_dataset_for_multiple_negatives_ranking_loss(
    #     in_distribution_split_with_scenario_tokens["train"])
    #
    # # pos_ds_train = create_dataset_for_multiple_negatives_ranking_loss(
    # #     in_distribution_split_with_scenario_tokens["train"])
    # pos_ds_train = create_dataset_for_multiple_negatives_ranking_loss(
    #     in_distribution_split_with_context["train"])
    # print(pos_ds_train)
