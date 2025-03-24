import random
from typing import Optional
from datasets import Dataset, DatasetDict, load_from_disk

from src.data.create_corpus_dataset import DatasetSplitType
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
        example['text'] = f"{formatted_context} {sep_token} [USER] {example['text']}" if formatted_context else example['text']

        return example

    split_dataset["queries"] = split_dataset["queries"].map(concatenate_context)

    return split_dataset


if __name__ == "__main__":
    dataset_folder = "../../data/processed"
    corpus_ds = load_from_disk(f"{dataset_folder}/corpus_dataset_with_context")
    in_distribution_split = create_splits_from_corpus_dataset(corpus_dataset=corpus_ds,
                                                              dataset_split_type=DatasetSplitType.InDistribution,
                                                              save_folder=dataset_folder,
                                                              dataset_save_name="dataset_split_in_distribution_labels_per_scenario", )
    ids_train = in_distribution_split["train"]
    ids_train_empty = DatasetDict({
        "queries": in_distribution_split["train"]["queries"],
        "passages": in_distribution_split["train"]["passages"],
    })
    in_distribution_split_with_context = add_context_to_texts(ids_train_empty, -1, "[SEP]")
    in_distribution_split_with_scenario_tokens = add_scenario_tokens_to_texts(in_distribution_split_with_context)

    pos_ds_train = create_dataset_for_multiple_negatives_ranking_loss(
        in_distribution_split_with_scenario_tokens["train"])

    # pos_ds_train = create_dataset_for_multiple_negatives_ranking_loss(
    #     in_distribution_split_with_scenario_tokens["train"])
    pos_ds_train = create_dataset_for_multiple_negatives_ranking_loss(
        in_distribution_split_with_context["train"])
    print(pos_ds_train)
