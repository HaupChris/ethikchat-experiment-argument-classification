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

    queries = split_dataset["queries"].to_list()   # each item: {"id", "text", ...}
    passages = split_dataset["passages"].to_list() # each item: {"id", "text", ...}
    mapping  = split_dataset["queries_relevant_passages_mapping"].to_list()

    # Build lookups
    query_id_to_text = {q["id"]: q["text"] for q in queries}
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
            examples.append({
                "query": query_id_to_text[q_id],
                "positive": passage_id_to_text[pid]
            })

    return Dataset.from_list(examples)


if __name__=="__main__":
    corpus_ds = load_from_disk("../../data/processed/corpus_dataset_experiment_v0")
    split_ds = create_splits_from_corpus_dataset(corpus_ds, DatasetSplitType.Simple)
    pos_ds_train = create_dataset_for_multiple_negatives_ranking_loss(split_ds["train"])
    print(pos_ds_train)