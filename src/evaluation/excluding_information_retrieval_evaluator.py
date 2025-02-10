from __future__ import annotations

import heapq
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ExcludingInformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class works exactly like InformationRetrievalEvaluator but optionally excludes
    certain documents for each query during ranking. You can provide an 'excluded_docs'
    dictionary: { query_id -> set of doc_ids } to skip for that particular query.

    Example usage:

    .. code-block:: python

        excluded_mapping = {
            "q_001": {"doc_123", "doc_456"},
            "q_010": {"doc_999"}
        }
        evaluator = ExcludingInformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=relevant_docs_dict,
            excluded_docs=excluded_mapping,
            name="my-eval",
        )
        scores = evaluator(model)

    All other parameters work the same as in InformationRetrievalEvaluator.
    """

    def __init__(
        self,
        queries: dict[str, str],   # qid => query
        corpus: dict[str, str],   # cid => doc
        relevant_docs: dict[str, set[str]],  # qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
        corpus_prompt: str | None = None,
        corpus_prompt_name: str | None = None,
        excluded_docs: Optional[dict[str, set[str]]] = None,
    ) -> None:
        """
        Initializes the ExcludingInformationRetrievalEvaluator.

        Args:
            queries (Dict[str, str]): A dictionary mapping query IDs to queries.
            corpus (Dict[str, str]): A dictionary mapping document IDs to documents.
            relevant_docs (Dict[str, Set[str]]): A dictionary mapping query IDs to a set of relevant doc IDs.
            excluded_docs (Dict[str, Set[str]]): A dictionary { query_id -> set of doc_ids } to exclude
                for that particular query. Default: None.

            The other parameters match the original InformationRetrievalEvaluator.
        """
        super().__init__()
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]
        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.query_prompt = query_prompt
        self.query_prompt_name = query_prompt_name
        self.corpus_prompt = corpus_prompt
        self.corpus_prompt_name = corpus_prompt_name

        self.relevant_docs = relevant_docs
        self.excluded_docs = excluded_docs if excluded_docs is not None else {}

        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None
        self.truncate_dim = truncate_dim

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]
        self._append_csv_headers(self.score_function_names)

    def _append_csv_headers(self, score_function_names):
        for score_name in score_function_names:
            for k in self.accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")

            for k in self.precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k}")
                self.csv_headers.append(f"{score_name}-Recall@{k}")

            for k in self.mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")

            for k in self.ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")

            for k in self.map_at_k:
                self.csv_headers.append(f"{score_name}-MAP@{k}")

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Information Retrieval Evaluation of the model on the {self.name} dataset{out_txt}:")

        # If user didn't provide explicit score_functions, use the model's default
        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disk if desired
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")
            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if not self.primary_metric:
            if self.main_score_function is None:
                # By default, pick the NDCG@max_k from whichever scoring function is largest
                score_function = max(
                    [(name, scores[name]["ndcg@k"][max(self.ndcg_at_k)]) for name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"

        metrics = {
            f"{score_function}_{metric_name.replace('@k', '@' + str(k))}": value
            for score_function, values_dict in scores.items()
            for metric_name, values in values_dict.items()
            for k, value in values.items()
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def compute_metrices(
        self, model: SentenceTransformer, corpus_model=None, corpus_embeddings: Tensor | None = None
    ) -> dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Encode queries
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            query_embeddings = model.encode(
                self.queries,
                prompt_name=self.query_prompt_name,
                prompt=self.query_prompt,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
            )

        # Prepare result structure
        queries_result_list = {}
        for name in self.score_functions:
            # We'll store, for each query, the top docs as a min-heap
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Process the corpus in chunks
        for corpus_start_idx in trange(
            0, len(self.corpus), self.corpus_chunk_size, desc="Corpus Chunks", disable=not self.show_progress_bar
        ):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            # Encode a chunk of corpus if no precomputed embeddings are given
            if corpus_embeddings is None:
                with (
                    nullcontext()
                    if self.truncate_dim is None
                    else corpus_model.truncate_sentence_embeddings(self.truncate_dim)
                ):
                    sub_corpus_embeddings = corpus_model.encode(
                        self.corpus[corpus_start_idx:corpus_end_idx],
                        prompt_name=self.corpus_prompt_name,
                        prompt=self.corpus_prompt,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                    )
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute similarity for this chunk
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # We do a top-k on this chunk
                top_k_vals, top_k_idx = torch.topk(
                    pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
                )
                top_k_vals = top_k_vals.cpu().tolist()
                top_k_idx = top_k_idx.cpu().tolist()

                # For each query, gather results
                for q_idx in range(len(query_embeddings)):
                    qid = self.queries_ids[q_idx]
                    exclude_set = self.excluded_docs.get(qid, set())

                    valid_hits = []
                    for corpus_idx, score_val in zip(top_k_idx[q_idx], top_k_vals[q_idx]):
                        cid = self.corpus_ids[corpus_start_idx + corpus_idx]
                        # If this doc is excluded for this query, skip it
                        if cid in exclude_set:
                            continue
                        # Keep track of it in a min-heap of size max_k
                        if len(valid_hits) < max_k:
                            heapq.heappush(valid_hits, (score_val, cid))
                        else:
                            heapq.heappushpop(valid_hits, (score_val, cid))

                    # Merge back into queries_result_list
                    queries_result_list[name][q_idx].extend(valid_hits)

        # Convert heaps to sorted lists
        for name in queries_result_list:
            for q_idx in range(len(queries_result_list[name])):
                # Sort by descending score
                sorted_hits = sorted(queries_result_list[name][q_idx], key=lambda x: x[0], reverse=True)
                queries_result_list[name][q_idx] = [
                    {"corpus_id": cid, "score": score_val} for score_val, cid in sorted_hits
                ]

        logger.info(f"Queries: {len(self.queries)}")
        logger.info(f"Corpus: {len(self.corpus)}\n")

        # Compute IR metrics
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}
        # Print results
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores

    def compute_metrics(self, queries_result_list: list[object]):
        # Same as original
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Calculate metrics
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]
            top_hits = queries_result_list[query_itr]
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k
            for k_val in self.accuracy_at_k:
                for hit in top_hits[:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision & Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)
                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0
                for rank, hit in enumerate(top_hits[:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)
                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(queries_result_list)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k]) if precisions_at_k[k] else 0

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k]) if recall_at_k[k] else 0

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k]) if ndcg[k] else 0

        for k in MRR:
            MRR[k] /= len(queries_result_list)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k]) if AveP_at_k[k] else 0

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    def output_scores(self, scores):
        for k in scores["accuracy@k"]:
            logger.info("Accuracy@{}: {:.2f}%".format(k, scores["accuracy@k"][k] * 100))

        for k in scores["precision@k"]:
            logger.info("Precision@{}: {:.2f}%".format(k, scores["precision@k"][k] * 100))

        for k in scores["recall@k"]:
            logger.info("Recall@{}: {:.2f}%".format(k, scores["recall@k"][k] * 100))

        for k in scores["mrr@k"]:
            logger.info("MRR@{}: {:.4f}".format(k, scores["mrr@k"][k]))

        for k in scores["ndcg@k"]:
            logger.info("NDCG@{}: {:.4f}".format(k, scores["ndcg@k"][k]))

        for k in scores["map@k"]:
            logger.info("MAP@{}: {:.4f}".format(k, scores["map@k"][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 because index starts at 0
        return dcg


if __name__ == "__main__":
    pass
    # Suppose we have:
    #  queries = {"q1": "some query text", ...}
    #  corpus = {"d1": "some doc text", ...}
    #  relevant_docs = {"q1": {"d2", "d3"}, ...}

    # Now we also want to exclude certain docs for specific queries:
    # excluded_docs_map = {
    #     "q1": {"d1"}  # e.g., doc d1 is a trivial substring, so exclude it for q1
    # }
    #
    # evaluator = ExcludingInformationRetrievalEvaluator(
    #     queries=queries,
    #     corpus=corpus,
    #     relevant_docs=relevant_docs,
    #     excluded_docs=excluded_docs_map,
    #     name="demo-ir-eval"
    # )
    #
    # results = evaluator(model)
    # print(results)
