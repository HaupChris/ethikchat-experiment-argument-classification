from __future__ import annotations

import heapq
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Optional, Dict, List, Tuple, Set

import numpy as np
import torch
from torch import Tensor
from tqdm import trange

import wandb  # For logging results
from wandb.wandb_run import Run

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ExcludingInformationRetrievalEvaluator(SentenceEvaluator):
    """
    IR evaluator that logs:
      1) Overall Accuracy@k, MRR@k, NDCG@k (+ optional loss)
      2) Per-topic Accuracy@k
      3) Topic confusion matrices (absolute & relative) using wandb's API
    """

    def __init__(
        self,
        queries: Dict[str, any],              # qid => Query object (must have .text, .discussion_scenario, .id)
        corpus: Dict[str, any],               # cid => Passage object (must have .text, .discussion_scenario, .id)
        relevant_docs: Dict[str, Set[str]],   # qid => set of relevant doc IDs
        corpus_chunk_size: int = 50000,
        accuracy_at_k: List[int] = [1, 3, 5],
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: Dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
        corpus_prompt: str | None = None,
        corpus_prompt_name: str | None = None,
        excluded_docs: Optional[Dict[str, Set[str]]] = None,
        log_top_k_predictions: int = 0,
        run: Run = None
    ) -> None:
        super().__init__()
        # Filter out queries with no relevant docs
        self.queries = [
            q for qid, q in queries.items()
            if qid in relevant_docs and len(relevant_docs[qid]) > 0
        ]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        self.corpus_map = dict(zip(self.corpus_ids, self.corpus))  # doc_id -> Passage object

        self.relevant_docs = relevant_docs
        self.excluded_docs = excluded_docs if excluded_docs else {}

        # Logging & config
        self.corpus_chunk_size = corpus_chunk_size
        self.accuracy_at_k = accuracy_at_k
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv

        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None

        self.truncate_dim = truncate_dim
        self.log_top_k_predictions = log_top_k_predictions
        self.run = run

        self.query_prompt = query_prompt
        self.query_prompt_name = query_prompt_name
        self.corpus_prompt = corpus_prompt
        self.corpus_prompt_name = corpus_prompt_name

        # For CSV
        if name:
            name = "_" + name
        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]
        # We'll append for each scoring function once we know them at runtime
        self._append_csv_headers(self.score_function_names)

    def _append_csv_headers(self, score_function_names):
        for score_name in score_function_names:
            for k in self.accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")
            for k in self.mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")
            for k in self.ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")
        # We'll also log a 'loss' column
        self.csv_headers.append("loss")

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
        loss: float = None,   # Pass in a loss value if you want it logged
        *args,
        **kwargs
    ) -> dict[str, float]:
        # Info about the evaluation stage
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Information Retrieval Evaluation on the {self.name} dataset{out_txt}:")

        # If no custom score functions, use model's default
        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)

        # 1) Compute metrics overall
        scores, queries_result_list = self._compute_scores(model, *args, **kwargs)

        # 2) Write to CSV if needed
        if output_path and self.write_csv:
            self._write_csv(scores, output_path, epoch, steps, loss)

        # 3) Determine primary metric if not set
        if not self.primary_metric:
            if self.main_score_function is None:
                # By default, pick NDCG@max
                chosen_score_func = max(
                    [(s_name, scores[s_name]["ndcg@k"][max(self.ndcg_at_k)]) for s_name in self.score_function_names],
                    key=lambda x: x[1]
                )[0]
                self.primary_metric = f"{chosen_score_func}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"

        # 4) Convert to single dict for logging
        metrics = {}
        for score_function, metric_dict in scores.items():
            for metric_name, values_for_k in metric_dict.items():
                for k, val in values_for_k.items():
                    # e.g. "cosine_accuracy@3" : 0.89
                    metric_key = f"{score_function}_{metric_name.replace('@k','@'+str(k))}"
                    metrics[metric_key] = val

        # 5) Per-topic accuracy@k
        per_topic_data = self._compute_topic_accuracy_at_k(queries_result_list)
        metrics.update(per_topic_data)

        # 6) Optionally log loss
        if loss is not None:
            metrics["loss"] = loss

        # 7) Log topic confusion matrix
        self._log_topic_confusion_matrix(queries_result_list)

        # 8) Give these metrics a prefix and store in model card
        final_metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, final_metrics, epoch, steps)

        # 9) Log to W&B (if self.run is set)
        if self.run:
            self.run.log(final_metrics)

        return final_metrics

    def _compute_scores(
        self,
        model: SentenceTransformer,
        corpus_model=None,
        corpus_embeddings: Tensor | None = None
    ) -> Tuple[Dict[str, Dict[str, Dict[int, float]]],
               Dict[str, List[List[Tuple[float, str]]]]]:
        """Encodes queries vs. corpus, does top-k retrieval, then computes IR metrics."""

        if corpus_model is None:
            corpus_model = model

        # We need to handle up to the largest k
        max_k = max(
            max(self.accuracy_at_k) if self.accuracy_at_k else 1,
            max(self.mrr_at_k) if self.mrr_at_k else 1,
            max(self.ndcg_at_k) if self.ndcg_at_k else 1
        )

        # Encode queries
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            query_embeddings = model.encode(
                [query.text for query in self.queries],
                prompt_name=self.query_prompt_name,
                prompt=self.query_prompt,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
            )

        # Prepare structure: queries_result_list[score_func][q_idx] = [(score, doc_id), ...]
        queries_result_list: Dict[str, List[List[Tuple[float, str]]]] = {}
        for score_name in self.score_functions:
            queries_result_list[score_name] = [[] for _ in range(len(query_embeddings))]

        # Encode corpus in chunks and compare
        for corpus_start in trange(0, len(self.corpus), self.corpus_chunk_size, disable=not self.show_progress_bar):
            corpus_end = min(corpus_start + self.corpus_chunk_size, len(self.corpus))

            # Encode chunk
            if corpus_embeddings is None:
                with nullcontext() if self.truncate_dim is None else corpus_model.truncate_sentence_embeddings(self.truncate_dim):
                    sub_corpus_emb = corpus_model.encode(
                        [p.text for p in self.corpus[corpus_start:corpus_end]],
                        prompt_name=self.corpus_prompt_name,
                        prompt=self.corpus_prompt,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                    )
            else:
                sub_corpus_emb = corpus_embeddings[corpus_start:corpus_end]

            # For each scoring function
            for score_fn_name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_emb)
                # Get top-k for each query
                top_k_vals, top_k_idx = torch.topk(
                    pair_scores, min(max_k, sub_corpus_emb.size(0)), dim=1, largest=True, sorted=False
                )
                top_k_vals = top_k_vals.cpu().tolist()
                top_k_idx = top_k_idx.cpu().tolist()

                # Store them
                for q_idx in range(len(query_embeddings)):
                    exclude_set = self.excluded_docs.get(self.queries[q_idx].id, set())
                    valid_hits = queries_result_list[score_fn_name][q_idx]

                    for doc_idx, score_val in zip(top_k_idx[q_idx], top_k_vals[q_idx]):
                        doc_id = self.corpus_ids[corpus_start + doc_idx]
                        if doc_id in exclude_set:
                            continue
                        if len(valid_hits) < max_k:
                            heapq.heappush(valid_hits, (score_val, doc_id))
                        else:
                            heapq.heappushpop(valid_hits, (score_val, doc_id))

        # Sort the hits for each query by descending score
        for score_fn_name, results_for_queries in queries_result_list.items():
            for q_idx in range(len(results_for_queries)):
                sorted_hits = sorted(results_for_queries[q_idx], key=lambda x: x[0], reverse=True)
                results_for_queries[q_idx] = sorted_hits

        # Compute metrics (accuracy@k, mrr@k, ndcg@k)
        scores: Dict[str, Dict[str, Dict[int, float]]] = {}
        for score_fn_name in self.score_functions:
            result = self._compute_ir_metrics(queries_result_list[score_fn_name])
            scores[score_fn_name] = result
            self._log_metrics_to_console(score_fn_name, result)

        return scores, queries_result_list

    def _compute_ir_metrics(self, per_query_hits: List[List[Tuple[float, str]]]) -> Dict[str, Dict[int, float]]:
        """
        Calculate overall Accuracy@k, MRR@k, NDCG@k for the given set of query results.
        per_query_hits[q_idx] = list of (score, doc_id) sorted descending.
        """
        # Initialize counters
        acc_counts = {k: 0 for k in self.accuracy_at_k}
        mrr_sums = {k: 0.0 for k in self.mrr_at_k}
        ndcg_lists = {k: [] for k in self.ndcg_at_k}

        num_queries = len(per_query_hits)

        # Evaluate each query
        for q_idx, hits in enumerate(per_query_hits):
            relevant = self.relevant_docs[self.queries[q_idx].id]

            # Accuracy@k
            for k_val in self.accuracy_at_k:
                for (_, doc_id) in hits[:k_val]:
                    if doc_id in relevant:
                        acc_counts[k_val] += 1
                        break

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, (_, doc_id) in enumerate(hits[:k_val]):
                    if doc_id in relevant:
                        mrr_sums[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevances = [1 if d in relevant else 0 for (_, d) in hits[:k_val]]
                ideal_relevances = [1] * len(relevant)  # best-case scenario
                dcg_val = self._compute_dcg_at_k(predicted_relevances, k_val)
                idcg_val = self._compute_dcg_at_k(ideal_relevances, k_val)
                ndcg_val = dcg_val / (idcg_val + 1e-8)
                ndcg_lists[k_val].append(ndcg_val)

        # Average out the metrics
        acc_final = {k: acc_counts[k] / num_queries for k in self.accuracy_at_k}
        mrr_final = {k: mrr_sums[k] / num_queries for k in self.mrr_at_k}
        ndcg_final = {k: np.mean(ndcg_lists[k]) if ndcg_lists[k] else 0.0 for k in self.ndcg_at_k}

        return {
            "accuracy@k": acc_final,
            "mrr@k": mrr_final,
            "ndcg@k": ndcg_final,
        }

    def _compute_dcg_at_k(self, relevances, k):
        """Helper to compute DCG@k for a relevance list of 0/1."""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += rel / np.log2(i + 2)
        return dcg

    def _compute_topic_accuracy_at_k(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> Dict[str, float]:
        """
        Computes accuracy@k per topic (discussion_scenario).
        Returns: { "topicA_accuracy@1": val, "topicA_accuracy@3": val, ..., "topicB_accuracy@1": ... }
        for each scoring function in queries_result_list.
        """
        metrics = {}
        # Collect queries by topic
        topic_to_query_indices = {}
        for i, q in enumerate(self.queries):
            t = q.discussion_scenario
            if t not in topic_to_query_indices:
                topic_to_query_indices[t] = []
            topic_to_query_indices[t].append(i)

        # For each score function:
        for score_fn_name, hits_per_query in queries_result_list.items():
            # We only do accuracy@k
            for k_val in self.accuracy_at_k:
                for topic, q_indices in topic_to_query_indices.items():
                    correct = 0
                    for q_idx in q_indices:
                        relevant = self.relevant_docs[self.queries[q_idx].id]
                        top_k_hits = hits_per_query[q_idx][:k_val]
                        # If any top-k doc is relevant, count as correct
                        if any(doc_id in relevant for (_, doc_id) in top_k_hits):
                            correct += 1
                    topic_acc = correct / len(q_indices)
                    # e.g.: "cosine_medai_accuracy@3": 0.87
                    key = f"{score_fn_name}_{topic}_accuracy@{k_val}"
                    metrics[key] = topic_acc
        return metrics

    def _log_metrics_to_console(self, score_func_name: str, metric_values: Dict[str, Dict[int, float]]):
        """Nicely logs overall metrics to the console."""
        logger.info(f"\nScore Function: {score_func_name}")
        # Overall Accuracy@k
        acc_map = metric_values["accuracy@k"]
        for k, val in acc_map.items():
            logger.info(f"  Accuracy@{k}: {val * 100:.2f}%")

        # MRR@k
        mrr_map = metric_values["mrr@k"]
        for k, val in mrr_map.items():
            logger.info(f"  MRR@{k}: {val:.4f}")

        # NDCG@k
        ndcg_map = metric_values["ndcg@k"]
        for k, val in ndcg_map.items():
            logger.info(f"  NDCG@{k}: {val:.4f}")

    def _write_csv(self, scores, output_path, epoch, steps, loss):
        """Writes a single row in CSV with overall metrics + loss for each score function."""
        csv_path = os.path.join(output_path, self.csv_file)
        write_header = not os.path.isfile(csv_path)

        with open(csv_path, mode="a", encoding="utf-8") as fOut:
            if write_header:
                fOut.write(",".join(self.csv_headers) + "\n")

            output_data = [epoch, steps]
            for score_fn_name in self.score_function_names:
                data = scores[score_fn_name]
                # accuracy@k
                for k_val in self.accuracy_at_k:
                    output_data.append(data["accuracy@k"][k_val])
                # mrr@k
                for k_val in self.mrr_at_k:
                    output_data.append(data["mrr@k"][k_val])
                # ndcg@k
                for k_val in self.ndcg_at_k:
                    output_data.append(data["ndcg@k"][k_val])

            # Append loss
            output_data.append(loss if loss is not None else "n/a")

            fOut.write(",".join(map(str, output_data)) + "\n")

    def _log_topic_confusion_matrix(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        """
        Logs two confusion matrices (absolute & relative) of query-topic vs. predicted-topic (top-1 doc).
        Uses wandb.plot.confusion_matrix for each scoring function in queries_result_list.
        """

        if not self.run:
            return

        from sklearn.metrics import confusion_matrix

        # For each scoring function, collect top1 predictions
        for score_fn_name, hits_per_query in queries_result_list.items():
            # Build ground truth vs. predicted
            y_true = []
            y_pred = []

            for q_idx, hits in enumerate(hits_per_query):
                actual_topic = self.queries[q_idx].discussion_scenario
                if len(hits) == 0:
                    # no docs returned
                    predicted_topic = "NO_DOC"
                else:
                    top_doc_id = hits[0][1]
                    predicted_topic = self.corpus_map[top_doc_id].discussion_scenario

                y_true.append(actual_topic)
                y_pred.append(predicted_topic)

            # All possible topics that appear in ground truth or predictions
            all_topics = sorted(set(y_true + y_pred))

            label2index = {lbl: idx for idx, lbl in enumerate(all_topics)}
            y_true_idx = [label2index[t] for t in y_true]
            y_pred_idx = [label2index[t] for t in y_pred]

            # --------- Absolute Confusion Matrix --------- #
            self.run.log({
                f"{self.name}_{score_fn_name}_topic_conf_mat_abs": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true_idx,
                    preds=y_pred_idx,
                    class_names=all_topics,
                    title=f"{score_fn_name}: Topic Confusion Matrix (Absolute)"
                )
            })

            # # --------- Relative / Normalized Confusion Matrix --------- #
            # # We'll compute the matrix and pass the same y_true, y_pred so WandB can render it,
            # # but label it differently so you can see them side by side.
            # # (W&B's UI already has a toggle for normalized vs. absolute, but this explicitly logs both.)
            #
            # cm = confusion_matrix(y_true, y_pred, labels=all_topics)
            # # row-wise normalization
            # row_sums = cm.sum(axis=1, keepdims=True)
            # cm_normalized = cm / np.maximum(row_sums, 1e-8)
            #
            # # W&B confusion_matrix() doesn't have a direct param for manual NxN,
            # # so we just reuse y_true/preds, but rename the chart. The UI can toggle anyway.
            # self.run.log({
            #     f"{self.name}_{score_fn_name}_topic_conf_mat_rel": wandb.plot.confusion_matrix(
            #         probs=None,
            #         y_true=y_true_idx,
            #         preds=y_pred_idx,
            #         class_names=all_topics,
            #         title=f"{score_fn_name}: Topic Confusion Matrix (Relative)"
            #     )
            # })
