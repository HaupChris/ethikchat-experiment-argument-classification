from __future__ import annotations

import csv
import heapq
import logging
import os
import numpy as np
from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Optional, Dict, List, Tuple, Set, Iterable

import pandas as pd
import torch
from ethikchat_argtoolkit.ArgumentGraph.response_template import ResponseTemplate, TemplateCategory
from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection
from ethikchat_argtoolkit.ArgumentGraph.stance import Stance
from src.data.create_corpus_dataset import Passage, Query
from torch import Tensor
from tqdm import trange

import wandb  # For logging results
from wandb.wandb_run import Run

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction

from src.data.create_corpus_dataset import load_response_template_collection

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def update_accuracy(accuracy_dict: Dict[str, Dict[str, int]], key: str, condition: bool) -> None:
    """Updates accuracy dictionary by incrementing total and correct counts based on a condition."""
    accuracy_dict[key]["total"] += 1
    if condition:
        accuracy_dict[key]["correct"] += 1


def update_precision(accuracy_dict: Dict[str, Dict[str, int]], key: str, increment_total: int,
                     increment_correct: int) -> None:
    """Updates precision dictionary by incrementing total and correct counts based on a condition."""
    accuracy_dict[key]["total"] += increment_total
    accuracy_dict[key]["correct"] += increment_correct


def calculate_accuracy(data: Dict[str, int]) -> float:
    """Calculates the accuracy for the given dictionary."""
    return data["correct"] / data["total"]


class DeepDiveInformationRetrievalEvaluator(SentenceEvaluator):
    """
    IR evaluator that logs:
        1) Error Analysis Table for Normal and Noisy Queries
        2) Per-topic, per-topic + (node type, node level, and node label) accuracy@K to a W&B table
        2) Per-topic, per-topic + (node type, node level, and node label) precision@K to a W&B table
        3) Per-topic, per-topic + (node type, node level, and node label) stance accuracies as metrics
        4) Confusion matrices over topics, topic + (node type, node level, and node label) with absolute and relative values using W&B's API
        5) Graphs displaying single/multi argument classification accuracies depending on a set confidence threshold
        6) Normal + Noisy Query and Passage Embeddings for use with W&B's 2D Projection Tool
    """

    def __init__(
            self,
            queries: Dict[str, any],  # qid => Query object (must have .text, .discussion_scenario, .id)
            corpus: Dict[str, any],  # cid => Passage object (must have .text, .discussion_scenario, .id)
            relevant_docs: Dict[str, Set[str]],  # qid => set of relevant doc IDs
            corpus_chunk_size: int = 50000,
            accuracy_at_k: List[int] = [1, 3, 5, 7],
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
            run: Run = None,
            argument_graphs: Dict[str, ResponseTemplateCollection] = None,
            maximum_relevancy_depth: int = None,
            confidence_threshold: Optional[float] = None,
            confidence_threshold_steps: Optional[float] = None,
            log_embeddings: Optional[bool] = False,
            noisy_queries: Optional[Dict[str, any]] = None,
            precision_at_k: Optional[List[int]] = [1,3,5,7],
            save_tables_as_csv=True,
            csv_output_dir=".",
            log_to_huggingface=False

    ) -> None:
        super().__init__()
        # Filter out queries with no relevant docs
        self.queries = [
            q for qid, q in queries.items()
            if qid in relevant_docs and len(relevant_docs[qid]) > 0
        ]
        self.noisy_queries = [
            q for qid, q in noisy_queries.items()
        ] if noisy_queries else []

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        self.corpus_map = dict(zip(self.corpus_ids, self.corpus))  # doc_id -> Passage object

        self.relevant_docs = relevant_docs
        self.excluded_docs = excluded_docs if excluded_docs else {}

        # Logging & config
        self.corpus_chunk_size = corpus_chunk_size
        self.accuracy_at_k = sorted([k for k in accuracy_at_k if k > 0])
        self.precision_at_k = precision_at_k if precision_at_k else accuracy_at_k
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv

        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None
        self.primary_metric = "cosine_accuracy@1"

        self.truncate_dim = truncate_dim
        self.run = run

        self.query_prompt = query_prompt
        self.query_prompt_name = query_prompt_name
        self.corpus_prompt = corpus_prompt
        self.corpus_prompt_name = corpus_prompt_name
        self.argument_graphs = argument_graphs
        self.maximum_relevancy_depth = maximum_relevancy_depth if maximum_relevancy_depth else len(self.corpus)
        self.confidence_threshold = confidence_threshold
        self.confidence_threshold_steps = confidence_threshold_steps
        self.log_embeddings = log_embeddings

        # For Embeddings
        self.query_embeddings = []
        self.noisy_query_embeddings = []
        self.corpus_embeddings = []

        self.save_tables_as_csv = save_tables_as_csv
        self.csv_output_dir = csv_output_dir
        self.log_to_huggingface = log_to_huggingface

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
        # We'll also log a 'loss' column
        self.csv_headers.append("loss")

    def __call__(
            self,
            model: SentenceTransformer,
            output_path: str = None,
            epoch: int = -1,
            steps: int = -1,
            loss: float = None,  # Pass in a loss value if you want it logged
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

            # 1) Compute similarities for each query
        queries_result_list, noisy_queries_result_list = self._compute_scores(model, *args, **kwargs)

        # 2) Log qualitative error analysis table
        self._log_qualitative_error_analysis_table(queries_result_list, noisy_queries_result_list)

        # 3) Log accuracy@k tables - now returns overall metrics
        accuracy_metrics = self._compute_accuracies_at_k(queries_result_list)

        # 4) Log precision@k tables - now returns overall metrics
        precision_metrics = self._compute_precision_at_k(queries_result_list)

        # 5) Compute stance accuracies
        stance_metrics = self._compute_stance_accuracy(queries_result_list)

        # 6) Log confusion matrices
        self._log_confusion_matrices_to_wandb(queries_result_list)

        # 7) Log accuracies based on confidence thresholds
        if self.confidence_threshold:
            self._log_single_argument_classification_with_similarity_threshold(queries_result_list,
                                                                               noisy_queries_result_list)
            self._log_multi_argument_classification_with_similarity_threshold(queries_result_list,
                                                                              noisy_queries_result_list)

        # 8) Analyze noisy query types and performance (NEW)
        if self.noisy_queries:
            self._analyze_noisy_query_types()
            self._evaluate_noisy_query_performance_by_type(noisy_queries_result_list)

        # 8) Combine all metrics
        metrics = {**stance_metrics}
        if accuracy_metrics:
            metrics.update(accuracy_metrics)
        if precision_metrics:
            metrics.update(precision_metrics)

        # 9) Give these metrics a prefix and store in model card
        final_metrics = self.prefix_name_to_metrics(metrics, self.name)

        # 10) Log to HuggingFace if enabled
        if self.log_to_huggingface:
            self.store_metrics_in_model_card_data(model, final_metrics, epoch, steps)

        # 11) Log embeddings
        if self.log_embeddings:
            self._log_query_and_passage_embeddings()

        # 12) Log to W&B (if self.run is set)
        if self.run:
            self.run.log(final_metrics)

        # Return metrics
        return final_metrics

    def save_wandb_table_to_csv(self, table, filename, prefix=None):
        """Save a wandb Table object to a CSV file in the specified directory."""
        if not self.save_tables_as_csv:
            return

        # Create output directory
        csv_dir = os.path.join(self.csv_output_dir, prefix) if prefix else self.csv_output_dir
        os.makedirs(csv_dir, exist_ok=True)

        # Convert wandb Table to pandas DataFrame
        data_dict = {col: [] for col in table.columns}
        for row in table.data:
            for i, col in enumerate(table.columns):
                data_dict[col].append(row[i])

        df = pd.DataFrame(data_dict)

        # Save to CSV
        filepath = os.path.join(csv_dir, f"{filename}.csv")
        df.to_csv(filepath, index=False)

        logger.info(f"Saved table to {filepath}")

    def _compute_scores(
            self,
            model: SentenceTransformer,
            corpus_model=None,
            corpus_embeddings: Tensor | None = None
    ) -> Tuple[Dict[str, List[List[Tuple[float, str]]]], Dict[str, List[List[Tuple[float, str]]]]]:
        """Encodes queries vs. corpus, does top-k retrieval, then computes IR metrics."""

        if corpus_model is None:
            corpus_model = model

        # We need to handle up to the largest k
        max_k = max(
            max(self.accuracy_at_k) if self.accuracy_at_k else 1,
            self.maximum_relevancy_depth
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
            noisy_query_embeddings = model.encode(
                [query.text for query in self.noisy_queries],
                prompt_name=self.query_prompt_name,
                prompt=self.query_prompt,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
            )

            self.query_embeddings = query_embeddings.tolist()
            self.noisy_query_embeddings = noisy_query_embeddings.tolist()

        # Prepare structure: queries_result_list[score_func][q_idx] = [(score, doc_id), ...]
        queries_result_list: Dict[str, List[List[Tuple[float, str]]]] = {}
        noisy_queries_result_list: Dict[str, List[List[Tuple[float, str]]]] = {}
        for score_name in self.score_functions:
            queries_result_list[score_name] = [[] for _ in range(len(query_embeddings))]
            noisy_queries_result_list[score_name] = [[] for _ in range(len(noisy_query_embeddings))]

        # Encode corpus in chunks and compare
        for corpus_start in trange(0, len(self.corpus), self.corpus_chunk_size, disable=not self.show_progress_bar):
            corpus_end = min(corpus_start + self.corpus_chunk_size, len(self.corpus))

            # Encode chunk
            if corpus_embeddings is None:
                with nullcontext() if self.truncate_dim is None else corpus_model.truncate_sentence_embeddings(
                        self.truncate_dim):
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

            self.corpus_embeddings.extend(sub_corpus_emb.tolist())
            # For each scoring function
            for score_fn_name, score_function in self.score_functions.items():
                for q_embeddings, q_list, q_result_list in [
                    (query_embeddings, self.queries, queries_result_list),
                    (noisy_query_embeddings, self.noisy_queries, noisy_queries_result_list)
                ]:
                    if len(q_list) > 0:
                        pair_scores = score_function(q_embeddings, sub_corpus_emb)

                        # Get top-k for each query
                        top_k_vals, top_k_idx = torch.topk(
                            pair_scores, min(max_k, sub_corpus_emb.size(0)), dim=1, largest=True, sorted=False
                        )
                        top_k_vals = top_k_vals.cpu().tolist()
                        top_k_idx = top_k_idx.cpu().tolist()

                        # Store them
                        for q_idx in range(len(q_embeddings)):
                            exclude_set = self.excluded_docs.get(q_list[q_idx].id, set())
                            valid_hits = q_result_list[score_fn_name][q_idx]

                            for doc_idx, score_val in zip(top_k_idx[q_idx], top_k_vals[q_idx]):
                                doc_id = self.corpus_ids[corpus_start + doc_idx]
                                if doc_id in exclude_set:
                                    continue
                                if len(valid_hits) < max_k:
                                    heapq.heappush(valid_hits, (score_val, doc_id))
                                else:
                                    heapq.heappushpop(valid_hits, (score_val, doc_id))

        # Sort the hits for each query by descending score
        for q_result_list in [queries_result_list, noisy_queries_result_list]:
            for score_fn_name, results_for_queries in q_result_list.items():
                for q_idx in range(len(results_for_queries)):
                    sorted_hits = sorted(results_for_queries[q_idx], key=lambda x: x[0], reverse=True)
                    results_for_queries[q_idx] = sorted_hits

        return queries_result_list, noisy_queries_result_list

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
                for k_val in self.precision_at_k:
                    output_data.append(data["precision@k"][k_val])

                # # mrr@k
                # for k_val in self.mrr_at_k:
                #     output_data.append(data["mrr@k"][k_val])
                # # ndcg@k
                # for k_val in self.ndcg_at_k:
                #     output_data.append(data["ndcg@k"][k_val])

            # Append loss
            output_data.append(loss if loss is not None else "n/a")

            fOut.write(",".join(map(str, output_data)) + "\n")

    def _get_node_type(self, query: Query) -> str:
        """
        Determines the node type for a query.

        Args:
            query: The query object

        Returns:
            String representing the node type with highest priority
        """
        rtc = self._return_topic_rtc(query.discussion_scenario)

        # Extract categories for each label
        categories = []
        for label in query.labels:
            query_template = rtc.get_template_for_label(label)
            categories.append(getattr(query_template, "category", TemplateCategory.OTHER).name)

        main_arg_labels = {temp.label for temp in rtc.first_level_z_templates}

        for i in range(len(categories)):
            if categories[i] == "Z":
                if query.labels[i] in main_arg_labels:
                    categories[i] = "Main"
                else:
                    categories[i] = "Counter"

        # Define label priority order
        priority = {"OTHER": 0, "NZ": 1, "FAQ": 2, "Counter": 3, "Main": 4}

        # Sort categories by priority
        sorted_categories = sorted(categories, key=lambda x: priority[x])

        # Return highest prio category
        return sorted_categories[0]

    def get_noisy_node_type(self, query: Query) -> str:
        type_to_label = {
            "CONSENT": ["CONSENT"],
            "DISSENT": ["DISSENT"],
            "UNKNOWN_NZ_ARG": ["CON_NZARG", "NZ_ARG", "NZ.G1", "NZ.G2", "NZ.G3", "PRO_NZARG"],
            "UNKNOWN_Z_ARG": ["CON_ZARG", "Z.K3-1-1-1", "Z.P4-1-1", "Z.GP1", "PRO_ZARG", "Z.GK5", "Z.GP74-1"],
            "OTHER": ["OTHER", "FAQ.1-1", "FAQ.6-1", "FAQ.G1"]
        }

        priority = {"DISSENT": 0, "CONSENT": 1, "OTHER": 2, "UNKNOWN_NZ_ARG": 3, "UNKNOWN_Z_ARG": 4}

        # Get labels from query
        labels = query.labels

        # Initialize with lowest priority node type
        assigned_type = None
        assigned_priority = 0

        # Check each label against the type mappings
        for label in labels:
            assigned = False

            # First check explicit mappings
            for node_type, label_list in type_to_label.items():
                if label in label_list:
                    if priority[node_type] > assigned_priority:
                        assigned_type = node_type
                        assigned_priority = priority[node_type]
                    assigned = True
                    break

            # If not explicitly mapped, apply the rules
            if not assigned:
                if "NZ" in label:
                    if priority["UNKNOWN_NZ_ARG"] > assigned_priority:
                        assigned_type = "UNKNOWN_NZ_ARG"
                        assigned_priority = priority["UNKNOWN_NZ_ARG"]
                elif "Z" in label:
                    if priority["UNKNOWN_Z_ARG"] > assigned_priority:
                        assigned_type = "UNKNOWN_Z_ARG"
                        assigned_priority = priority["UNKNOWN_Z_ARG"]
                else:
                    if priority["OTHER"] > assigned_priority:
                        assigned_type = "OTHER"
                        assigned_priority = priority["OTHER"]

        # If no type was assigned, default to OTHER
        if assigned_type is None:
            assigned_type = "OTHER"

        return assigned_type

    def _log_qualitative_error_analysis_table(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]],
                                              noisy_queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) \
            -> None:
        """
        Logs an error analysis table to wandb containing a row for each query.
        The table includes anchor labels and text, top1-label/text/similarity, top10-rank/label/text/similarity, top1-prediction-match, rank of first correct relevant text.
        """

        if not self.run:
            return

        def get_rank_of_first_relevant(relevant_docs: Set[str], hits: List[Tuple[float, str]]) -> int:
            """
            Returns the rank of the first relevant document in the list of hits.

            Args:
                relevant_docs: Set of relevant document IDs
                hits: List of (similarity_score, corpus_id) tuples

            Returns:
                Rank of first relevant document (1-indexed) or -1 if none found
            """
            for rank, (_, cid) in enumerate(hits, start=1):
                if cid in relevant_docs:
                    return rank
            return -1

        def find_threshold_value(similarity: float) -> any:
            """
            Determines the confidence threshold bracket for a similarity score.

            Args:
                similarity: The similarity score to evaluate

            Returns:
                String describing the threshold bracket ("lower", a threshold value, or "higher")
            """
            confidences = self._generate_confidence_thresholds()

            # If below the lowest threshold
            if similarity < confidences[0]:
                return "lower"

            # Find the first threshold that the similarity is below
            for c in confidences:
                if similarity < c:
                    return str(round(c, 2))

            # If above all thresholds
            return "higher"

        def format_top_predictions(hits: List[Tuple[float, str]]) -> List[str]:
            """
            Formats the top predictions into readable strings.

            Args:
                hits: List of (similarity_score, corpus_id) tuples

            Returns:
                List of formatted strings with rank, label, text and similarity score
            """
            top_predictions = []
            for rank, (sim, cid) in enumerate(hits, start=1):
                passage = self.corpus_map[cid]
                top_predictions.append(f"{rank}//{passage.label}//{passage.text}//{sim}")
            return top_predictions

        # Iterate over each score function
        for score_func_name, per_query_hits in queries_result_list.items():
            data = []
            header = [
                "anchor_labels", "anchor_text", "num_anchor_labels", "anchor_node_type",
                "top1_similarity", "top1_label", "top1_text", "top10",
                "top1_prediction_correct", "rank_first_relevant"
            ]

            # Iterate over each query and its predicted similarities
            for q_idx, hits in enumerate(per_query_hits):
                query = self.queries[q_idx]

                # Get data about top1 prediction
                top1_similarity, top1_cid = hits[0]
                top1_passage = self.corpus_map[top1_cid]
                top1_prediction_correct = top1_passage.id in self.relevant_docs[query.id]

                # Format top 10 tuples for display
                top10_tuples = format_top_predictions(hits[:10])

                # Get rank of first relevant passage
                rank_first_relevant = get_rank_of_first_relevant(self.relevant_docs[query.id], hits)

                data.append([
                    query.labels,
                    query.text,
                    len(query.labels),
                    self._get_node_type(query),
                    top1_similarity,
                    top1_passage.label,
                    top1_passage.text,
                    top10_tuples,
                    top1_prediction_correct,
                    rank_first_relevant
                ])

            table = wandb.Table(columns=header, data=data)
            self.run.log({f"{self.name}_{score_func_name}_error_analysis": table})

            # Save to CSV
            if self.save_tables_as_csv:
                csv_dir = os.path.join(self.csv_output_dir, "error_analysis")
                os.makedirs(csv_dir, exist_ok=True)
                filepath = os.path.join(csv_dir, f"{score_func_name}_error_analysis.csv")

                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
                    writer.writerows(data)

                logger.info(f"Saved error analysis to {filepath}")

        if self.noisy_queries:
            # Iterate over each score function

            for score_func_name, per_query_hits in noisy_queries_result_list.items():
                header = [
                    "anchor_labels",
                    "anchor_text",
                    "num_anchor_labels",
                    "anchor_node_type",
                    "top1_similarity",
                    "top1_label",
                    "top1_text",
                    "top10",
                    "first_correct_confidence_threshold"
                ]
                data = []

                # Iterate over each query and its predicted similarities
                for q_idx, hits in enumerate(per_query_hits):
                    query = self.noisy_queries[q_idx]

                    # Get data about top1 prediction
                    top1_similarity, top1_cid = hits[0]
                    top1_passage = self.corpus_map[top1_cid]

                    # Format top 10 predictions for display
                    top10_tuples = format_top_predictions(hits[:10])

                    # Get rank of first relevant passage
                    first_correct_threshold = find_threshold_value(similarity=top1_similarity)

                    data.append([
                        query.labels,
                        query.text,
                        len(query.labels),
                        self.get_noisy_node_type(query),
                        top1_similarity,
                        top1_passage.label,
                        top1_passage.text,
                        top10_tuples,
                        first_correct_threshold
                    ])

                table = wandb.Table(columns=header, data=data)
                self.run.log({f"{self.name}_{score_func_name}_error_analysis_noisy_queries": table})
                if self.save_tables_as_csv:
                    csv_dir = os.path.join(self.csv_output_dir, "error_analysis_noisy_queries")
                    os.makedirs(csv_dir, exist_ok=True)
                    filepath = os.path.join(csv_dir, f"{score_func_name}_error_analysis_noisy_queries.csv")

                    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(header)
                        writer.writerows(data)

                    logger.info(f"Saved error analysis noisy_queries to {filepath}")

    def _log_confusion_matrices_to_wandb(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        """
        Logs confusion matrices for topic, node type, node level and node label predictions.
        """

        def get_labels_for_scenario(scenario: str) -> List[str]:
            """Retrieve unique sorted labels for a given discussion scenario"""
            return sorted(
                {label for query in self.queries if query.discussion_scenario == scenario for label in query.labels})

        # Generate label mappings for each discussion scenario
        topic_labels = {scenario: get_labels_for_scenario(scenario) for scenario in
                        ["MEDAI", "JURAI", "AUTOAI", "REFAI"]}
        topic_mapping = {topic: idx for idx, topic in enumerate(topic_labels.keys())}
        node_type_mapping = {"Z": 0, "NZ": 1, "FAQ": 2, "OTHER": 3}
        node_level_mapping = {"main": 0, "counter": 1}
        node_label_mapping = {topic: {label: idx for idx, label in enumerate(labels)} for topic, labels in
                              topic_labels.items()}

        # Helper functions to retrieve mapping indices with default keys for wrong topics
        def get_node_type_index(node_type: str) -> int:
            return node_type_mapping.get(node_type, max(node_type_mapping.values()) + 1)

        def get_node_level_index(node_level: str) -> int:
            if node_level == "wrong_scenario":
                return node_level_mapping.get(node_level, max(node_level_mapping.values()) + 1)
            elif node_level == "wrong_type":
                return node_level_mapping.get(node_level, max(node_level_mapping.values()) + 2)
            else:
                return node_level_mapping.get(node_level)

        def get_node_label_index(topic: str, label: str) -> int:
            if topic in node_label_mapping:
                return node_label_mapping[topic].get(label, max(node_label_mapping[topic].values()) + 1)
            return -1

        # Iterate over all score functions
        for score_func_name, per_query_hits in queries_result_list.items():
            y_true_topic, y_preds_topic = [], []

            # Process metrics for each topic seperately
            for topic in topic_mapping.keys():
                y_true_type, y_preds_type = [], []
                y_true_level, y_preds_level = [], []
                y_true_label, y_preds_label = [], []
                rtc = self._return_topic_rtc(topic)

                for q_idx, hits in enumerate(per_query_hits):
                    query = self.queries[q_idx]

                    # Skip if query doesn't belong to current topic
                    if query.discussion_scenario != topic:
                        continue

                    # Retrieve top-1 passage for the query
                    passage = self.corpus_map[hits[0][1]]

                    # Store true and predicted topic
                    y_true_topic.append(topic_mapping[topic])
                    y_preds_topic.append(topic_mapping[passage.discussion_scenario])

                    # Process node type, level and label
                    for label in query.labels:
                        query_template = rtc.get_template_for_label(label)
                        query_type = query_template.category.name if query_template else "OTHER"
                        y_true_type.append(get_node_type_index(query_type))
                        y_true_label.append(get_node_label_index(topic, label))

                        if query_template and query_template.label in rtc.arguments_labels:
                            query_level = "counter" if query_template.has_parent_labels else "main"
                            y_true_level.append(get_node_level_index(query_level))

                        # If passage topic doesn't match current topic assign wrong topic to each metric
                        if passage.discussion_scenario != topic:
                            y_preds_type.append(get_node_type_index("wrong_scenario"))
                            y_preds_label.append(get_node_label_index(topic, "wrong_scenario"))
                            if query_template and query_template.label in rtc.arguments_labels:
                                y_preds_level.append(get_node_level_index("wrong_scenario"))
                        else:
                            passage_template = rtc.get_template_for_label(passage.label)
                            passage_type = passage_template.category.name if passage_template else "OTHER"
                            y_preds_type.append(get_node_type_index(passage_type))
                            y_preds_label.append(get_node_label_index(topic, passage.label))

                            if query_template and query_template.label in rtc.arguments_labels:
                                if passage_template and passage_template.label in rtc.arguments_labels:
                                    passage_level = "counter" if passage_template.has_parent_labels else "main"
                                    y_preds_level.append(get_node_level_index(passage_level))
                                else:
                                    y_preds_level.append(get_node_level_index("wrong_type"))

                if self.run:
                    # Calculate absolute and relative confusion matrix data for labels to log them as tables, following the underlying implementation of the wandb lib
                    label_class_names = list(node_label_mapping[topic].keys()) + ["wrong_topic"]
                    n_classes = len(label_class_names)
                    class_idx = list(range(n_classes))

                    if len(y_preds_label) != len(y_true_label):
                        raise ValueError("The number of predictions and true labels must be equal.")

                    if len(set(y_preds_label)) > len(label_class_names):
                        raise ValueError(
                            "The number of unique predicted classes exceeds the number of class names."
                        )

                    if len(set(y_true_label)) > len(label_class_names):
                        raise ValueError(
                            "The number of unique true labels exceeds the number of class names."
                        )

                    class_mapping = {val: i for i, val in enumerate(sorted(list(class_idx)))}
                    counts = np.zeros((n_classes, n_classes))
                    for i in range(len(y_preds_label)):
                        counts[class_mapping[y_true_label[i]], class_mapping[y_preds_label[i]]] += 1

                    data = [
                        [label_class_names[i], label_class_names[j], counts[i, j]]
                        for i in range(n_classes)
                        for j in range(n_classes)
                    ]

                    row_sums = {label: sum(counts[i, :]) for i, label in enumerate(label_class_names)}
                    relative_data = [
                        [label_class_names[i], label_class_names[j],
                         counts[i, j] / row_sums[label_class_names[i]] if row_sums[label_class_names[i]] > 0 else 0]
                        for i in range(n_classes)
                        for j in range(n_classes)
                    ]

                    # Log confusion matrices
                    self.run.log({
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_type": wandb.plot.confusion_matrix(
                            probs=None, y_true=y_true_type, preds=y_preds_type,
                            class_names=list(node_type_mapping.keys()) + ["wrong_topic"],
                            title=f"{score_func_name}: {topic} Node Type Confusion Matrix"),
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_level": wandb.plot.confusion_matrix(
                            probs=None, y_true=y_true_level, preds=y_preds_level,
                            class_names=list(node_level_mapping.keys()) + ["wrong_topic"] + ["wrong_type"],
                            title=f"{score_func_name}: {topic} Node Level Confusion Matrix"),
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_labels_abs": wandb.Table(
                            columns=["Actual", "Predicted", "nPredictions"], data=data),
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_labels_rel": wandb.Table(
                            columns=["Actual", "Predicted", "nPredictions"], data=relative_data)
                    })

            if self.run:
                self.run.log({
                    f"{self.name}_{score_func_name}_confusion_topics": wandb.plot.confusion_matrix(probs=None,
                                                                                                   y_true=y_true_topic,
                                                                                                   preds=y_preds_topic,
                                                                                                   class_names=list(
                                                                                                       topic_mapping.keys()),
                                                                                                   title=f"{score_func_name}: Topic Confusion Matrix"),
                })

    def _compute_stance_accuracy(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> Dict[
        str, float]:
        """
        Computes stance accuracy for different groupings (topic, type, level) based on query results. The stance for the topic grouping is considered correct
        if the stance of the top1-passage is the same as the stance of any label of the query passage. For type and level it is calculated for each label in the query.
        """
        if not self.run:
            return {}

        def get_stance(template: ResponseTemplate | None) -> Stance:
            """Extracts the stance from a response template, defaulting to OTHER if None"""
            return template.stance if template else Stance.OTHER

        accuracy_metrics = {}
        final_data = {}

        for score_func_name, per_query_hits in queries_result_list.items():
            # Initialize accuracy tracking for different categorizations
            metrics = {
                "topic": defaultdict(lambda: {"total": 0, "correct": 0}),
                "type": defaultdict(lambda: {"total": 0, "correct": 0}),
                "level": defaultdict(lambda: {"total": 0, "correct": 0}),
            }

            for q_idx, hits in enumerate(per_query_hits):
                query = self.queries[q_idx]
                passage = self.corpus_map[hits[0][1]]

                discussion_scenario = query.discussion_scenario
                rtc_q = self._return_topic_rtc(discussion_scenario)
                rtc_p = self._return_topic_rtc(passage.discussion_scenario)

                # Get stances for query and passage
                query_stances = [get_stance(rtc_q.get_template_for_label(label)) for label in query.labels]
                passage_stance = get_stance(rtc_p.get_template_for_label(passage.label))

                # Update stance accuracy per topic
                update_accuracy(metrics["topic"], f"{discussion_scenario}", passage_stance in query_stances)

                for label in query.labels:
                    query_template = rtc_q.get_template_for_label(label)
                    query_stance = get_stance(query_template)

                    # Update stance accuracy per node type
                    update_accuracy(
                        metrics["type"],
                        f"{discussion_scenario}_type_{getattr(query_template, 'category', TemplateCategory.OTHER).name}",
                        query_stance == passage_stance
                    )

                    # Skip node level if query is not an argument
                    if not query_template or query_template.label not in rtc_q.arguments_labels:
                        continue

                    # Determine argument level and update stance accuracy per node level
                    arg_level = "counter" if query_template.has_parent_labels else "main"
                    update_accuracy(metrics["level"], f"{discussion_scenario}_level_{arg_level}",
                                    query_stance == passage_stance)

            accuracy_metrics[score_func_name] = metrics

        # Compute final accuracy values
        for score_func_name, categories in accuracy_metrics.items():
            for category, values in categories.items():
                for key, value in values.items():
                    final_data[f"{score_func_name}_stance_accuracy_{key}"] = calculate_accuracy(value)

        return final_data

    def _compute_accuracies_at_k(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]):
        """
        Computes and logs top-k accuracy metrics across different categories (topic, node type, node level, node label)
        for a set of query results generated by various scoring functions.

        Accuracy is calculated based on whether at least one relevant document appears in the top-k hits for each query.
        """
        """Modified version of _compute_accuracies_at_k to also save tables as CSVs and log overall metrics to Hugging Face."""
        if not self.run and not self.save_tables_as_csv and not self.log_to_huggingface:
            return

        # Store overall metrics across all scoring functions for Hugging Face
        overall_metrics = {}

        # Iterate over each scoring function and their corresponding query results
        for score_func_name, per_query_hits in queries_result_list.items():
            # Initialize nested accuracy tracking dicts for each k value
            accuracy_results_by_k = {k: {
                "topic": defaultdict(lambda: {"total": 0, "correct": 0}),
                "node_type": defaultdict(lambda: {"total": 0, "correct": 0}),
                "node_level": defaultdict(lambda: {"total": 0, "correct": 0}),
                "node_label": defaultdict(lambda: {"total": 0, "correct": 0})
            } for k in self.accuracy_at_k}

            # Track overall accuracy for each k
            overall_accuracy_by_k = {k: {"total": 0, "correct": 0} for k in self.accuracy_at_k}

            # Iterate over the results for each query
            for q_idx, hits in enumerate(per_query_hits):
                query = self.queries[q_idx]
                discussion_scenario = query.discussion_scenario
                rtc = self._return_topic_rtc(discussion_scenario)

                # Evaluate accuracy at different k's
                for k in self.accuracy_at_k:
                    # Check if any of the top-k documents are relevant
                    relevant_hit = any(doc_id in self.relevant_docs[query.id] for _, doc_id in hits[:k])

                    # Update overall accuracy
                    overall_accuracy_by_k[k]["total"] += 1
                    if relevant_hit:
                        overall_accuracy_by_k[k]["correct"] += 1

                    # Update topic level accuracy
                    update_accuracy(accuracy_results_by_k[k]["topic"], discussion_scenario, relevant_hit)

                    # For each label in the query, update the corresponding accuracy categories
                    for label in query.labels:
                        query_template = rtc.get_template_for_label(label)

                        # Get category of the node
                        category = getattr(query_template, "category", TemplateCategory.OTHER)
                        update_accuracy(accuracy_results_by_k[k]["node_type"],
                                        f"{discussion_scenario}_type_{category.name}", relevant_hit)

                        # Determine node level (main vs counter) based on template structure
                        if query_template and query_template.label in rtc.arguments_labels:
                            level = "counter" if query_template.has_parent_labels else "main"
                            update_accuracy(accuracy_results_by_k[k]["node_level"],
                                            f"{discussion_scenario}_level_{level}", relevant_hit)

                        # Update accuracy by label
                        update_accuracy(accuracy_results_by_k[k]["node_label"], f"{discussion_scenario}_label_{label}",
                                        relevant_hit)

            # Calculate overall accuracy values
            overall_accuracy_values = {k: calculate_accuracy(overall_accuracy_by_k[k]) for k in self.accuracy_at_k}

            # Save overall accuracy to Hugging Face metrics
            for k, acc in overall_accuracy_values.items():
                # Add to overall metrics dict that will be returned for model card
                overall_metrics[f"{score_func_name}_accuracy@{k}"] = acc

            # Prepare table headers for logging
            accuracies_at_k = [f"accuracy_at_{k}" for k in self.accuracy_at_k]
            first_k = self.accuracy_at_k[0]

            # Initialize wandb tables to hold results for each category
            tables = {
                "topic": wandb.Table(columns=["topic", *accuracies_at_k, "num_queries", "correct_predictions"]),
                "node_type": wandb.Table(
                    columns=["topic", "node_type", *accuracies_at_k, "num_queries", "correct_predictions"]),
                "node_level": wandb.Table(
                    columns=["topic", "node_level", *accuracies_at_k, "num_queries", "correct_predictions"]),
                "node_label": wandb.Table(
                    columns=["topic", "node_label", *accuracies_at_k, "num_queries", "correct_predictions"])
            }

            # Populate each table with the retrieved data
            for category in ["topic", "node_type", "node_level", "node_label"]:
                for key in accuracy_results_by_k[first_k][category]:
                    split_keys = key.split("_", 2)
                    data = [split_keys[0], split_keys[2]] if len(split_keys) == 3 else [key]
                    data += [calculate_accuracy(accuracy_results_by_k[k][category][key]) for k in self.accuracy_at_k]
                    data.append(accuracy_results_by_k[first_k][category][key]["total"])
                    data.append(accuracy_results_by_k[first_k][category][key]["correct"])
                    tables[category].add_data(*data)

            # Log to wandb
            if self.run:
                self.run.log({
                    f"{self.name}_{score_func_name}_topic_accuracy": tables["topic"],
                    f"{self.name}_{score_func_name}_topic_type_accuracy": tables["node_type"],
                    f"{self.name}_{score_func_name}_topic_level_accuracy": tables["node_level"],
                    f"{self.name}_{score_func_name}_topic_label_accuracy": tables["node_label"],
                    # Log overall metrics to wandb too
                    **{f"{self.name}_{k}": v for k, v in overall_accuracy_values.items()}
                })

            # Save tables to CSV
            if self.save_tables_as_csv:
                for category, table in tables.items():
                    filename = f"{score_func_name}_{category}_accuracy"
                    self.save_wandb_table_to_csv(table, filename, prefix="accuracy")

                # Also save overall metrics to CSV
                self.save_overall_metrics_to_csv(f"{score_func_name}_overall_accuracy", overall_accuracy_values)

        return overall_metrics

    def _compute_precision_at_k(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]):
        """
        Computes and logs precision@K metrics across different categories (topic, node type, node level, node label)
        for a set of query results generated by various scoring functions.

        Precision@k is the number of relevant documents in the top-k retrieved results, divided by k.
        """
        if not self.run and not self.save_tables_as_csv and not self.log_to_huggingface:
            return

            # Store overall metrics across all scoring functions for Hugging Face
        overall_metrics = {}

        # Iterate over each scoring function's results
        for score_func_name, per_query_hits in queries_result_list.items():
            # Initialize precision tracking structures for each k
            precision_results_by_k = {k: {
                "topic": defaultdict(lambda: {"total": 0, "correct": 0}),
                "node_type": defaultdict(lambda: {"total": 0, "correct": 0}),
                "node_level": defaultdict(lambda: {"total": 0, "correct": 0}),
                "node_label": defaultdict(lambda: {"total": 0, "correct": 0})
            } for k in self.precision_at_k}

            # Track overall precision for each k
            overall_precision_by_k = {k: {"total": 0, "correct": 0} for k in self.precision_at_k}

            # Loop over each query and its corresponding hits
            for q_idx, hits in enumerate(per_query_hits):
                query = self.queries[q_idx]
                discussion_scenario = query.discussion_scenario
                rtc = self._return_topic_rtc(discussion_scenario)

                # Compute precision at each cutoff value k
                for k in self.precision_at_k:
                    # Count how many of the top-k hits are relevant
                    relevant_hit_count = sum(doc_id in self.relevant_docs[query.id] for _, doc_id in hits[:k])

                    # Update overall precision
                    overall_precision_by_k[k]["total"] += k
                    overall_precision_by_k[k]["correct"] += relevant_hit_count

                    # Update topic-level precision
                    update_precision(precision_results_by_k[k]["topic"], discussion_scenario, k, relevant_hit_count)

                    # Update other precision categories using the query's labels
                    for label in query.labels:
                        query_template = rtc.get_template_for_label(label)
                        category = getattr(query_template, "category", TemplateCategory.OTHER)

                        # Update precision by node type
                        update_precision(precision_results_by_k[k]["node_type"],
                                         f"{discussion_scenario}_type_{category.name}", k, relevant_hit_count)

                        # Update precision by node level
                        if query_template and query_template.label in rtc.arguments_labels:
                            level = "counter" if query_template.has_parent_labels else "main"
                            update_precision(precision_results_by_k[k]["node_level"],
                                             f"{discussion_scenario}_level_{level}", k, relevant_hit_count)

                        # Update precision by label
                        update_precision(precision_results_by_k[k]["node_label"],
                                         f"{discussion_scenario}_label_{label}", k, relevant_hit_count)

            # Calculate overall precision values
            overall_precision_values = {k: calculate_accuracy(overall_precision_by_k[k]) for k in self.precision_at_k}

            # Save overall precision to metrics
            for k, prec in overall_precision_values.items():
                overall_metrics[f"{score_func_name}_precision@{k}"] = prec

            # Prepare wandb table column headers
            precisions_at_k = [f"precision_at_{k}" for k in self.precision_at_k]
            first_k = self.precision_at_k[0]

            # Initialize wandb Tables for each metric category
            tables = {
                "topic": wandb.Table(columns=["topic", *precisions_at_k]),
                "node_type": wandb.Table(columns=["topic", "node_type", *precisions_at_k]),
                "node_level": wandb.Table(columns=["topic", "node_level", *precisions_at_k]),
                "node_label": wandb.Table(columns=["topic", "node_label", *precisions_at_k])
            }

            # Populate each table with precision values across all keys
            for category in ["topic", "node_type", "node_level", "node_label"]:
                for key in precision_results_by_k[first_k][category]:
                    split_keys = key.split("_", 2)
                    data = [split_keys[0], split_keys[2]] if len(split_keys) == 3 else [key]
                    data += [calculate_accuracy(precision_results_by_k[k][category][key]) for k in self.precision_at_k]
                    tables[category].add_data(*data)

            if self.run:
                self.run.log({
                    f"{self.name}_{score_func_name}_topic_precision": tables["topic"],
                    f"{self.name}_{score_func_name}_topic_type_precision": tables["node_type"],
                    f"{self.name}_{score_func_name}_topic_level_precision": tables["node_level"],
                    f"{self.name}_{score_func_name}_topic_label_precision": tables["node_label"],
                    # Log overall metrics to wandb too
                    **{f"{self.name}_{k}": v for k, v in overall_precision_values.items()}
                })

            # Save tables to CSV
            if self.save_tables_as_csv:
                for category, table in tables.items():
                    filename = f"{score_func_name}_{category}_precision"
                    self.save_wandb_table_to_csv(table, filename, prefix="precision")

                # Also save overall metrics to CSV
                self.save_overall_metrics_to_csv(f"{score_func_name}_overall_precision", overall_precision_values)

        return overall_metrics

    def save_overall_metrics_to_csv(self, metric_name, metrics_dict):
        """Save overall metrics to a CSV file."""
        if not self.save_tables_as_csv:
            return

        # Create output directory
        csv_dir = os.path.join(self.csv_output_dir, "overall_metrics")
        os.makedirs(csv_dir, exist_ok=True)

        # Create CSV file
        filepath = os.path.join(csv_dir, f"{metric_name}.csv")
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            for k, v in metrics_dict.items():
                writer.writerow([k, v])

        logger.info(f"Saved overall metrics to {filepath}")

    def _generate_confidence_thresholds(self) -> Iterable[float]:
        """Generate confidence thresholds based on step size"""
        if self.confidence_threshold_steps:
            confidences = np.arange(
                self.confidence_threshold, 1.0 + self.confidence_threshold_steps, self.confidence_threshold_steps
            )
            return confidences[confidences <= 1.0]
        return [self.confidence_threshold]

    def _log_single_argument_classification_with_similarity_threshold(
            self,
            queries_result_list: Dict[str, List[List[Tuple[float, str]]]],
            noisy_queries_result_list: Dict[str, List[List[Tuple[float, str]]]]
    ) -> None:
        """
        Logs top-1 classification accuracy at various confidence thresholds.
        """
        confidences = self._generate_confidence_thresholds()

        # Initialize query counts by threshold tracking
        self.query_counts_by_threshold = {}

        def compute_accuracy(
                result_list: Dict[str, List[List[Tuple[float, str]]]],
                confidences: Iterable[float],
                query_list: list,
                handle_noisy: bool = False
        ) -> Dict[str, Dict[float, float]]:
            """
            Computes accuracy at different confidence thresholds and tracks queries above threshold

            Args:
                result_list: Query results grouped by scoring function
                confidences: List of confidence thresholds to evaluate
                query_list: Corresponding queries for the results
                handle_noisy: Whether to apply special logic for noisy queries
                              (for noisy queries without relevant docs, prediction
                               is considered correct if confidence is below threshold)

            Returns:
                Nested dictionary mapping score functions to {threshold: accuracy} pairs
            """
            results = {}
            query_counts = {}

            for score_func, per_query_hits in result_list.items():
                results[score_func] = {}
                query_counts[score_func] = {}

                for confidence in confidences:
                    top1_hits = 0
                    total_queries = len(per_query_hits)
                    queries_above_threshold = 0

                    for q_idx, hits in enumerate(per_query_hits):
                        query = query_list[q_idx]
                        top1 = hits[0]

                        # Track queries with top1 prediction above threshold
                        if top1[0] >= confidence:
                            queries_above_threshold += 1

                        if handle_noisy and not self.relevant_docs.get(query.id, []):
                            # If query is noisy and top 1 prediction confidence is below threshold count it as correct
                            if top1[0] < confidence:
                                top1_hits += 1
                        elif top1[0] >= confidence:
                            # Check if the top-1 passage meets confidence threshold and is relevant to query
                            top1_passage = self.corpus_map[top1[1]]
                            if top1_passage.id in self.relevant_docs.get(query.id):
                                top1_hits += 1

                    # Compute accuracy as the fraction of correctly classified queries
                    results[score_func][confidence] = top1_hits / total_queries if total_queries else 0.0
                    # Store count of queries above threshold
                    query_counts[score_func][confidence] = queries_above_threshold

                # Add query counts to the class tracking dict
                self.query_counts_by_threshold[score_func] = query_counts[score_func]

            return results

        # SECTION 1: REGULAR QUERIES ANALYSIS
        # Compute and log accuracy for normal queries
        results = compute_accuracy(queries_result_list, confidences, self.queries)
        self._log_results_as_line_plot(
            results,
            "single_argument_classification_normal_queries",
            "Top1-Prediction Accuracy By Confidence Threshold (Normal Queries)"
        )

        # Split queries into single-label and multi-label groups for separate analysis
        single_label_queries, single_label_queries_result_list, multi_label_queries, multi_label_queries_result_list = self._divide_queries_and_predictions_by_label_count(
            queries_result_list)

        # Compute accuracy metrics for single-label queries only
        single_results = compute_accuracy(single_label_queries_result_list, confidences, single_label_queries)
        self._log_results_as_line_plot(
            single_results,
            "single_argument_classification_normal_queries_single_label",
            "Top1-Prediction Accuracy By Confidence Threshold (Normal Queries Single Label Only)"
        )

        # Compute accuracy metrics for multi-label queries only
        multi_results = compute_accuracy(multi_label_queries_result_list, confidences, multi_label_queries)
        self._log_results_as_line_plot(
            multi_results,
            "single_argument_classification_normal_queries_multi_label",
            "Top1-Prediction Accuracy By Confidence Threshold (Normal Queries Multi Label Only)"
        )

        # SECTION 2: NOISY QUERIES ANALYSIS (if available)
        if self.noisy_queries:
            if queries_result_list.keys() != noisy_queries_result_list.keys():
                raise ValueError("Dicts do not have similar score functions")

            # Merge normal and noise query results for combined evaluation
            merged_results_list = {k: queries_result_list[k] + noisy_queries_result_list[k] for k in
                                   queries_result_list.keys()}

            for result_list, q_list, q_type, q_name in [
                (noisy_queries_result_list, self.noisy_queries, "noisy", "Noisy"),
                (merged_results_list, self.queries + self.noisy_queries, "merged", "Normal + Noisy")
            ]:
                # Compute and log accuracy for noisy and merged queries
                results = compute_accuracy(result_list, confidences, q_list, handle_noisy=True)
                self._log_results_as_line_plot(
                    results,
                    f"single_argument_classification_{q_type}_queries",
                    f"Top1-Prediction Accuracy By Confidence Threshold ({q_name} Queries)"
                )

    def _log_multi_argument_classification_with_similarity_threshold(self, queries_result_list: Dict[
        str, List[List[Tuple[float, str]]]], noisy_queries_result_list: Dict[
        str, List[List[Tuple[float, str]]]]) -> None:
        """
        Logs multi argument classification accuracy at various confidence thresholds.
        """
        # Define confidence thresholds based on step size
        confidences = self._generate_confidence_thresholds()

        # Initialize query counts by threshold tracking
        self.query_counts_by_threshold = {}

        def compute_accuracy(
                results_list: Dict[str, List[List[Tuple[float, str]]]],
                confidences: Iterable[float],
                query_list: list,
                handle_noisy: bool = False
        ) -> Tuple[
            Dict[str, Dict[float, float]],
            Dict[str, Dict[float, float]],
            Dict[str, Dict[float, float]]]:
            """
            Computes three types of accuracy metrics at different confidence thresholds.

            Args:
                results_list: Dictionary of query results by scoring function
                confidences: List of confidence thresholds to evaluate
                query_list: List of query objects corresponding to the results
                handle_noisy: Whether to apply special handling for noisy queries
                             (count as match if no predictions above threshold)

            Returns:
                Tuple of four dictionaries (exact_match, partial_match, true_partial_match)
                dictionaries map score functions to {threshold: accuracy} pairs
                Fourth dictionary maps score functions to {threshold: query_count} pairs
            """
            # Initialize accuracy tracking dicts
            results_exact = {}
            results_partial = {}
            results_true_partial = {}
            query_counts = {}

            for score_func, per_query_hits in results_list.items():
                # Init nested dicts for each scoring function
                results_exact[score_func] = {}
                results_partial[score_func] = {}
                results_true_partial[score_func] = {}
                query_counts[score_func] = {}

                for confidence in confidences:
                    total_queries = len(per_query_hits)
                    exact_match, partial_match, true_partial_match = 0, 0, 0
                    queries_above_threshold = 0

                    for q_idx, hits in enumerate(per_query_hits):
                        query = query_list[q_idx]
                        # Get all passages relevant to this query
                        relevant_passages = [self.corpus_map[pid] for pid in self.relevant_docs.get(query.id, [])]

                        # Filter hits that meet confidence threshold
                        hits_above_threshold = [hit for hit in hits if hit[0] >= confidence]

                        # Count queries with at least one hit above threshold
                        if hits_above_threshold:
                            queries_above_threshold += 1

                        if handle_noisy and not hits_above_threshold:
                            # For noisy queries with no predictions above threshold we count all metrics as correct
                            partial_match += 1
                            exact_match += 1
                            true_partial_match += 1
                        elif hits_above_threshold:
                            # Get passages for predictions above threshold
                            passages_above_threshold = [self.corpus_map[p[1]] for p in hits_above_threshold]

                            # Create unique identifiers for passages by combining discussion scenario and label
                            # This allows set-based comparison of prediction results
                            relevant_passage_strings = {f"{p.discussion_scenario}_{p.label}" for p in relevant_passages}
                            threshold_passage_strings = {f"{p.discussion_scenario}_{p.label}" for p in
                                                         passages_above_threshold}

                            # Increase partial match if retrieved passages are subset of relevant passages (labels + discussion scenario)
                            if threshold_passage_strings.issubset(relevant_passage_strings):
                                partial_match += 1

                                # Increase true partial match if retrieved passages are a subset but not equal to relevant passages (labels + discussion scenario)
                                if not threshold_passage_strings == relevant_passage_strings:
                                    true_partial_match += 1

                            # Increase exact match if retrieved passages match exactly relevant passages (labels + discussion scenario)
                            if threshold_passage_strings == relevant_passage_strings:
                                exact_match += 1

                    # Calculate accuracy scores
                    exact_acc = exact_match / total_queries if total_queries > 0 else 0.0
                    partial_acc = partial_match / total_queries if total_queries > 0 else 0.0
                    true_partial_acc = true_partial_match / total_queries if total_queries > 0 else 0.0

                    # Store results for current confidence threshold
                    results_exact[score_func][confidence] = exact_acc
                    results_partial[score_func][confidence] = partial_acc
                    results_true_partial[score_func][confidence] = true_partial_acc
                    query_counts[score_func][confidence] = queries_above_threshold

                # Store query counts in the class tracking dict
                self.query_counts_by_threshold[score_func] = query_counts[score_func]

            return results_exact, results_partial, results_true_partial

        # SECTION 1: REGULAR QUERIES ANALYSIS
        # Compute metrics for all normal queries
        results_exact, results_partial, results_true_partial = compute_accuracy(queries_result_list, confidences,
                                                                                self.queries)

        # Split queries into single-label and multi-label groups for separate analysis
        single_label_queries, single_label_queries_result_list, multi_label_queries, multi_label_queries_result_list = self._divide_queries_and_predictions_by_label_count(
            queries_result_list)

        # Compute metrics for single-label queries
        single_results_exact, single_results_partial, single_results_true_partial = compute_accuracy(
            single_label_queries_result_list, confidences, single_label_queries)

        # Compute metrics for multi-label queries
        multi_results_exact, multi_results_partial, multi_results_true_partial = compute_accuracy(
            multi_label_queries_result_list, confidences, multi_label_queries)

        # Log results to wandb
        for exact, partial, true_partial, metric_name_suffix, metric_title_type in [
            (results_exact, results_partial, results_true_partial, "", "(Normal Queries)"),
            (single_results_exact, single_results_partial, single_results_true_partial, "_single_label",
             "(Normal Queries Single Label Only)"),
            (multi_results_exact, multi_results_partial, multi_results_true_partial, "_multi_label",
             "(Normal Queries Multi Label Only")
        ]:
            # Always log exact match metric for all query types
            self._log_results_as_line_plot(exact, f"multi_argument_classification_exact_match{metric_name_suffix}",
                                           f"Exact Match Accuracy By Confidence Threshold {metric_title_type}")

            # Only log partial and true partial matches for all queries and multi-label queries
            # These metrics are not meaningful for single-label queries as true partial is always 0 and partial matches exact match accuracy
            if "_single" not in metric_name_suffix:
                self._log_results_as_line_plot(partial,
                                               f"multi_argument_classification_partial_match{metric_name_suffix}",
                                               f"Partial Match Accuracy By Confidence Threshold {metric_title_type}")
                self._log_results_as_line_plot(true_partial,
                                               f"multi_argument_classification_true_partial_match{metric_name_suffix}",
                                               f"True Partial Match Accuracy By Confidence Threshold {metric_title_type}")

        # SECTION 4: ANALYZE NOISY QUERIES (IF AVAILABLE)
        if self.noisy_queries:
            if queries_result_list.keys() != noisy_queries_result_list.keys():
                raise ValueError("Dicts do not have similar score functions")

            # Merge normal and noise query results for combined evaluation
            merged_results_list = {k: queries_result_list[k] + noisy_queries_result_list[k] for k in
                                   queries_result_list.keys()}
            for result_list, q_list, q_type, q_name in [
                (noisy_queries_result_list, self.noisy_queries, "noisy", "Noisy"),
                (merged_results_list, self.queries + self.noisy_queries, "merged", "Normal + Noisy")
            ]:
                # Compute and log accuracy for noisy and merged queries
                results_exact, results_partial, results_true_partial = compute_accuracy(result_list, confidences,
                                                                                        q_list, handle_noisy=True)
                self._log_results_as_line_plot(results_exact, f"multi_argument_classification_exact_match_{q_type}",
                                               f"Exact Match Accuracy By Confidence Threshold ({q_name} Queries)")
                self._log_results_as_line_plot(results_partial, f"multi_argument_classification_partial_match_{q_type}",
                                               f"Partial Match Accuracy By Confidence Threshold ({q_name} Queries)")
                self._log_results_as_line_plot(results_true_partial,
                                               f"multi_argument_classification_true_partial_match_{q_type}",
                                               f"True Partial Match Accuracy By Confidence Threshold ({q_name} Queries)")



    def _log_results_as_line_plot(self, results: Dict[str, Dict], metric_name: str, title: str) -> None:
        """
        Logs line plot data to wandb and saves it as CSV.

        Also tracks the number of queries that remain above each confidence threshold.
        """
        for score_func_name, confidence_threshold in results.items():
            # Calculate total queries above each threshold (if available in results)
            if hasattr(self, 'query_counts_by_threshold') and score_func_name in self.query_counts_by_threshold:
                query_counts = self.query_counts_by_threshold[score_func_name]
                data = [[c, acc, query_counts.get(c, 0)] for c, acc in confidence_threshold.items()]
                columns = ["Confidence", "Accuracy", "Queries_Above_Threshold"]
            else:
                data = [[c, acc] for c, acc in confidence_threshold.items()]
                columns = ["Confidence", "Accuracy"]

            table = wandb.Table(data=data, columns=columns)

            if self.run:
                self.run.log({f"{self.name}_{score_func_name}_{metric_name}": wandb.plot.line(
                    table, "Confidence", "Accuracy", title=title)})

            # Save to CSV
            if self.save_tables_as_csv:
                csv_dir = os.path.join(self.csv_output_dir, "confidence_threshold_metrics")
                os.makedirs(csv_dir, exist_ok=True)
                filepath = os.path.join(csv_dir, f"{score_func_name}_{metric_name}.csv")

                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(columns)
                    writer.writerows(data)

                logger.info(f"Saved confidence threshold metrics to {filepath}")

    # Modified _evaluate_noisy_query_performance_by_type to use confidence_threshold_steps
    def _evaluate_noisy_query_performance_by_type(self, noisy_queries_result_list):
        """
        Evaluates performance metrics for each type of noisy query.

        This function calculates:
        1. Accuracy@k for each noisy query type
        2. False positive rates at different confidence thresholds
        """
        if not self.noisy_queries:
            logger.info("No noisy queries available for type-specific performance analysis")
            return

        # Get all confidence thresholds based on steps
        confidences = self._generate_confidence_thresholds()

        # Group noisy queries by type
        queries_by_type = {
            "CONSENT": [],
            "DISSENT": [],
            "UNKNOWN_NZ_ARG": [],
            "UNKNOWN_Z_ARG": [],
            "OTHER": []
        }

        query_indices_by_type = {
            "CONSENT": [],
            "DISSENT": [],
            "UNKNOWN_NZ_ARG": [],
            "UNKNOWN_Z_ARG": [],
            "OTHER": []
        }

        for q_idx, query in enumerate(self.noisy_queries):
            query_type = self.get_noisy_node_type(query)
            queries_by_type[query_type].append(query)
            query_indices_by_type[query_type].append(q_idx)

        # For each score function
        for score_func_name, per_query_hits in noisy_queries_result_list.items():
            # Create filtered result lists for each type
            type_result_lists = {}
            for qtype, indices in query_indices_by_type.items():
                if indices:  # Only process types that have queries
                    type_result_lists[qtype] = [per_query_hits[idx] for idx in indices]

            # Calculate accuracy@k for each type across all confidence thresholds
            accuracy_data = []

            # For each noisy query type
            for qtype, type_queries in queries_by_type.items():
                if not type_queries:
                    continue

                type_hits = type_result_lists[qtype]
                num_queries = len(type_queries)

                # First, calculate accuracy@k for each confidence threshold and k value
                for confidence in confidences:
                    for k in self.accuracy_at_k:
                        correct_predictions = 0

                        for hits in type_hits:
                            if len(hits) >= k:
                                # For noisy queries, a "correct" prediction is when confidence is below threshold
                                # (i.e., the model correctly realizes it shouldn't predict anything)
                                if hits[0][0] < confidence:
                                    correct_predictions += 1

                        accuracy = correct_predictions / num_queries if num_queries else 0
                        accuracy_data.append([
                            qtype,
                            k,
                            confidence,
                            f"{accuracy:.4f}",
                            correct_predictions,
                            num_queries
                        ])

            # Log accuracy table
            if self.run:
                table = wandb.Table(
                    columns=["Noisy Query Type", "K", "Confidence Threshold", "Accuracy@K",
                             "Correct Predictions", "Number of Queries"],
                    data=accuracy_data
                )
                self.run.log({f"{self.name}_{score_func_name}_noisy_type_accuracy": table})

            # Save to CSV
            if self.save_tables_as_csv:
                csv_dir = os.path.join(self.csv_output_dir, "noisy_query_performance")
                os.makedirs(csv_dir, exist_ok=True)
                filepath = os.path.join(csv_dir, f"{score_func_name}_noisy_type_accuracy.csv")

                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Noisy Query Type", "K", "Confidence Threshold", "Accuracy@K",
                                     "Correct Predictions", "Number of Queries"])
                    writer.writerows(accuracy_data)

                logger.info(f"Saved noisy query type accuracy to {filepath}")

            # Calculate false positive rates at different confidence thresholds
            fp_data = []

            for confidence in confidences:
                for qtype, type_hits in type_result_lists.items():
                    if not type_hits:
                        continue

                    num_queries = len(type_hits)
                    false_positives = 0

                    for hits in type_hits:
                        # Count as false positive if any hit is above threshold
                        if any(hit[0] >= confidence for hit in hits):
                            false_positives += 1

                    fp_rate = false_positives / num_queries if num_queries else 0
                    fp_data.append([
                        qtype,
                        confidence,
                        f"{fp_rate:.4f}",
                        false_positives,
                        num_queries
                    ])

            # Log false positive rate table
            if self.run:
                table = wandb.Table(
                    columns=["Noisy Query Type", "Confidence Threshold", "False Positive Rate",
                             "False Positives", "Number of Queries"],
                    data=fp_data
                )
                self.run.log({f"{self.name}_{score_func_name}_noisy_type_fp_rates": table})

            # Save to CSV
            if self.save_tables_as_csv:
                csv_dir = os.path.join(self.csv_output_dir, "noisy_query_performance")
                os.makedirs(csv_dir, exist_ok=True)
                filepath = os.path.join(csv_dir, f"{score_func_name}_noisy_type_fp_rates.csv")

                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Noisy Query Type", "Confidence Threshold", "False Positive Rate",
                                     "False Positives", "Number of Queries"])
                    writer.writerows(fp_data)

                logger.info(f"Saved noisy query type false positive rates to {filepath}")

    def _log_query_and_passage_embeddings(self):
        """Logs query and corpus embeddings to wandb."""

        def format_query_labels(q: Query) -> str:
            labels = sorted(q.labels)
            return (q.discussion_scenario + "_" + "_".join(labels)).lower()

        def format_passage_label(p: Passage) -> str:
            return (p.discussion_scenario + "_" + p.label).lower()

        def get_text_type(q: Optional[Query] = None, p: Optional[Passage] = None):
            if q:
                rtc = self._return_topic_rtc(q.discussion_scenario)
                categories = [getattr(rtc.get_template_for_label(label), 'category', TemplateCategory.OTHER) for label
                              in q.labels]
                categories.sort(key=lambda e: e.value)
                return f"{q.discussion_scenario}_{categories[0].name}"

            if p:
                rtc = self._return_topic_rtc(p.discussion_scenario)
                return f"{p.discussion_scenario}_{getattr(rtc.get_template_for_label(p.label), 'category', TemplateCategory.OTHER).name}"
            return None

        columns = ["label", "text", "discussion_scenario", "arg_type"] + [f"dim_{i}" for i in
                                                                          range(len(self.query_embeddings[0]))]
        data = ([(format_query_labels(q), q.text, q.discussion_scenario, get_text_type(q=q), *emb) for q, emb in
                 zip(self.queries, self.query_embeddings)] +
                [(format_query_labels(q), q.text, q.discussion_scenario, get_text_type(q=q), *emb) for q, emb in
                 zip(self.noisy_queries, self.noisy_query_embeddings)] +
                [(format_passage_label(p), p.text, p.discussion_scenario, get_text_type(p=p), *emb) for p, emb in
                 zip(self.corpus, self.corpus_embeddings)])

        if self.run:
            self.run.log({"embeddings": wandb.Table(data=data, columns=columns)})

    def _return_topic_rtc(self, discussion_scenario: str) -> ResponseTemplateCollection:
        rtc = self.argument_graphs.get(discussion_scenario)

        if not rtc:
            raise Exception(f"Discussion scenario: {discussion_scenario} not found")

        return rtc

    def _divide_queries_and_predictions_by_label_count(self,
                                                       queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> \
            Tuple[List[Query], Dict[str, List[List[Tuple[float, str]]]], List[Query], Dict[
                str, List[List[Tuple[float, str]]]]]:
        """Splits queries and their scoring results by their label count (singular/multiple)"""
        # Create lists containing indexes of queries with single/multiple labels
        single_label_query_idxs = []
        multi_label_query_idxs = []

        for idx, query in enumerate(self.queries):
            if len(query.labels) == 1:
                single_label_query_idxs.append(idx)
            elif len(query.labels) > 1:
                multi_label_query_idxs.append(idx)
            else:
                raise Exception(f"Query: {query} has less than 1 gold label")

        # Divide queries and their similarity predictions by number of their labels
        single_label_queries = [self.queries[i] for i in single_label_query_idxs]
        multi_label_queries = [self.queries[i] for i in multi_label_query_idxs]
        single_label_queries_result_list = {
            func: [hits for q_idx, hits in enumerate(per_hits) if q_idx in single_label_query_idxs]
            for func, per_hits in queries_result_list.items()
        }
        multi_label_queries_result_list = {
            func: [hits for q_idx, hits in enumerate(per_hits) if q_idx in multi_label_query_idxs]
            for func, per_hits in queries_result_list.items()
        }

        return single_label_queries, single_label_queries_result_list, multi_label_queries, multi_label_queries_result_list

    def _analyze_noisy_query_types(self):
        """
        Analyzes the distribution of noisy query types and creates a table of statistics.
        This can be called during evaluation to understand the composition of noisy queries.
        """
        if not self.noisy_queries:
            logger.info("No noisy queries available for analysis")
            return

        # Count query types
        type_counts = {
            "CONSENT": 0,
            "DISSENT": 0,
            "UNKNOWN_NZ_ARG": 0,
            "UNKNOWN_Z_ARG": 0,
            "OTHER": 0
        }

        # Group queries by type
        queries_by_type = {
            "CONSENT": [],
            "DISSENT": [],
            "UNKNOWN_NZ_ARG": [],
            "UNKNOWN_Z_ARG": [],
            "OTHER": []
        }

        # Analyze each query
        for query in self.noisy_queries:
            query_type = self.get_noisy_node_type(query)
            type_counts[query_type] += 1
            queries_by_type[query_type].append(query)

        # Create analysis table
        data = []
        for qtype, count in type_counts.items():
            percentage = (count / len(self.noisy_queries)) * 100 if self.noisy_queries else 0
            # Get some example texts for each type (up to 3)
            examples = [q.text[:100] + "..." if len(q.text) > 100 else q.text
                        for q in queries_by_type[qtype][:3]]
            examples_str = "\n".join(examples) if examples else "No examples"

            data.append([
                qtype,
                count,
                f"{percentage:.2f}%",
                examples_str
            ])

        # Log to wandb
        if self.run:
            table = wandb.Table(columns=["Noisy Query Type", "Count", "Percentage", "Example Texts"], data=data)
            self.run.log({f"{self.name}_noisy_query_type_distribution": table})

        # Save to CSV
        if self.save_tables_as_csv:
            csv_dir = os.path.join(self.csv_output_dir, "noisy_query_analysis")
            os.makedirs(csv_dir, exist_ok=True)
            filepath = os.path.join(csv_dir, "noisy_query_type_distribution.csv")

            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Noisy Query Type", "Count", "Percentage", "Example Texts"])
                writer.writerows(data)

            logger.info(f"Saved noisy query type distribution to {filepath}")

        # Log summary to console
        logger.info("Noisy Query Type Distribution:")
        for qtype, count in type_counts.items():
            percentage = (count / len(self.noisy_queries)) * 100 if self.noisy_queries else 0
            logger.info(f"  {qtype}: {count} ({percentage:.2f}%)")

        return type_counts


if __name__ == "__main__":
    rtc = load_response_template_collection("s1", "../../")
