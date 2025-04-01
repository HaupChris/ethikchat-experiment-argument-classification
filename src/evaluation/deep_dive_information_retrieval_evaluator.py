from __future__ import annotations

import heapq
import logging
import os
import numpy as np
import bisect
from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Optional, Dict, List, Tuple, Set

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


def calculate_accuracy(data: Dict[str, int]) -> float:
    """Calculates the accuracy for the given dictionary."""
    return data["correct"] / data["total"]


class DeepDiveInformationRetrievalEvaluator(SentenceEvaluator):
    """
    IR evaluator that logs:
      1) Per-topic, per-topic + (node type, node level, and node label) accuracy@K to a W&B table
      2) Per-topic, per-topic + (node type, node level, and node label) stance accuracies as metrics
      3) Confusion matrices over topics, topic + (node type, node level, and node label) with absolute and relative values using W&B's API
      4) A W&B table for qualitative error analysis containing the columns (anchor_labels, anchor_text, top1_similarity, top1_label, top1_text, top5,
      top1_prediction_correct, rank_first_relevant) for each query in the dataset
      5) Graphs displaying single/multi argument classification accuracies depending on a set confidence threshold
      6) Query and Passage Embeddings
    """

    def __init__(
        self,
        queries: Dict[str, any],              # qid => Query object (must have .text, .discussion_scenario, .id)
        corpus: Dict[str, any],               # cid => Passage object (must have .text, .discussion_scenario, .id)
        relevant_docs: Dict[str, Set[str]],   # qid => set of relevant doc IDs
        corpus_chunk_size: int = 50000,
        accuracy_at_k: List[int] = [1, 3, 5],
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
        self.accuracy_at_k = sorted([k for k in accuracy_at_k if k > 0])
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv

        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None

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
        self.corpus_embeddings = []

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

        # 1) Compute similarities for each query
        queries_result_list = self._compute_scores(model, *args, **kwargs)

        # 2) Log qualitative error analysis table
        self._log_qualitative_error_analysis_table(queries_result_list)

        # 3) Log accuracy@k tables
        self._compute_accuracies_at_k(queries_result_list)

        # 4) Compute stance accuracies
        metrics = self._compute_stance_accuracy(queries_result_list)

        # 5) Log confusion matrices
        self._log_confusion_matrices_to_wandb(queries_result_list)

        # 6) Log accuracies based on confidence thresholds
        if self.confidence_threshold:
            self._log_single_argument_classification_with_similarity_threshold(queries_result_list)
            self._log_multi_argument_classification_with_similarity_threshold(queries_result_list)

        # 7) Give these metrics a prefix and store in model card
        final_metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, final_metrics, epoch, steps)

        # 8) Log embeddings
        if self.log_embeddings:
            self._log_query_and_passage_embeddings()

        # 9) Log to W&B (if self.run is set)
        if self.run:
            self.run.log(final_metrics)

        return final_metrics

    def _compute_scores(
        self,
        model: SentenceTransformer,
        corpus_model=None,
        corpus_embeddings: Tensor | None = None
    ) -> Dict[str, List[List[Tuple[float, str]]]]:
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

            self.query_embeddings = query_embeddings.tolist()

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

            self.corpus_embeddings.extend(sub_corpus_emb.tolist())
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

        return queries_result_list

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

    def _log_qualitative_error_analysis_table(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        """
        Logs an error analysis table to wandb containing a row for each query.
        The table includes anchor labels and text, top1-label/text/similarity, top5-rank/label/text/similarity, top1-prediction-match, rank of first correct relevant text.
        """

        if not self.run:
            return

        def get_rank_of_first_relevant(relevant_docs: Set[str], hits: List[Tuple[float, str]]) -> int:
            """
            Returns the rank of the first relevant document based on similarities
            """
            for rank, (_, cid) in enumerate(hits, start=1):
                if cid in relevant_docs:
                    return rank

            return -1

        # Iterate over each score function
        for score_func_name, per_query_hits in queries_result_list.items():
            table = wandb.Table(columns=[
                "anchor_labels",
                "anchor_text",
                "top1_similarity",
                "top1_label",
                "top1_text",
                "top5",
                "top1_prediction_correct",
                "rank_first_relevant"
            ])

            # Iterate over each query and its predicted similarities
            for q_idx, hits in enumerate(per_query_hits):
                query = self.queries[q_idx]

                # Get data about top1 prediction
                top1_similarity, top1_cid = hits[0]
                top1_passage = self.corpus_map[top1_cid]
                top1_prediction_correct = top1_passage.id in self.relevant_docs[query.id]

                # Get the top 5 predictions
                top5 = [(sim, self.corpus_map[cid]) for sim, cid in hits[:5]]
                top5_tuples = [
                    f"{rank}//{passage.label}//{passage.text}//{sim}"
                    for rank, (sim, passage) in enumerate(top5, start=1)
                ]

                # Get rank of first relevant passage
                rank_first_relevant = get_rank_of_first_relevant(self.relevant_docs[query.id], hits)

                table.add_data(
                    query.labels,
                    query.text,
                    top1_similarity,
                    top1_passage.label,
                    top1_passage.text,
                    top5_tuples,
                    top1_prediction_correct,
                    rank_first_relevant
                )

            self.run.log({f"{self.name}_{score_func_name}_error_analysis": table})

    def _log_confusion_matrices_to_wandb(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        """
        Logs confusion matrices for topic, node type, node level and node label predictions.
        """
        def get_labels_for_scenario(scenario: str) -> List[str]:
            """Retrieve unique sorted labels for a given discussion scenario"""
            return sorted({label for query in self.queries if query.discussion_scenario == scenario for label in query.labels})

        # Generate label mappings for each discussion scenario
        topic_labels = {scenario: get_labels_for_scenario(scenario) for scenario in ["MEDAI", "JURAI", "AUTOAI", "REFAI"]}
        topic_mapping = {topic: idx for idx, topic in enumerate(topic_labels.keys())}
        node_type_mapping = {"Z": 0, "NZ": 1, "FAQ": 2, "OTHER": 3}
        node_level_mapping = {"main": 0, "counter": 1}
        node_label_mapping = {topic: {label: idx for idx, label in enumerate(labels)} for topic, labels in topic_labels.items()}

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
                        [label_class_names[i], label_class_names[j], counts[i, j] / row_sums[label_class_names[i]] if row_sums[label_class_names[i]] > 0 else 0]
                        for i in range(n_classes)
                        for j in range(n_classes)
                    ]

                    # Log confusion matrices
                    self.run.log({
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_type": wandb.plot.confusion_matrix(probs=None, y_true=y_true_type, preds=y_preds_type, class_names=list(node_type_mapping.keys()) + ["wrong_topic"], title=f"{score_func_name}: {topic} Node Type Confusion Matrix"),
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_level": wandb.plot.confusion_matrix(probs=None, y_true=y_true_level, preds=y_preds_level, class_names=list(node_level_mapping.keys()) + ["wrong_topic"] + ["wrong_type"], title=f"{score_func_name}: {topic} Node Level Confusion Matrix"),
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_labels_abs": wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_labels_rel": wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=relative_data)
                    })

            if self.run:
                self.run.log({
                    f"{self.name}_{score_func_name}_confusion_topics": wandb.plot.confusion_matrix(probs=None, y_true=y_true_topic, preds=y_preds_topic, class_names=list(topic_mapping.keys()), title=f"{score_func_name}: Topic Confusion Matrix"),
                })

    def _compute_stance_accuracy(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> Dict[str, float]:
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
                    update_accuracy(metrics["level"], f"{discussion_scenario}_level_{arg_level}", query_stance == passage_stance)

            accuracy_metrics[score_func_name] = metrics

        # Compute final accuracy values
        for score_func_name, categories in accuracy_metrics.items():
            for category, values in categories.items():
                for key, value in values.items():
                    final_data[f"{score_func_name}_stance_accuracy_{key}"] = calculate_accuracy(value)

        return final_data

    def _compute_accuracies_at_k(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        if not self.run:
            return

        for score_func_name, per_query_hits in queries_result_list.items():
            accuracy_results_by_k = {k: {
                "topic": defaultdict(lambda: {"total": 0, "correct": 0}),
                "node_type": defaultdict(lambda: {"total": 0, "correct": 0}),
                "node_level": defaultdict(lambda: {"total": 0, "correct": 0}),
                "node_label": defaultdict(lambda: {"total": 0, "correct": 0})
            } for k in self.accuracy_at_k}

            for q_idx, hits in enumerate(per_query_hits):
                query = self.queries[q_idx]
                discussion_scenario = query.discussion_scenario
                rtc = self._return_topic_rtc(discussion_scenario)

                for k in self.accuracy_at_k:
                    relevant_hit = any(doc_id in self.relevant_docs[query.id] for _, doc_id in hits[:k])
                    update_accuracy(accuracy_results_by_k[k]["topic"], discussion_scenario, relevant_hit)

                    for label in query.labels:
                        query_template = rtc.get_template_for_label(label)
                        category = getattr(query_template, "category", TemplateCategory.OTHER)
                        update_accuracy(accuracy_results_by_k[k]["node_type"], f"{discussion_scenario}_type_{category.name}", relevant_hit)

                        if query_template and query_template.label in rtc.arguments_labels:
                            level = "counter" if query_template.has_parent_labels else "main"
                            update_accuracy(accuracy_results_by_k[k]["node_level"], f"{discussion_scenario}_level_{level}", relevant_hit)

                        update_accuracy(accuracy_results_by_k[k]["node_label"], f"{discussion_scenario}_label_{label}", relevant_hit)

            accuracies_at_k = [f"accuracy_at_{k}" for k in self.accuracy_at_k]
            first_k = self.accuracy_at_k[0]

            tables = {
                "topic": wandb.Table(columns=["topic", *accuracies_at_k, "num_queries", "correct_predictions"]),
                "node_type": wandb.Table(columns=["topic", "node_type", *accuracies_at_k, "num_queries", "correct_predictions"]),
                "node_level": wandb.Table(columns=["topic", "node_level", *accuracies_at_k, "num_queries", "correct_predictions"]),
                "node_label": wandb.Table(columns=["topic", "node_label", *accuracies_at_k, "num_queries", "correct_predictions"])
            }

            for category in ["topic", "node_type", "node_level", "node_label"]:
                for key in accuracy_results_by_k[first_k][category]:
                    split_keys = key.split("_", 2)
                    data = [split_keys[0], split_keys[2]] if len(split_keys) == 3 else [key]
                    data += [calculate_accuracy(accuracy_results_by_k[k][category][key]) for k in self.accuracy_at_k]
                    data.append(accuracy_results_by_k[first_k][category][key]["total"])
                    data.append(accuracy_results_by_k[first_k][category][key]["correct"])
                    tables[category].add_data(*data)

            if self.run:
                self.run.log({
                    f"{self.name}_{score_func_name}_topic_accuracy": tables["topic"],
                    f"{self.name}_{score_func_name}_topic_type_accuracy": tables["node_type"],
                    f"{self.name}_{score_func_name}_topic_level_accuracy": tables["node_level"],
                    f"{self.name}_{score_func_name}_topic_label_accuracy": tables["node_label"]
                })

    def _log_single_argument_classification_with_similarity_threshold(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        """
        Logs top-1 classification accuracy at various confidence thresholds.
        """
        # Define confidence thresholds based on step size
        confidences = [self.confidence_threshold]
        if self.confidence_threshold_steps:
            confidences = np.arange(
                self.confidence_threshold, 1.0 + self.confidence_threshold_steps, self.confidence_threshold_steps
            )
            confidences = confidences[confidences <= 1.0]

        # Initialize accuracy tracking dict
        results = {}

        for score_func, per_query_hits in queries_result_list.items():
            results[score_func] = {}

            # Compute accuracy at various confidence thresholds
            for confidence in confidences:
                top1_hits = 0
                total_queries = len(per_query_hits)

                # Evaluate each query
                for q_idx, hits in enumerate(per_query_hits):
                    query = self.queries[q_idx]
                    top1 = hits[0]

                    # Check if top-1 passage meets confidence threshold and is relevant
                    if top1[0] >= confidence:
                        top1_passage = self.corpus_map[top1[1]]
                        if top1_passage.id in self.relevant_docs[query.id]:
                            top1_hits += 1

                # Calculate accuracy
                accuracy = top1_hits / total_queries if total_queries > 0 else 0.0
                results[score_func][confidence] = accuracy

        self._log_results_as_line_plot(results, "single_argument_classification", "Top-1-Prediction Accuracy By Confidence Threshold")

    def _log_multi_argument_classification_with_similarity_threshold(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        """
        Logs multi argument classification accuracy at various confidence thresholds.
        """
        # Define confidence thresholds based on step size
        confidences = [self.confidence_threshold]
        if self.confidence_threshold_steps:
            confidences = np.arange(
                self.confidence_threshold, 1.0 + self.confidence_threshold_steps, self.confidence_threshold_steps
            )
            confidences = confidences[confidences <= 1.0]

        # Initialize accuracy tracking dicts
        results_exact = {}
        results_partial = {}
        results_true_partial = {}

        for score_func, per_query_hits in queries_result_list.items():
            results_exact[score_func] = {}
            results_partial[score_func] = {}
            results_true_partial[score_func] = {}

            # Compute accuracy at different confidence thresholds
            for confidence in confidences:
                total_queries = len(per_query_hits)
                exact_match, partial_match, true_partial_match = 0, 0, 0

                # Evaluate each query
                for q_idx, hits in enumerate(per_query_hits):
                    query = self.queries[q_idx]
                    relevant_passages = [self.corpus_map[pid] for pid in self.relevant_docs[query.id]]

                    # Filter hits based on confidence threshold
                    hits_above_threshold = [hit for hit in hits if hit[0] >= confidence]
                    if hits_above_threshold:
                        passages_above_threshold = [self.corpus_map[p[1]] for p in hits_above_threshold]

                        # Check if all retrieved passages are relevant
                        false_passage = any(passage not in relevant_passages for passage in passages_above_threshold)
                        if not false_passage:
                            relevant_passage_strings = {f"{p.discussion_scenario}_{p.label}" for p in relevant_passages}
                            threshold_passage_strings = {f"{p.discussion_scenario}_{p.label}" for p in passages_above_threshold}

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

                results_exact[score_func][confidence] = exact_acc
                results_partial[score_func][confidence] = partial_acc
                results_true_partial[score_func][confidence] = true_partial_acc

        self._log_results_as_line_plot(results_exact, "multi_argument_classification_exact_match", "Exact Match Accuracy By Confidence Threshold")
        self._log_results_as_line_plot(results_partial, "multi_argument_classification_partial_match", "Partial Match Accuracy By Confidence Threshold")
        self._log_results_as_line_plot(results_partial, "multi_argument_classification_true_partial_match", "True Partial Match Accuracy By Confidence Threshold")

    def _log_results_as_line_plot(self, results: Dict[str, Dict], metric_name: str, title: str) -> None:
        for score_func_name, confidence_threshold in results.items():
            data = [[c, acc] for c, acc in confidence_threshold.items()]
            table = wandb.Table(data=data, columns=["Confidence", "Accuracy"])
            if self.run:
                self.run.log({f"{self.name}_{score_func_name}_{metric_name}": wandb.plot.line(table, "Confidence", "Accuracy", title=title)})

    def _log_query_and_passage_embeddings(self):
        """Logs query and corpus embeddings to wandb."""
        def format_query_labels(q: Query) -> str:
            labels = sorted(q.labels)
            return (q.discussion_scenario + "_" + "_".join(labels)).lower()

        def format_passage_label(p: Passage) -> str:
            return (p.discussion_scenario + "_" + p.label).lower()

        columns = ["label", "text"] + [f"dim_{i}" for i in range(len(self.query_embeddings[0]))]
        data = [(format_query_labels(q), q.text, *emb) for q, emb in zip(self.queries, self.query_embeddings)] + [(format_passage_label(p), p.text, *emb) for p, emb in zip(self.corpus, self.corpus_embeddings)]

        if self.run:
            self.run.log({"embeddings2": wandb.Table(data=data, columns=columns)})

    def _return_topic_rtc(self, discussion_scenario: str) -> ResponseTemplateCollection:
        rtc = self.argument_graphs.get(discussion_scenario)

        if not rtc:
            raise Exception(f"Discussion scenario: {discussion_scenario} not found")

        return rtc

if __name__ == "__main__":
    rtc = load_response_template_collection("s1", "../../")
