from __future__ import annotations

import heapq
import logging
import os
from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Optional, Dict, List, Tuple, Set

import numpy as np
import torch
from ethikchat_argtoolkit.ArgumentGraph.response_template import ResponseTemplate, TemplateCategory
from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection
from ethikchat_argtoolkit.ArgumentGraph.stance import Stance
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


class Evaluator(SentenceEvaluator):
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
        run: Run = None,
        argument_graphs: Dict[str, ResponseTemplateCollection] = None
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
        self.argument_graphs = argument_graphs

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

        # 6) Stance
        print(self._compute_stance_accuracy(queries_result_list))
        ##metrics.update(stance_accuracy)

        # 7) Optionally log loss
        if loss is not None:
            metrics["loss"] = loss

        # 8) Log topic confusion matrix
        self._log_topic_confusion_matrix(queries_result_list)

        # 9) Give these metrics a prefix and store in model card
        final_metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, final_metrics, epoch, steps)

        # 10) Log to W&B (if self.run is set)
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

                    # Store true and predicted topics
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

                        if passage.discussion_scenario != topic:
                            # Assign wrong scenario for each metric
                            y_preds_type.append(get_node_type_index("wrong_scenario"))
                            y_preds_label.append(get_node_label_index(topic, "wrong_scenario"))
                            if query_template and query_template.label in rtc.arguments_labels:
                                y_preds_level.append(get_node_level_index("wrong_scenario"))
                        else:
                            passage_template = rtc.get_template_for_label(passage.label)
                            passage_type = passage_template.category.name if passage_template else "OTHER"
                            y_preds_type.append(get_node_type_index(passage_type))
                            y_preds_label.append(get_node_label_index(topic, passage_template.label))

                            if query_template and query_template.label in rtc.arguments_labels:
                                if passage_template and passage_template.label in rtc.arguments_labels:
                                    passage_level = "counter" if passage_template.has_parent_labels else "main"
                                    y_preds_level.append(get_node_level_index(passage_level))
                                else:
                                    y_preds_level.append(get_node_level_index("wrong_type"))

                if self.run:
                    self.run.log({
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_type": wandb.plot.confusion_matrix(probs=None, y_true=y_true_type, preds=y_preds_type, class_names=list(node_type_mapping.keys()) + ["wrong_topic"], title=f"{score_func_name}: {topic} Node Type Confusion Matrix (Absolute)"),
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_level": wandb.plot.confusion_matrix(probs=None, y_true=y_true_level, preds=y_preds_level, class_names=list(node_level_mapping.keys()) + ["wrong_topic"] + ["wrong_type"], title=f"{score_func_name}: {topic} Node Level Confusion Matrix (Absolute)"),
                        f"{self.name}_{score_func_name}_confusion_{topic}_node_labels": wandb.plot.confusion_matrix(probs=None, y_true=y_true_label, preds=y_preds_label, class_names=list(node_label_mapping[topic].keys()) + ["wrong_topic"], title=f"{score_func_name}: {topic} Label Confusion Matrix (Absolute)")
                    })

            if self.run:
                self.run.log({
                    f"{self.name}_{score_func_name}_confusion_topics": wandb.plot.confusion_matrix(probs=None, y_true=y_true_topic, preds=y_preds_topic, class_names=list(topic_mapping.keys()), title=f"{score_func_name}: Topic Confusion Matrix (Absolute)")
                })

    def _compute_stance_accuracy(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> Dict[str, float]:
        """
        Computes stance accuracy for different groupings (topic, type, level, label) based on query results. The stance for the topic grouping is considered correct
        if the stance of the top1-passage is the same as the stance of any label of the query passage. For type, level, and label it is calculated for each label in the query.
        """
        if not self.run:
            return {}

        def update_accuracy(accuracy_dict: Dict[str, Dict[str, int]], key: str, condition: bool) -> None:
            """Updates accuracy dictionary used to track different accuracy metrics by incrementing total and correct count based on a given condition"""
            accuracy_dict[key]["total"] += 1
            if condition:
                accuracy_dict[key]["correct"] += 1

        def calculate_accuracy(data: Dict[str, int]) -> float:
            """Calculates the accuracy for the given dictionary"""
            return data["correct"] / data["total"]

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
                "label": defaultdict(lambda: {"total": 0, "correct": 0}),
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

                    # Update stance accuracy per node label
                    update_accuracy(metrics["label"], f"{discussion_scenario}_label_{label}", query_stance == passage_stance)

                    # Skip node level if query is not an argument
                    if not query_template or query_template.label not in rtc_q.arguments_labels:
                        continue

                    # Determine argument level and update stance accuracy per node level
                    arg_level = "counter" if query_template.has_parent_labels else "main"
                    update_accuracy(metrics["label"], f"{discussion_scenario}_level_{arg_level}", query_stance == passage_stance)

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

        top_ks = sorted([k for k in self.accuracy_at_k if k > 0])

        def update_accuracy(accuracy_dict: Dict[str, Dict[str, int]], key: str, condition: bool) -> None:
            """Updates accuracy dictionary used to track different accuracy metrics by incrementing total and correct count based on a given condition"""
            accuracy_dict[key]["total"] += 1
            if condition:
                accuracy_dict[key]["correct"] += 1

        def calculate_accuracy(data: Dict[str, int]) -> float:
            """
            Calculates the accuracy for the given dictionary
            """
            return data["correct"] / data["total"]

        for score_func_name, per_query_hits in queries_result_list.items():
            accuracy_results_by_k = {}

            for k in top_ks:
                accuracy_per_topic = defaultdict(lambda: {"total": 0, "correct": 0})
                accuracy_per_topic_and_node_type = defaultdict(lambda: {"total": 0, "correct": 0})
                accuracy_per_topic_and_node_level = defaultdict(lambda: {"total": 0, "correct": 0})
                accuracy_per_topic_and_node_label = defaultdict(lambda: {"total": 0, "correct": 0})

                for q_idx, hits in enumerate(per_query_hits):
                    query = self.queries[q_idx]
                    hits = any(doc_id in self.relevant_docs[query.id] for _, doc_id in hits[:k])
                    discussion_scenario = query.discussion_scenario
                    rtc = self._return_topic_rtc(discussion_scenario)

                    update_accuracy(accuracy_per_topic, f"{discussion_scenario}", hits)

                    for label in query.labels:
                        query_template = rtc.get_template_for_label(label)

                        update_accuracy(
                            accuracy_per_topic_and_node_type,
                            f"{discussion_scenario}_type_{getattr(query_template, 'category', TemplateCategory.OTHER).name}",
                            hits
                        )

                        if query_template and query_template in rtc.arguments_labels:
                            update_accuracy(
                                accuracy_per_topic_and_node_level,
                                f"{discussion_scenario}_level_{'counter' if query_template.has_parent_labels else 'main'}",
                                hits
                            )

                        update_accuracy(accuracy_per_topic_and_node_label, f"{discussion_scenario}_label_{label}", hits)

                accuracy_results_by_k[k] = {
                    "topic": accuracy_per_topic,
                    "node_type": accuracy_per_topic_and_node_type,
                    "node_level": accuracy_per_topic_and_node_level,
                    "node_label": accuracy_per_topic_and_node_label
                }

            accuracies_at_k = [f"accuracy_at_{k}" for k in top_ks]

            topic_table = wandb.Table(columns=[
                "topic", *accuracies_at_k, "num_queries", "correct_predictions"
            ])

            topic_type_table = wandb.Table(columns=[
                "topic", "node_type", *accuracies_at_k, "num_queries", "correct_predictions"
            ])

            topic_level_table = wandb.Table(columns=[
                "topic", "node_level", *accuracies_at_k, "num_queries", "correct_predictions"
            ])

            topic_label_table = wandb.Table(columns=[
                "topic", "nodel_label", *accuracies_at_k, "num_queries", "correct_predictions"
            ])

            first_k = next(iter(top_ks))

            for topic in accuracy_results_by_k[first_k]["topic"]:
                data = [topic]
                for k in top_ks:
                    data.append(calculate_accuracy(accuracy_results_by_k[k]["topic"][topic]))
                data.append(accuracy_results_by_k[first_k]["topic"][topic]["total"])
                data.append(accuracy_results_by_k[first_k]["topic"][topic]["correct"])
                topic_table.add_data(data)

            for topic_type in accuracy_results_by_k[first_k]["node_type"]:
                split = topic_type.split("_type_")
                topic = split[0]
                arg_type = split[1]

                data = [topic, arg_type]
                for k in top_ks:
                    data.append(calculate_accuracy(accuracy_results_by_k[k]["node_type"][topic_type]))
                data.append(accuracy_results_by_k[first_k]["node_type"][topic_type]["total"])
                data.append(accuracy_results_by_k[first_k]["node_type"][topic_type]["correct"])
                topic_type_table.add_data(data)

            for topic_level in accuracy_results_by_k[first_k]["node_level"]:
                split = topic_level.split("_level_")
                topic = split[0]
                arg_level = split[1]

                data = [topic, arg_level]
                for k in top_ks:
                    data.append(calculate_accuracy(accuracy_results_by_k[k]["node_level"][topic_level]))
                data.append(accuracy_results_by_k[first_k]["node_level"][topic_level]["total"])
                data.append(accuracy_results_by_k[first_k]["node_level"][topic_level]["correct"])
                topic_level_table.add_data(data)

            for topic_label in accuracy_results_by_k[first_k]["node_label"]:
                split = topic_label.split("_label_")
                topic = split[0]
                arg_label = split[1]

                data = [topic, arg_label]
                for k in top_ks:
                    data.append(calculate_accuracy(accuracy_results_by_k[k]["node_label"][topic_label]))
                data.append(accuracy_results_by_k[first_k]["node_label"][topic_label]["total"])
                data.append(accuracy_results_by_k[first_k]["node_label"][topic_label]["correct"])
                topic_label_table.add_data(data)

            if self.run:
                self.run.log({
                    f"{self.name}_{score_func_name}_topic_accuracy": topic_table,
                    f"{self.name}_{score_func_name}_topic_type_accuracy": topic_type_table,
                    f"{self.name}_{score_func_name}_topic_level_accuracy": topic_level_table,
                    f"{self.name}_{score_func_name}_topic_label_accuracy": topic_label_table,
                })

    def _return_topic_rtc(self, discussion_scenario: str) -> ResponseTemplateCollection:
        rtc = self.argument_graphs.get(discussion_scenario)

        if not rtc:
            raise Exception(f"Discussion scenario: {discussion_scenario} not found")

        return rtc

if __name__ == "__main__":
    rtc = load_response_template_collection("s1", "../../")
