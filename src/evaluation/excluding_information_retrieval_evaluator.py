from __future__ import annotations

import heapq
import logging
import os
from contextlib import nullcontext
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Callable, Optional, Any, Dict, List, Tuple, Set

import numpy as np
import torch
from ethikchat_argtoolkit.ArgumentGraph.response_template import TemplateCategory
from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection
from torch import Tensor
from tqdm import trange

import wandb  # Needed for logging tables
from wandb.wandb_run import Run

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction

from src.data.create_corpus_dataset import Query, Passage

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

class ExcludingInformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class extends the standard IR evaluator by:
      1) Allowing per-query exclusions of trivial or undesired documents.
      2) Logging top-k predictions to Weights & Biases (wandb) for error analysis.

    Now it also supports additional metrics / visualizations:
      * (3) A confusion-style breakdown between query labels vs. retrieved doc labels
      * (4) False positives / false negatives table
      * (5) A rank histogram of the first relevant doc
      * (6) Multi-label coverage for each query
      * (7) (Optional) Embedding visualization using t-SNE

    Usage:
        excluded_docs_map = {
            "q1": {"doc_abc", "doc_def"},
            "q5": {"doc_xyz"}
        }
        evaluator = ExcludingInformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=relevant_docs_dict,
            excluded_docs=excluded_docs_map,
            log_top_k_predictions=5,  # e.g., log top 5
            name="my-eval",
            query_labels=some_dict_qid_to_labels,
            doc_labels=some_dict_docid_to_label,
            log_label_confusion=True,
            log_fp_fn=True,
            log_rank_histogram=True,
            log_multilabel_coverage=True,
            log_tsne_embeddings=True,
        )
        results = evaluator(model)
        # This logs IR metrics AND extra tables/plots for deeper analysis.
    """

    def __init__(
        self,
        queries: Dict[str, Query],   # qid => query
        corpus: Dict[str, Passage],   # cid => doc
        relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        map_at_k: List[int] = [100],
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
        log_top_k_predictions: int = 0,  # New param: how many predictions to log per query to wandb
        run: Run = None,
        project_root: str = None,
    ) -> None:
        super().__init__()
        # Filter queries so we only keep the ones that have relevant docs
        self.queries = [query for query_id, query in queries.items() if query_id in relevant_docs and len(relevant_docs[query_id]) > 0]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        # Build a dictionary {doc_id -> doc_text} for logging
        self.corpus_map = dict(zip(self.corpus_ids, self.corpus))

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

        # How many top docs to log to wandb for each query
        self.log_top_k_predictions = log_top_k_predictions
        self.run = run

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]
        self._append_csv_headers(self.score_function_names)

        self.project_root = project_root if project_root is not None else "../../"
        self.medai_rtc = ResponseTemplateCollection.from_csv_files(os.path.join(self.project_root, "data", "external", "argument_graphs", "szenario_s1"))
        self.jurai_rtc = ResponseTemplateCollection.from_csv_files(os.path.join(self.project_root, "data", "external", "argument_graphs", "szenario_s2"))
        self.autoai_rtc = ResponseTemplateCollection.from_csv_files(os.path.join(self.project_root, "data", "external", "argument_graphs", "szenario_s3"))
        self.refai_rtc = ResponseTemplateCollection.from_csv_files(os.path.join(self.project_root, "data", "external", "argument_graphs", "szenario_s4"))

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
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs
    ) -> dict[str, float]:
        # Build a string describing where we are in training
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

        # If no custom scoring functions were provided, use the model's default
        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)

        # Main IR procedure
        scores, queries_result_list = self.compute_metrices(model, *args, **kwargs)

        # Write results to CSV if needed
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            write_header = not os.path.isfile(csv_path)
            with open(csv_path, mode="a", encoding="utf-8") as fOut:
                if write_header:
                    fOut.write(",".join(self.csv_headers) + "\n")

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

                fOut.write(",".join(map(str, output_data)) + "\n")

        # Determine the primary metric if not set
        if not self.primary_metric:
            if self.main_score_function is None:
                # By default, pick the largest NDCG@max
                score_function = max(
                    [(s_name, scores[s_name]["ndcg@k"][max(self.ndcg_at_k)]) for s_name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"

        # Convert to a single metric dict
        metrics = {
            f"{score_function}_{metric_name.replace('@k', '@' + str(k))}": value
            for score_function, values_dict in scores.items()
            for metric_name, values in values_dict.items()
            for k, value in values.items()
        }
        # Add the standard prefix used by the ST library
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)

        # #Finally, if requested, log the top-k predictions to wandb
        # if self.log_top_k_predictions > 0:
        #     self.log_top_k_predictions_to_wandb(queries_result_list)
        self.log_accuracy_at_k_to_wandb(queries_result_list)
        return metrics

    def compute_metrices(
        self,
        model: SentenceTransformer,
        corpus_model=None,
        corpus_embeddings: Tensor | None = None
    ) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, List[List[Tuple[float, str]]]]]:
        """
        Runs the retrieval: encodes queries vs. corpus, does top-k retrieval,
        and computes standard IR metrics. Returns:
          (scores, queries_result_list)

        where:
          scores:  {score_function_name -> { "accuracy@k":..., "mrr@k":..., ...}}
          queries_result_list:  {score_function_name -> [list_of_top_docs_per_query]}
        """
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
                [query.text for query in self.queries],
                prompt_name=self.query_prompt_name,
                prompt=self.query_prompt,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
            )

        # We'll store results as: queries_result_list[score_func][query_idx] = list of (score, doc_id)
        queries_result_list: Dict[str, List[List[Tuple[float, str]]]] = {}
        for score_name in self.score_functions:
            queries_result_list[score_name] = [[] for _ in range(len(query_embeddings))]

        # Process corpus in chunks
        for corpus_start_idx in trange(
            0, len(self.corpus), self.corpus_chunk_size, desc="Corpus Chunks", disable=not self.show_progress_bar
        ):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            # Encode chunk of corpus if needed
            if corpus_embeddings is None:
                with (
                    nullcontext()
                    if self.truncate_dim is None
                    else corpus_model.truncate_sentence_embeddings(self.truncate_dim)
                ):
                    sub_corpus_emb = corpus_model.encode(
                        [passage.text for passage in self.corpus[corpus_start_idx:corpus_end_idx]],
                        prompt_name=self.corpus_prompt_name,
                        prompt=self.corpus_prompt,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                    )
            else:
                sub_corpus_emb = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # For each scoring function
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_emb)
                # top-k in each row (i.e., for each query)
                top_k_vals, top_k_idx = torch.topk(
                    pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
                )
                top_k_vals = top_k_vals.cpu().tolist()
                top_k_idx = top_k_idx.cpu().tolist()

                # For each query, integrate chunk results
                for q_idx in range(len(query_embeddings)):
                    query = self.queries[q_idx]
                    exclude_set = self.excluded_docs.get(query.id, set())

                    # We'll gather valid hits in a min-heap
                    valid_hits = queries_result_list[name][q_idx]
                    for cid_idx, score_val in zip(top_k_idx[q_idx], top_k_vals[q_idx]):
                        doc_id = self.corpus_ids[corpus_start_idx + cid_idx]
                        if doc_id in exclude_set:
                            # Skip excluded documents
                            continue
                        if len(valid_hits) < max_k:
                            heapq.heappush(valid_hits, (score_val, doc_id))
                        else:
                            heapq.heappushpop(valid_hits, (score_val, doc_id))

        # Now sort those heaps by descending score
        for name in queries_result_list:
            for q_idx in range(len(queries_result_list[name])):
                sorted_hits = sorted(queries_result_list[name][q_idx], key=lambda x: x[0], reverse=True)
                queries_result_list[name][q_idx] = sorted_hits
        logger.info(f"Queries: {len(self.queries)}")
        logger.info(f"Corpus: {len(self.corpus)}\n")

        # Compute final metrics (scores)
        scores: Dict[str, Dict[str, Dict[int, float]]] = {}
        for score_name in self.score_functions:
            # Convert structure for metric calc
            metrics = self.compute_metrics(queries_result_list[score_name], score_name)
            scores[score_name] = metrics


        # Print them
        for score_name in self.score_function_names:
            logger.info(f"Score-Function: {score_name}")
            self.output_scores(scores[score_name])

        return scores, queries_result_list

    def compute_metrics(
        self,
        queries_result_list_for_score: List[List[Tuple[float, str]]],
        score_name: str
    ) -> Dict[str, Dict[int, float]]:
        """
        Takes the final sorted list of results for each query (already top-k).
        Each item is a list of (score, doc_id).
        """
        # Initialize
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # For each query
        for q_idx, top_hits in enumerate(queries_result_list_for_score):
            query_id = self.queries[q_idx].id
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k
            for k_val in self.accuracy_at_k:
                for (scr, doc_id) in top_hits[:k_val]:
                    if doc_id in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision & Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for (scr, doc_id) in top_hits[:k_val]:
                    if doc_id in query_relevant_docs:
                        num_correct += 1
                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, (scr, doc_id) in enumerate(top_hits[:k_val]):
                    if doc_id in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if doc_id in query_relevant_docs else 0
                    for (scr, doc_id) in top_hits[:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)
                ndcg_val = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_val)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0
                for rank, (scr, doc_id) in enumerate(top_hits[:k_val]):
                    if doc_id in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)
                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(queries_result_list_for_score)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k]) if precisions_at_k[k] else 0.0

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k]) if recall_at_k[k] else 0.0

        for k in MRR:
            MRR[k] /= len(queries_result_list_for_score)

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k]) if ndcg[k] else 0.0

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k]) if AveP_at_k[k] else 0.0

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    def output_scores(self, scores):
        """
        Prints out the standard IR metrics to the logger.
        """
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
        dcg = 0.0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start at idx=0
        return dcg

    def log_top_k_predictions_to_wandb(
        self,
        queries_result_list: Dict[str, List[List[Tuple[float, str]]]]
    ):
        """
        Logs a wandb.Table for each similarity function in queries_result_list,
        capturing the top N predictions (self.log_top_k_predictions) for each query.
        """

        top_k = self.log_top_k_predictions
        if top_k <= 0:
            return

        # For each similarity function
        for score_func_name, per_query_hits in queries_result_list.items():
            # Create a fresh table
            table = wandb.Table(columns=[
                "query_id", "query_text", "rank",
                "doc_id", "doc_text", "score", "is_relevant"
            ])

            # Populate the table
            for q_idx, hits in enumerate(per_query_hits):
                query_id = self.queries_ids[q_idx]
                query_text = self.queries[q_idx]
                # Sort descending by score
                sorted_hits = sorted(hits, key=lambda x: x[0], reverse=True)
                # Take the top_k
                for rank_idx, (score_val, doc_id) in enumerate(sorted_hits[:top_k], start=1):
                    is_rel = (doc_id in self.relevant_docs[query_id])
                    doc_text = self.corpus_map[doc_id]
                    table.add_data(
                        query_id,
                        query_text,
                        rank_idx,
                        doc_id,
                        doc_text,
                        score_val,
                        is_rel
                    )

            # Log to wandb
            if self.run:
                self.run.log({f"{self.name}_{score_func_name}_top{top_k}_predictions": table})

    def log_accuracy_at_k_to_wandb(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        if not self.accuracy_at_k:
            return

        top_ks = sorted([k for k in self.accuracy_at_k if k > 0])

        def update_accuracy(accuracy_dict: Dict[str, Dict[str, int]], key: str, condition: bool) -> None:
            if condition:
                accuracy_dict[key]["correct"] += 1
            accuracy_dict[key]["total"] += 1

        for score_func_name, per_query_hits in queries_result_list.items():
            accuracy_results_by_k = {}
            for k in top_ks:
                accuracy_per_topic = defaultdict(lambda: {"total": 0, "correct": 0})
                accuracy_per_topic_and_node_type = defaultdict(lambda: {"total": 0, "correct": 0})
                accuracy_per_topic_and_node_level = defaultdict(lambda: {"total": 0, "correct": 0})
                accuracy_per_topic_and_node_label = defaultdict(lambda: {"total": 0, "correct": 0})

                for q_idx, hits in enumerate(per_query_hits):
                    query = self.queries[q_idx]
                    any_relevant_passages = any(passage_idx in self.relevant_docs[query.id] for _, passage_idx in hits[:k])
                    discussion_scenario = query.discussion_scenario.lower()
                    rtc = self._return_topic_rtc(discussion_scenario)

                    update_accuracy(accuracy_per_topic, discussion_scenario, any_relevant_passages)

                    for label in query.labels:
                        update_accuracy(accuracy_per_topic_and_node_label, f"{discussion_scenario}_{label}", any_relevant_passages)

                        label_template = rtc.get_template_for_label(label)

                        if label_template is None:
                            update_accuracy(accuracy_per_topic_and_node_type, f"{discussion_scenario}_other", any_relevant_passages)
                        else:
                            update_accuracy(accuracy_per_topic_and_node_type, f"{discussion_scenario}_{label_template.category}", any_relevant_passages)

                            if label_template.category == TemplateCategory.Z:
                                if label_template.has_parent_labels:
                                    update_accuracy(accuracy_per_topic_and_node_level, f"{discussion_scenario}_counter", any_relevant_passages)
                                else:
                                    update_accuracy(accuracy_per_topic_and_node_level, f"{discussion_scenario}_main", any_relevant_passages)

                accuracy_results_by_k[k] = {
                    "topic": accuracy_per_topic,
                    "node_type": accuracy_per_topic_and_node_type,
                    "node_level": accuracy_per_topic_and_node_level,
                    "node_label": accuracy_per_topic_and_node_label
                }
    def _log_top1_classification(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        """
            Logs evaluation results for top-1 classification to Weights & Biases (wandb).

            The following tables and visualizations are logged:
            - A main table (`top1_classification_table`) containing details for each query and its top-1 result.
            - Histograms of confidence scores for correct and incorrect predictions.
            - Bar charts showing the distribution of label type for correct and incorrect predictions.
            - A bar chart showing the accuracy (correct prediction ratio) for each label type.

            Table Columns:
                - `query_id`: Unique identifier for the query.
                - `query_text`: The text of the query.
                - `query_labels`: Ground truth labels for the query, joined as a comma-separated string.
                - `top1_id`: Identifier of the highest-rated passage
                - `top1_text`: Text of the highest-rated passage
                - `top1_label`: Label of the highest-rated passage
                - `confidence_level`: Confidence score (similarity score) between the query and the top-1 passage.
                - `match`: Boolean indicating whether the top-1 passage is among the relevant passages for the query.

            Visualizations:
                - Confidence level distributions for correct and incorrect predictions.
                - Label type distributions for correct and incorrect predictions.
                - Accuracy (correct prediction ratio) for each label type.
            """
        if not self.run:
            return

        top1_classification_table = wandb.Table(
            columns=["query_id", "query_labels", "query_text", "num_query_labels", "top1_id", "top1_label", "top1_text","confidence_level", "match"]
        )
        top1_classification_correct_histogram = wandb.Table(columns=["confidence_level"])
        top1_classification_incorrect_histogram = wandb.Table(columns=["confidence_level"])

        def _determine_label_group(label: str) -> str:
            """
            Helper function to categorize labels into groups based on their prefixes or concrete values.
            """
            if label.startswith("Z."):
                return "Z"
            elif label.startswith("NZ."):
                return "NZ"
            elif label == "CONSENT":
                return "CONSENT"
            elif label == "DISSENT":
                return "DISSENT"
            elif label.startswith("FAQ."):
                return "FAQ"
            else:
                return label

        score_func_name = next(iter(queries_result_list.keys()))

        # Lists to store labels for correct and incorrect predictions
        correct_labels = []
        incorrect_labels = []

        # Generate a unique identifier for each label in the corpus
        labels = set()
        for query_labels in self.query_labels.values():
            labels = labels.union(query_labels)

        for doc_label in self.doc_labels.values():
            labels.add(doc_label)

        labels_to_idx = {label: idx for idx, label in enumerate(labels)}

        eval_data = []
        # Iterate over each query in the evaluation corpus and its top-1 classification
        for q_idx, top_hits in enumerate(queries_result_list[score_func_name]):
            query_id = self.queries_ids[q_idx]
            query_text = self.queries[q_idx]
            query_labels = self.query_labels[query_id]
            num_query_labels = len(query_labels)
            query_relevant_docs = self.relevant_docs[query_id]

            # Extract top-1 classification details
            score, doc_id = top_hits[0]
            doc_text = self.corpus_map[doc_id]
            doc_label = self.doc_labels[doc_id]

            # Check if the top-1 classification is among the relevant documents of the initial query
            correct_prediction = doc_id in query_relevant_docs

            # Log data depending on whether the prediction is correct or incorrect
            if correct_prediction:
                top1_classification_correct_histogram.add_data(score)
                correct_labels.append(_determine_label_group(doc_label))
                eval_data.append([labels_to_idx[doc_label], labels_to_idx[doc_label], query_id, query_text, doc_label, doc_id, doc_text, doc_label, score, correct_prediction, num_query_labels])
            else:
                top1_classification_incorrect_histogram.add_data(score)
                for label in query_labels:
                    incorrect_labels.append(_determine_label_group(label))
                    eval_data.append([labels_to_idx[doc_label], labels_to_idx[label], query_id, query_text, label, doc_id, doc_text, doc_label, score, correct_prediction, num_query_labels])

            # Add relevant data to the main evaluation table
            top1_classification_table.add_data(query_id, ", ".join(query_labels), query_text, num_query_labels, doc_id, doc_label, doc_text, score, correct_prediction)

        # Count occurrences of each label for correct and incorrect predictions
        correct_label_counts = Counter(correct_labels)
        incorrect_label_counts = Counter(incorrect_labels)

        # Get the set of all unique labels in the corpus from the correct and incorrect labels
        all_labels = set(correct_label_counts.keys()).union(set(incorrect_label_counts.keys()))

        # Calculate the accuracy for each label
        data = []
        for label in all_labels:
            correct = correct_label_counts.get(label, 0)
            incorrect = incorrect_label_counts.get(label, 0)
            total = correct + incorrect
            ratio = correct / total if total > 0 else 0
            data.append([label, ratio])

        correct_label_table = wandb.Table(data=[[k, v] for k, v in correct_label_counts.items()], columns=["label", "count"])
        incorrect_label_table = wandb.Table(data=[[k, v] for k, v in incorrect_label_counts.items()], columns=["label", "count"])
        ratio_table = wandb.Table(data=data, columns=["label", "correct_ratio"])

        scatter_plot = wandb.Table(
            data=eval_data,
            columns=[
                "predicted_label_idx", "reference_label_idx",
                "query_id", "query_text", "query_label",
                "top1_id", "top1_text", "top1_label",
                "confidence", "match", "num_labels_query"
            ]
        )
        # Log all tables and visualization to wandb
        self.run.log({
            f"{self.name}_top1_classification_table": top1_classification_table,
            f"{self.name}_top1_classification_confidence_level_distribution_correct": wandb.plot.histogram(
                top1_classification_correct_histogram,
                value="confidence_level",
                title="Top-1 Classification Confidence Level Distribution For Correct Predictions"
            ),
            f"{self.name}_top1_classification_confidence_level_distribution_incorrect": wandb.plot.histogram(
                top1_classification_incorrect_histogram,
                value="confidence_level",
                title="Top-1 Classification Confidence Level Distribution For Incorrect Predictions"
            ),
            f"{self.name}_top1_classification_label_distribution_correct": wandb.plot.bar(
                correct_label_table, "label", "count", title="Top-1 Classification Label Distribution For Correct Predictions"
            ),
            f"{self.name}_top1_classification_label_distribution_incorrect": wandb.plot.bar(
                incorrect_label_table, "label", "count", title="Top-1 Classification Label Distribution For Incorrect Predictions"
            ),
            f"{self.name}_top1_classification_label_correct_ratio": wandb.plot.bar(
                ratio_table, "label", "correct_ratio", title="Top-1 Classification Accuracy per Label"
            ),
            f"{self.name}_top1_classification_scatter_plot": wandb.plot.scatter(scatter_plot, x="predicted_label_idx", y="reference_label_idx", title="Top-1 Classification Scatter Plot")
        })

    def _return_topic_rtc(self, discussion_scenario: str) -> ResponseTemplateCollection:
        discussion_scenario = discussion_scenario.lower()
        if discussion_scenario == "medai":
            return self.medai_rtc
        elif discussion_scenario == "jurai":
            return self.jurai_rtc
        elif discussion_scenario == "autoai":
            return self.autoai_rtc
        elif discussion_scenario == "refai":
            return self.refai_rtc
        else:
            raise Exception(f"Discussion Scenario '{discussion_scenario}' doesnt have a valid ResponseTemplateCollection")