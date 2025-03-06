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
        max_depth_first_relevant_text: int = -1
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

        self.max_depth_first_relevant_text = max_depth_first_relevant_text if max_depth_first_relevant_text > 0 else len(self.corpus_ids)

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
        metrics.update(self.log_accuracy_at_k_to_wandb(queries_result_list))
        self.log_error_analysis_table_to_wandb(queries_result_list)
        self.log_confusion_matrices_to_wandb(queries_result_list)

        # Add the standard prefix used by the ST library
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)

        # #Finally, if requested, log the top-k predictions to wandb
        # if self.log_top_k_predictions > 0:
        #     self.log_top_k_predictions_to_wandb(queries_result_list)

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
            self.max_depth_first_relevant_text
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

    def log_accuracy_at_k_to_wandb(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> Dict[str, float] | None:
        """
        Logs accuracy at different k values to wandb.
        Accuracy is tracked based on topic, node type, node level, and node label.
        """
        if not self.accuracy_at_k:
            return

        # Filter and sort list of top-k values
        top_ks = sorted([k for k in self.accuracy_at_k if k > 0])

        def update_accuracy(accuracy_dict: Dict[str, Dict[str, int]], key: str, condition: bool) -> None:
            """
            Updates accuracy dictionary used to track different accuracy metrics by incrementing total and correct count based on a given condition
            """
            if condition:
                accuracy_dict[key]["correct"] += 1
            accuracy_dict[key]["total"] += 1

        def custom_sort_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            """
            Sorts dictionary keys based on the custom priority:
            medai > jurai > autoai > refai
            """
            priority = {
                "medai": 0,
                "jurai": 1,
                "autoai": 2,
                "refai": 3
            }

            def sort_key(key):
                for prefix, prio in priority.items():
                    if key.startswith(prefix):
                        return prio, key
                return 4, key

            return {k: d[k] for k in sorted(d, key=sort_key)}

        def calculate_accuracy(data: Dict[str, int]) -> float:
            """
            Calculates the accuracy for the given dictionary
            """
            return data["correct"] / data["total"]

        # Iterate over all score functions
        for score_func_name, per_query_hits in queries_result_list.items():
            accuracy_results_by_k = {}

            for k in top_ks:
                # Init accuracy tracking dicts
                accuracy_per_topic = defaultdict(lambda: {"total": 0, "correct": 0})
                accuracy_per_topic_and_node_type = defaultdict(lambda: {"total": 0, "correct": 0})
                accuracy_per_topic_and_node_level = defaultdict(lambda: {"total": 0, "correct": 0})
                accuracy_per_topic_and_node_label = defaultdict(lambda: {"total": 0, "correct": 0})

                # Iterate over each query and its corresponding hits sorted by descending similarity
                for q_idx, hits in enumerate(per_query_hits):
                    query = self.queries[q_idx]
                    passages = [
                        self.corpus_map[passage_idx] for _, passage_idx in hits[:k] if passage_idx in self.relevant_docs[query.id]
                    ]
                    discussion_scenario = query.discussion_scenario.lower()
                    rtc = self._return_topic_rtc(discussion_scenario)

                    # Update topic accuracy based on whether any relevant top k passage has the same discussion scenario
                    update_accuracy(accuracy_per_topic, f"{discussion_scenario}", any(query.discussion_scenario.lower() == passage.discussion_scenario.lower() for passage in passages))

                    # Iterate over each label in the query
                    for label in query.labels:
                        query_template = rtc.get_template_for_label(label)

                        # Flags to ensure accuracy gets updated exactly once per metric
                        type_flag = False
                        level_flag = False
                        label_flag = False

                        # Iterate over the top k relevant passages
                        for passage in passages:
                            # Skip passages with wrong discussion scenario
                            if query.discussion_scenario.lower() != passage.discussion_scenario.lower():
                                continue

                            passage_template = rtc.get_template_for_label(passage.label)

                            # Check type matching for query and passage
                            if not type_flag:
                                if query_template is None and passage_template is None:
                                    # Neither are a template -> both are OTHER
                                    update_accuracy(
                                        accuracy_per_topic_and_node_type,
                                        f"{discussion_scenario}_type_OTHER",
                                        True
                                    )
                                    type_flag = True
                                elif query_template is None or passage_template is None:
                                    continue
                                elif query_template.category == passage_template.category:
                                    # Update if categories match
                                    update_accuracy(
                                        accuracy_per_topic_and_node_type,
                                        f"{discussion_scenario}_type_{query_template.category.name}",
                                        True
                                    )
                                    type_flag = True

                            # Check level matching for query and passage
                            if not level_flag:
                                if query_template is None or passage_template is None:
                                    continue

                                if query_template.has_parent_labels == passage_template.has_parent_labels:
                                    # If parent labels bool matches update level accuracy
                                    update_accuracy(
                                        accuracy_per_topic_and_node_level,
                                        f"{discussion_scenario}_level_{'counter' if query_template.has_parent_labels else 'main'}",
                                        True
                                    )
                                    level_flag = True

                            # Check label matching for query and passage
                            if not label_flag:
                                if label == passage.label:
                                    update_accuracy(
                                        accuracy_per_topic_and_node_label,
                                        f"{discussion_scenario}_label_{label}",
                                        True
                                    )
                                    label_flag = True

                            # If all flags are set exit the loop early
                            if type_flag and level_flag and label_flag:
                                break

                        # Update any remaining accuracy which still has a false flag
                        if not type_flag:
                            query_type = "OTHER" if not query_template else query_template.category.name
                            update_accuracy(
                                accuracy_per_topic_and_node_type,
                                f"{discussion_scenario}_type_{query_type}",
                                False
                            )

                        if not level_flag:
                            if query_template is not None:
                                update_accuracy(
                                    accuracy_per_topic_and_node_level,
                                    f"{discussion_scenario}_level_{'counter' if query_template.has_parent_labels else 'main'}",
                                    False
                                )

                        if not label_flag:
                            update_accuracy(
                                accuracy_per_topic_and_node_label,
                                f"{discussion_scenario}_label_{label}",
                                False
                            )

                # Store accuracy results for each k value
                accuracy_results_by_k[k] = {
                    "topic": custom_sort_dict(accuracy_per_topic),
                    "node_type": custom_sort_dict(accuracy_per_topic_and_node_type),
                    "node_level": custom_sort_dict(accuracy_per_topic_and_node_level),
                    "node_label": custom_sort_dict(accuracy_per_topic_and_node_label)
                }

            # Flatten the dictionary and calculate accuracy value for each key
            data = {}
            for k, v in accuracy_results_by_k.items():
                for _k, _ in accuracy_results_by_k[k].items():
                    for key, value in accuracy_results_by_k[k][_k].items():
                        data[f"accuracy_{key}@{k}"] = calculate_accuracy(value)

            return data

    def log_error_analysis_table_to_wandb(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        """
        Logs an error analysis table to wandb containing a row for each query.
        The table includes anchor labels and text, top1-label/text/similarity, top5-rank/label/text/similarity, top1-prediction-match, rank of first correct relevant text.
        """
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

            if self.run:
                self.run.log({f"{self.name}_{score_func_name}_error_analysis": table})

    def log_confusion_matrices_to_wandb(self, queries_result_list: Dict[str, List[List[Tuple[float, str]]]]) -> None:
        """
        Logs confusion matrices for topic, node type, node level and node label predictions.
        """
        # Filter and sort list of top-k values
        top_ks = sorted([k for k in self.accuracy_at_k if k > 0])

        def get_labels_for_scenario(scenario: str) -> List[str]:
            """Retrieve unique sorted labels for a given discussion scenario"""
            return sorted({label for query in self.queries if query.discussion_scenario.lower() == scenario for label in query.labels})

        # Generate label mappings for each discussion scenario
        topic_labels = {scenario: get_labels_for_scenario(scenario) for scenario in ["medai", "jurai", "autoai", "refai"]}

        # Define mappings for topics, node types, nodel levels and labels
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
            for k in top_ks:
                y_true_topic, y_preds_topic = [], []

                for topic in topic_mapping.keys():
                    y_true_type, y_preds_type = [], []
                    y_true_level, y_preds_level = [], []
                    y_true_label, y_preds_label = [], []

                    rtc = self._return_topic_rtc(topic)
                    i = 0
                    for q_idx, hits in enumerate(per_query_hits):
                        query = self.queries[q_idx]

                        # Skip if query doesn't belong to current topic
                        if query.discussion_scenario.lower() != topic:
                            continue

                        # Retrieve top-k passages for the query
                        passages = [self.corpus_map[passage_idx] for _, passage_idx in hits[:k]]

                        # Store true and predicted topics
                        y_true_topic.extend([topic_mapping[topic]] * k)
                        y_preds_topic.extend([topic_mapping[passage.discussion_scenario.lower()] for passage in passages])

                        # Process node type, level and label
                        for label in query.labels:
                            query_template = rtc.get_template_for_label(label)
                            query_type = query_template.category.name if query_template else "OTHER"

                            if query_template:
                                query_level = "counter" if query_template.has_parent_labels else "main"
                                y_true_level.extend([get_node_level_index(query_level)] * k)

                            # Store true labels
                            y_true_type.extend([get_node_type_index(query_type)] * k)
                            y_true_label.extend([get_node_label_index(topic, label)] * k)

                            for passage in passages:
                                if passage.discussion_scenario.lower() != topic:
                                    # Assign wrong scenario for each metric
                                    y_preds_type.append(get_node_type_index("wrong_scenario"))
                                    y_preds_label.append(get_node_label_index(topic, "wrong_scenario"))
                                    if query_template:
                                        y_preds_level.append(get_node_level_index("wrong_scenario"))
                                else:
                                    passage_template = rtc.get_template_for_label(passage.label)
                                    passage_type = passage_template.category.name if passage_template else "OTHER"

                                    if query_template:
                                        if passage_template:
                                            passage_level = "counter" if passage_template.has_parent_labels else "main"
                                            y_preds_level.append(get_node_level_index(passage_level))
                                        else:
                                            y_preds_level.append(get_node_level_index("wrong_type"))

                                    y_preds_type.append(get_node_type_index(passage_type))
                                    y_preds_label.append(get_node_label_index(topic, passage_template.label))

                    if self.run:
                        self.run.log({
                            f"{self.name}_{score_func_name}_confusion_{topic}_node_type_at_{k}": wandb.plot.confusion_matrix(probs=None, y_true=y_true_type, preds=y_preds_type, class_names=list(node_type_mapping.keys()) + ["wrong_topic"]),
                            f"{self.name}_{score_func_name}_confusion_{topic}_node_level_at_{k}": wandb.plot.confusion_matrix(probs=None, y_true=y_true_level, preds=y_preds_level, class_names=list(node_level_mapping.keys()) + ["wrong_topic"] + ["wrong_type"]),
                            f"{self.name}_{score_func_name}_confusion_{topic}_node_labels_at_{k}": wandb.plot.confusion_matrix(probs=None, y_true=y_true_label, preds=y_preds_label, class_names=list(node_label_mapping[topic].keys()) + ["wrong_topic"])
                        })

                if self.run:
                    self.run.log({
                        f"{self.name}_{score_func_name}_confusion_topics_at_{k}": wandb.plot.confusion_matrix(probs=None, y_true=y_true_topic, preds=y_preds_topic, class_names=list(topic_mapping.keys()))
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
