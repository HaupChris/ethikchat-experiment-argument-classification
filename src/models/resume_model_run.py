from datasets import load_from_disk
from ethikchat_evaluators.ClassifierEvaluators.sentence_transformer_intent_classifier_evaluator import SentenceTransformerIntentClassifierEvaluator
from sentence_transformers import SentenceTransformer

from src.data.create_corpus_dataset import Passage, Query
from src.evaluation.deep_dive_information_retrieval_evaluator import DeepDiveInformationRetrievalEvaluator
from src.features.build_features import create_dataset_for_multiple_negatives_ranking_loss
from train_model_sweep import load_argument_graphs
from typing import List, Tuple
from dotenv import load_dotenv
import wandb
import os

def main(runs: List[Tuple[str, str]]) -> None:
    arguments_graphs = load_argument_graphs("../../")

    for run_id, model_path in runs:
        run = wandb.init(project="argument-classification", id=run_id, resume="must")
        config = run.config

        print(f"W&B continuing run: {run.name}")

        project_root = config.get("project_root", "/home/erik/Documents/Uni/ethikchat-experiment-argument-classification")
        env_path = os.path.join(project_root, ".env")
        load_dotenv(env_path, override=True)

        print(f"Project Root: {project_root}")
        print(f"Using model: {config.model_name}")
        print(f"Dataset: {config.dataset_dir}")
        print(f"run_name: {run.name}")

        wandb.login()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Directory not found: {model_path}")

        model = SentenceTransformer(model_path)

        dataset_path = os.path.join(project_root, config.dataset_dir, config.dataset_name)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"No dataset found at {dataset_path}")

        eval_dataset = load_from_disk(dataset_path)
        eval_passages = {
            row["id"]: Passage(row["id"], row["text"], row["label"], row["discussion_scenario"], row["passage_source"]) for
            row in eval_dataset["passages"]}
        eval_queries = {row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"]) for row in
                        eval_dataset["queries"]}
        eval_relevant_passages = {
            row["query_id"]: set(row["passages_ids"]) for row in eval_dataset["queries_relevant_passages_mapping"]
        }
        eval_trivial_passages = {
            row["query_id"]: set(row["passages_ids"]) for row in eval_dataset["queries_trivial_passages_mapping"]
        }

        evaluator = DeepDiveInformationRetrievalEvaluator(
            corpus=eval_passages,
            queries=eval_queries,
            relevant_docs=eval_relevant_passages,
            excluded_docs=eval_trivial_passages,
            show_progress_bar=True,
            write_csv=True,
            run=run,
            argument_graphs=arguments_graphs
        )

        evaluator(model)
        run.finish()


if __name__ == "__main__":
    # TODO: Create sbatch file to compute evaluation on SLURM server

    main([("ucf4sxqd", "deutsche-telekom/gbert-large-paraphrase-euclidean")])