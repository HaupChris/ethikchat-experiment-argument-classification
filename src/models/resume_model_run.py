from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from src.data.create_corpus_dataset import Passage, Query
from src.evaluation.deep_dive_information_retrieval_evaluator import DeepDiveInformationRetrievalEvaluator
from train_model_sweep import load_argument_graphs
from typing import List, Tuple
from dotenv import load_dotenv
import wandb
import os
import torch
import gc
import argparse

def main(project_root: str, models_dir: str, runs: List[Tuple[str, str]], dataset_path: str) -> None:
    """
    Args:
        project_root: Path to root of this project
        models_dir: Path from project root to directory where models are saved
        runs: List of tuples containing (wandb_run_id, wandb_run_name)
        dataset_path: Path to evaluation dataset
    """
    # Load argument graphs and environment variables
    arguments_graphs = load_argument_graphs("../../")
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path)

    for run_id, run_name in runs:
        # Resume previous W&B run
        run = wandb.init(project="argument-classification", id=run_id, resume="must")

        print(f"W&B continuing run: {run.name}")
        print(f"Project Root: {project_root}")
        print(f"Using model: {run_name}")
        print(f"Dataset: {dataset_path}")
        print(f"run_name: {run.name}")

        wandb.login()

        # Construct path to model and ensure it exists
        run_path = os.path.join(models_dir, run_name)

        if not os.path.exists(run_path):
            raise FileNotFoundError(f"Directory not found: {run_path}")

        # get latest checkpoint of that run
        # checkpoint folders are named "checpoint-<save_step>". Get the latest checkpoint
        max_checkpoint_step=max([int(folder.split("-")[1]) for folder in os.listdir(run_path) if "checkpoint" in folder])
        checkpoint_path = os.path.join(run_path, f"checkpoint-{max_checkpoint_step}")


        model = SentenceTransformer(checkpoint_path)

        # Ensure that the passed dataset path exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"No dataset found at {dataset_path}")

        eval_dataset = load_from_disk(dataset_path)

        # Prepare queries and passages from the dataset for the evaluator
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

        # Load the evaluator and run it
        evaluator = DeepDiveInformationRetrievalEvaluator(
            corpus=eval_passages,
            queries=eval_queries,
            relevant_docs=eval_relevant_passages,
            excluded_docs=eval_trivial_passages,
            show_progress_bar=True,
            write_csv=True,
            run=run,
            argument_graphs=arguments_graphs,
            confidence_threshold=0.8,
            confidence_threshold_steps=0.01,
        )

        evaluator(model)
        run.finish()

        # Delete model instance and free up GPU memory
        del model
        clear_unused_gpu_mem()

def clear_unused_gpu_mem():
    """Clears unused GPU memory to free up space"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resume W&B run for further evaluation of a model.')
    parser.add_argument('--project_root', type=str, help="Path to the project root.")
    parser.add_argument('--models_dir', type=str, help="Directory containing all saved models (assuming they are all in one place).")
    parser.add_argument('--run_ids', type=str, nargs="+", help='List of W&B run ids.')
    parser.add_argument('--run_names', type=str, nargs="+", help='List of directory names of models on the slurm server')
    parser.add_argument('--dataset_path', type=str, help='Directory of the dataset used for evaluation.')
    args = parser.parse_args()

    main(
        project_root=args.project_root,
        models_dir=args.models_dir,
        runs=list(zip(args.run_ids, args.run_names)),
        dataset_path=args.dataset_path
    )