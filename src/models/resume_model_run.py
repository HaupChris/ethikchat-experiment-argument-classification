from datasets import load_from_disk, DatasetDict
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer

from src.data.classes import Passage, Query
from src.evaluation.deep_dive_information_retrieval_evaluator import DeepDiveInformationRetrievalEvaluator
from src.features.build_features import add_context_to_texts, add_scenario_tokens_to_texts
from train_model_sweep import load_argument_graphs, check_dataset_texts_for_truncation
from typing import List, Tuple
from dotenv import load_dotenv
import json
import wandb
import os
import torch
import gc
import argparse


def prepare_datasets(
        corpus_dataset: DatasetDict,
        split_dataset: DatasetDict,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        add_discussion_scenario_info: bool,
        maximum_sequence_length: int,
):
    """
    Loads the corpus dataset from disk, creates splits,
    and optionally uses smaller data for local testing if `is_test_run=True`.
    Returns all training data, evaluation data, and the evaluators.
    """

    tokenizer_sep_token = getattr(tokenizer, "sep_token", None)

    if tokenizer_sep_token is None:
        sep_token = "[SEP]"
    else:
        sep_token = tokenizer_sep_token

    test_split = split_dataset
    test_split = add_context_to_texts(test_split, context_length, sep_token)

    corpus_dataset = add_context_to_texts(corpus_dataset, context_length, sep_token, "noisy_queries")

    if add_discussion_scenario_info:
        test_split = add_scenario_tokens_to_texts(test_split)
        corpus_dataset = add_scenario_tokens_to_texts(corpus_dataset, ["noisy_queries"])

    check_dataset_texts_for_truncation(tokenizer, test_split, "test", maximum_sequence_length)
    check_dataset_texts_for_truncation(tokenizer, corpus_dataset, "corpus_dataset", maximum_sequence_length,
                                       ["noisy_queries"])

    # Build references for TEST
    test_passages = {
        row["id"]: Passage(
            row["id"], row["text"], row["label"],
            row["discussion_scenario"], row["passage_source"]
        )
        for row in test_split["passages"]
    }
    test_queries = {
        row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"])
        for row in test_split["queries"]
    }
    test_relevant_passages = {
        row["query_id"]: set(row["passages_ids"])
        for row in test_split["queries_relevant_passages_mapping"]
    }
    test_trivial_passages = {
        row["query_id"]: set(row["passages_ids"])
        for row in test_split["queries_trivial_passages_mapping"]
    }
    noisy_queries = {
        row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"])
        for row in corpus_dataset["noisy_queries"]
    }

    return test_queries, test_passages, test_relevant_passages, test_trivial_passages, noisy_queries


def main(project_root: str, models_dir: str, runs: List[Tuple[str, str]], test_dataset_path: str,
         corpus_dataset_path: str) -> None:
    """
    Args:
        project_root: Path to root of this project
        models_dir: Path from project root to directory where models are saved
        runs: List of tuples containing (wandb_run_id, wandb_run_name)
        test_dataset_path: Path to test dataset
        corpus_dataset_path
    """
    # Load argument graphs and environment variables

    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path)

    for run_id in runs:
        # Resume previous W&B run
        run = wandb.init(project="argument-classification", id=run_id, resume="must")

        # get dataset parameters from run config
        context_length = run.config["context_length"]
        add_discussion_scenario_info = run.config["add_discussion_scenario_info"]

        print(f"W&B continuing run: {run.name}")
        print(f"Project Root: {project_root}")
        print(f"Using model: {run.name}")
        print(f"Test Dataset: {test_dataset_path}")
        print(f"Corpus Dataset: {corpus_dataset_path}")
        print(f"run_name: {run.name}")
        print("+++++++++++++++++++++++++++  Run config: ++++++++++++++++++++++++++++++++++++++++++")
        print(run.config)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        wandb.login()

        # Construct path to model and ensure it exists
        run_path = os.path.join(models_dir, run.name)

        if not os.path.exists(run_path):
            raise FileNotFoundError(f"Directory not found: {run_path}")

        # get latest checkpoint of that run
        # checkpoint folders are named "checpoint-<save_step>". Get the latest checkpoint
        latest_checkpoint_step = max(
            [int(folder.split("-")[1]) for folder in os.listdir(run_path) if "checkpoint" in folder])
        latest_checkpoint_path = os.path.join(run_path, f"checkpoint-{latest_checkpoint_step}")

        # get the best checkpoint of the model. this is logged in trainer_state.json
        with open(os.path.join(latest_checkpoint_path, "trainer_state.json")) as json_file:
            trainer_state = json.loads(json_file.read())
            best_model_checkpoint_path = trainer_state["best_model_checkpoint"]
            if not os.path.exists(best_model_checkpoint_path):
                # running script locally, not on cluster
                best_model_checkpoint_path = os.path.join(run_path, trainer_state["best_model_checkpoint"].split("/")[-1])

        print(f"Loading checkpoint: {best_model_checkpoint_path}")
        model = SentenceTransformer(best_model_checkpoint_path)

        # Ensure that the passed dataset path exists
        if not os.path.exists(test_dataset_path):
            raise FileNotFoundError(f"No dataset found at {test_dataset_path}")

        if not os.path.exists(corpus_dataset_path):
            raise FileNotFoundError(f"No dataset found at {corpus_dataset_path}")

        test_dataset = load_from_disk(test_dataset_path)
        corpus_dataset = load_from_disk(corpus_dataset_path)

        (test_queries,
         test_passages,
         test_relevant_passages,
         test_trivial_passages,
         noisy_queries) = prepare_datasets(corpus_dataset,
                                           test_dataset,
                                           model.tokenizer,
                                           context_length,
                                           add_discussion_scenario_info,
                                           model.max_seq_length)

        arguments_graphs = load_argument_graphs("../../")

        deep_dive_evaluator_test = DeepDiveInformationRetrievalEvaluator(
            corpus=test_passages,
            queries=test_queries,
            noisy_queries=noisy_queries,
            relevant_docs=test_relevant_passages,
            excluded_docs=test_trivial_passages,
            show_progress_bar=True,
            accuracy_at_k=[1, 3, 5, 7],
            precision_at_k=[1, 3, 5, 7],
            write_csv=True,
            run=wandb.run,
            argument_graphs=arguments_graphs,
            confidence_threshold=0.7,
            confidence_threshold_steps=0.01,
            name="Test_After_run_deepdive",
            save_tables_as_csv=True,
            csv_output_dir=run_path
        )

        deep_dive_evaluator_test(model)
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
    parser = argparse.ArgumentParser(description='Resume W&B run for further testing of a model.')
    parser.add_argument('--project_root', type=str, help="Path to the project root.")
    parser.add_argument('--models_dir', type=str,
                        help="Directory containing all saved models (assuming they are all in one place).")
    parser.add_argument('--run_ids', type=str, nargs="+", help='List of W&B run ids.')
    parser.add_argument('--test_dataset_path', type=str, help='Directory of the dataset used for testing.')
    parser.add_argument('--corpus_dataset_path', type=str, help='Directory of the corpus dataset used for testing.')
    args = parser.parse_args()

    main(
        project_root=args.project_root,
        models_dir=args.models_dir,
        runs=list(args.run_ids),
        test_dataset_path=args.test_dataset_path,
        corpus_dataset_path=args.corpus_dataset_path
    )

        # test
    # project_root="/home/christian/PycharmProjects/ethikchat-experiment-argument-classification"
    # sweep_id="6te7vzul"
    #
    # runs = [("nq1xf4g7", "lilac-sweep-65/")]
    # main(
    #     project_root=project_root,
    #     models_dir=f"{project_root}/experiments_outputs/{sweep_id}",
    #     runs=runs,
    #     test_dataset_path=f"{project_root}/data/processed/with_context/dataset_split_in_distribution_from_v3/test",
    #     corpus_dataset_path=f"{project_root}/data/processed/with_context/corpus_dataset_v3"
    # )
