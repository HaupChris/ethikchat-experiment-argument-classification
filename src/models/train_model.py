import argparse
import os
import sys
# from collections import Counter
from datetime import datetime
from typing import Dict

import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments, SentenceTransformerTrainer
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from src.data.create_corpus_dataset import DatasetSplitType, Query, Passage, PassageSource
from src.data.dataset_splits import create_splits_from_corpus_dataset
from src.evaluation.excluding_information_retrieval_evaluator import ExcludingInformationRetrievalEvaluator
from src.features.build_features import create_dataset_for_multiple_negatives_ranking_loss
from src.models.experiment_config import ExperimentConfig



def main(exp_config: ExperimentConfig, is_test_run=False):
    """
    Train a sentence transformer model with the given parameters and log the results
    to wandb.

    Args:
        exp_config: ExperimentConfig
        is_test_run: bool, defines if the training is a test run and only a small subset of the data is used
    """

    print("=== MAIN START ===")
    print(f"Experiment config:\n{exp_config}")
    print(f"Is test run? {is_test_run}")

    # Load environment variables
    env_path = exp_config.project_root + "/.env"
    print(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)
    api_key = os.getenv("WANDB_API_KEY")
    print("W&B API Key found?" if api_key else "W&B API Key NOT found!")

    print("Logging into Weights & Biases...")
    wandb.login(key=api_key)
    print("W&B login complete.")

    run_name = f"{exp_config.model_name_escaped}_lr{exp_config.learning_rate}_bs{exp_config.batch_size}_{exp_config.run_time}"
    print(f"Constructed run name: {run_name}")

    print("Initializing wandb run...")
    run = wandb.init(
        project="argument-classification",
        name=run_name,
        config=exp_config.model_dump(),
    )
    print("Wandb run initialized.")

    # Load model
    print(f"Loading model: {exp_config.model_name}")
    model = SentenceTransformer(exp_config.model_name)
    print("Model loaded successfully.")

    # Define loss
    print("Defining CachedMultipleNegativesRankingLoss...")
    loss = CachedMultipleNegativesRankingLoss(
        model=model,
        show_progress_bar=False,
        mini_batch_size=16
    )
    print("Loss defined.")

    # Load dataset
    dataset_path = f"{exp_config.project_root}/{exp_config.dataset_dir}/{exp_config.dataset_name}"
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print("Dataset loaded.")
    print("Creating dataset splits...")
    splitted_dataset = create_splits_from_corpus_dataset(dataset, exp_config.dataset_split_type)
    print("Dataset splits created.")

    # Prepare train and eval datasets
    print("Preparing train_pos and eval_pos...")
    if is_test_run:
        train_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["train"], 2)
        eval_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["validation"], 2)
        train_pos = train_pos.shuffle(seed=42).select(range(10))
        eval_pos = eval_pos.shuffle(seed=42).select(range(10))
        print("Using small subset for test run.")
        # Prepare evaluation data
        print("Preparing evaluation data structures...")
        eval_dataset = splitted_dataset["validation"]
        eval_dataset["passages"] = eval_dataset["passages"].shuffle(seed=42).select(range(400))
        eval_dataset["queries"] = eval_dataset["queries"].shuffle(seed=42).select(range(100))

        eval_passages = {row["id"]: Passage(row["id"], row["text"], row["label"], row["discussion_scenario"], row["passage_source"]) for row in eval_dataset["passages"]}
        eval_queries = {row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"]) for row in eval_dataset["queries"]}

        eval_relevant_passages = {
            row["query_id"]: {passage_id for passage_id in row["passages_ids"] if passage_id in eval_passages.keys()}
            for row in eval_dataset["queries_relevant_passages_mapping"]
            if row["query_id"] in eval_queries.keys()
        }
        eval_trivial_passages = {
            row["query_id"]: {passage_id for passage_id in row["passages_ids"] if passage_id in eval_passages.keys()}
            for row in eval_dataset["queries_trivial_passages_mapping"]
            if row["query_id"] in eval_queries.keys()
        }
    else:
        train_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["train"])
        eval_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["validation"])
        # Prepare evaluation data
        print("Preparing evaluation data structures...")
        eval_dataset = splitted_dataset["validation"]
        eval_passages = {row["id"]: row["text"] for row in eval_dataset["passages"]}
        eval_queries = {row["id"]: row["text"] for row in eval_dataset["queries"]}
        eval_relevant_passages = {
            row["query_id"]: set(row["passages_ids"])
            for row in eval_dataset["queries_relevant_passages_mapping"]
        }
        eval_trivial_passages = {
            row["query_id"]: set(row["passages_ids"])
            for row in eval_dataset["queries_trivial_passages_mapping"]
        }


        print("Full dataset used for training/validation.")

    print("Train/Eval datasets prepared.")

    print("Instantiating ExcludingInformationRetrievalEvaluator for eval...")
    excluding_ir_evaluator_eval = ExcludingInformationRetrievalEvaluator(
        corpus=eval_passages,
        queries=eval_queries,
        relevant_docs=eval_relevant_passages,
        excluded_docs=eval_trivial_passages,
        show_progress_bar=True,
        write_csv=True,
        log_top_k_predictions=5,
        run=run,
        project_root=exp_config.project_root,
    )
    print("ExcludingInformationRetrievalEvaluator for eval created.")

    # Prepare test data
    print("Preparing test data structures...")
    test_dataset = splitted_dataset["test"]
    test_passages = {row["id"]: row["text"] for row in test_dataset["passages"]}
    test_queries = {row["id"]: row["text"] for row in test_dataset["queries"]}
    test_relevant_passages = {
        row["query_id"]: set(row["passages_ids"])
        for row in test_dataset["queries_relevant_passages_mapping"]
    }
    test_trivial_passages = {
        row["query_id"]: set(row["passages_ids"])
        for row in test_dataset["queries_trivial_passages_mapping"]
    }

    print("Instantiating ExcludingInformationRetrievalEvaluator for test...")
    excluding_ir_evaluator_test = ExcludingInformationRetrievalEvaluator(
        corpus=test_passages,
        queries=test_queries,
        relevant_docs=test_relevant_passages,
        excluded_docs=test_trivial_passages,
        show_progress_bar=True,
        write_csv=True,
        log_top_k_predictions=5,
        run=run,
        project_root=exp_config.project_root,
    )
    print("ExcludingInformationRetrievalEvaluator for test created.")

    print("Setting up training arguments...")
    train_args = SentenceTransformerTrainingArguments(
        output_dir=exp_config.model_run_dir,
        num_train_epochs=exp_config.num_epochs,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=8,
        learning_rate=exp_config.learning_rate,
        warmup_ratio=0.1,
        fp16=(not is_test_run),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=4000,
        save_strategy="steps",
        save_steps=4000,
        save_total_limit=2,
        run_name=run_name,
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
    )
    print("Training arguments set.")

    print("Initializing CustomSentenceTransformerTrainer...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_pos,
        eval_dataset=eval_pos,
        loss=loss,
        evaluator=excluding_ir_evaluator_eval,
    )
    print("Trainer initialized.")

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    print("Running final evaluation on validation set...")
    final_eval_results = excluding_ir_evaluator_eval(model)
    print("Final evaluation results:", final_eval_results)
    run.log(final_eval_results)

    # print("Running evaluation on test set...")
    # test_eval_results = excluding_ir_evaluator_test(model)
    # print("Test evaluation results:", test_eval_results)
    # run.log(test_eval_results)

    # print("Saving model as W&B artifact...")
    # artifact = wandb.Artifact(name=run_name, type="model")
    # artifact.add_dir(exp_config.model_run_dir)
    # run.log_artifact(artifact)
    # print("Artifact logged.")

    print("Finishing wandb run...")
    run.finish()
    print("=== MAIN END ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentence transformer model.')
    parser.add_argument('--project_root', type=str, help='Path to the project root.')
    parser.add_argument('--experiment_dir', type=str, help='Path to the experiments directory.')
    parser.add_argument('--experiment_run', type=str, help='Name of the experiment run.')
    parser.add_argument('--dataset_dir', type=str, help='Path to the dataset directory.')
    parser.add_argument('--dataset_name', type=str, help='Path to the dataset.')
    parser.add_argument('--dataset_split_type', type=str, help='Type of dataset split to use.')
    parser.add_argument('--model_name', type=str, help='Name of the model to train.')
    parser.add_argument('--model_name_escaped', type=str, help='Name of the model to train with "/" replaced by "-".')
    parser.add_argument('--model_run_dir', type=str, help='Path to the model run directory.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate.')
    parser.add_argument('--batch_size', type=str, help='Batch size.')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs.')
    parser.add_argument('--loss_function', type=str, help='Name of the loss function to use.')
    parser.add_argument('--is_test_run', action='store_true', help='Run training in test mode.')
    args = parser.parse_args()

    sys.path.append("/home/erik/Documents/Uni/ethikchat-experiment-argument-classification")
    print("Extended PYTHONPATH:", sys.path)

    expirment_timestamp_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.model_name is None:
        print("Starting training in testing mode...")

        args_project_root = "/home/erik/Documents/Uni/ethikchat-experiment-argument-classification"
        args_experiment_run = "v1"
        args_experiment_dir = "experiments_outputs"
        args_dataset_dir = "data/processed"
        args_dataset_name = "corpus_dataset_v1"
        args_model_name = "deutsche-telekom/gbert-large-paraphrase-euclidean"
        args_model_name_escaped = args_model_name.replace("/", "-")
        args_learning_rate = 2e-05
        args_batch_size = 4
        args_model_run_dir = os.path.join(
            args_project_root,
            args_experiment_dir,
            args_experiment_run,
            f"{args_model_name_escaped}_lr{args_learning_rate}_bs{args_batch_size}"
        )

        experiment_config = {
            "project_root": args_project_root,
            "experiment_dir": args_experiment_dir,
            "experiment_run": args_experiment_run,
            "dataset_dir": args_dataset_dir,
            "dataset_name": args_dataset_name,
            "model_name": args_model_name,
            "model_name_escaped": args_model_name_escaped,
            "learning_rate": args_learning_rate,
            "batch_size": args_batch_size,
            "model_run_dir": args_model_run_dir,
            "dataset_split_type": DatasetSplitType.InDistribution,
            "num_epochs": 2,
            "loss_function": "MultipleNegativesRankingLoss",
            "run_time": expirment_timestamp_start
        }

        experiment_config = ExperimentConfig(**experiment_config)

        if not os.path.exists(args_model_run_dir):
            os.makedirs(args_model_run_dir)

        print(f"Running test training for model {args_model_name}")
        main(experiment_config, is_test_run=True)
    else:
        print("Starting training in standard mode...")
        if args.batch_size == "auto":
            args_batch_size = "auto"
        else:
            args_batch_size = int(args.batch_size)
        args_learning_rate = float(args.learning_rate)

        experiment_config = {
            "project_root": args.project_root,
            "experiment_dir": args.experiment_dir,
            "experiment_run": args.experiment_run,
            "dataset_dir": args.dataset_dir,
            "dataset_name": args.dataset_name,
            "model_name": args.model_name,
            "model_name_escaped": args.model_name_escaped,
            "learning_rate": args_learning_rate,
            "batch_size": args_batch_size,
            "model_run_dir": args.model_run_dir,
            "dataset_split_type": DatasetSplitType.from_str(args.dataset_split_type),
            "num_epochs": args.num_epochs,
            "loss_function": args.loss_function,
            "run_time": expirment_timestamp_start
        }

        experiment_config = ExperimentConfig(**experiment_config)

        main(experiment_config, is_test_run=args.is_test_run)
