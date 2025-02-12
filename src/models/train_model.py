import argparse
import os
import sys
from datetime import datetime

import wandb

from datasets import load_from_disk
from dotenv import load_dotenv
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from src.data.create_corpus_dataset import DatasetSplitType
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
    """
    # login to wandb by loading the API key from the .env file
    load_dotenv(exp_config.project_root + "/.env")
    api_key=os.getenv("WANDB_API_KEY")

    wandb.login(key=api_key)

    # Initialize wandb
    run = wandb.init(
        project="argument-classification",
        name=f"{exp_config.model_name_escaped}_lr{exp_config.learning_rate}_bs{exp_config.batch_size}_{exp_config.run_time}",  # Sync run name with wandb
        config=exp_config.model_dump(),
    )

    # Load model
    model = SentenceTransformer(exp_config.model_name)

    # Define loss
    loss = MultipleNegativesRankingLoss(model=model)

    # Load dataset
    dataset = load_from_disk(f"{exp_config.project_root}/{exp_config.dataset_dir}/{exp_config.dataset_name}")
    splitted_dataset = create_splits_from_corpus_dataset(dataset, exp_config.dataset_split_type)

    if is_test_run:
        train_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["train"], 1)
        eval_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["validation"], 1)
        train_pos = train_pos.shuffle(seed=42).select(range(10))
        eval_pos = eval_pos.shuffle(seed=42).select(range(10))
    else:
        train_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["train"])
        eval_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["validation"])

    # Prepare evaluation data
    eval_dataset = splitted_dataset["validation"]
    eval_corpus = {row["id"]: row["text"] for row in eval_dataset["passages"]}
    eval_queries = {row["id"]: row["text"] for row in eval_dataset["queries"]}
    eval_relevant_passages = {row["query_id"]: set(row["passages_ids"])
                              for row in eval_dataset["queries_relevant_passages_mapping"]}
    eval_trivial_passages = {row["query_id"]: set(row["passages_ids"])
                                for row in eval_dataset["queries_trivial_passages_mapping"]}

    ir_evaluator_eval = InformationRetrievalEvaluator(
        corpus=eval_corpus,
        queries=eval_queries,
        relevant_docs=eval_relevant_passages,
        show_progress_bar=True,
        write_csv=True,
    )

    excluding_ir_evaluator_eval = ExcludingInformationRetrievalEvaluator(
        corpus=eval_corpus,
        queries=eval_queries,
        relevant_docs=eval_relevant_passages,
        excluded_docs=eval_trivial_passages,
        show_progress_bar=True,
        write_csv=True,
        log_top_k_predictions=5,  # log the top 5 docs per query
        run=run
    )

    excluding_ir_evaluator_eval(model)


    train_args = SentenceTransformerTrainingArguments(
        output_dir=exp_config.model_run_dir,
        num_train_epochs=exp_config.num_epochs,
        auto_find_batch_size=True,
        learning_rate=exp_config.learning_rate,
        warmup_ratio=0.1,
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=4000,
        save_strategy="epoch",
        save_total_limit=2,
        run_name=f"{exp_config.model_name_escaped}_lr{exp_config.learning_rate}_bs{exp_config.batch_size}_{exp_config.run_time}",  # Sync run name with wandb
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=1,

    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_pos,
        eval_dataset=eval_pos,
        loss=loss,
        evaluator=excluding_ir_evaluator_eval
    )

    trainer.train()

    # Run final evaluation
    final_eval_results = excluding_ir_evaluator_eval(model)
    run.log(final_eval_results)

    # Save model to wandb as an artifact

    artifact = wandb.Artifact(name=f"{exp_config.model_name_escaped}_lr{exp_config.learning_rate}_bs{exp_config.batch_size}_{exp_config.run_time}", type="model")
    artifact.add_dir(exp_config.model_run_dir)
    run.log_artifact(artifact)

    # save the slurm output to wandb

    # Finish wandb run
    run.finish()


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
    args = parser.parse_args()

    sys.path.append("/home/ls6/hauptmann/ethikchat-experiment-argument-classification")
    print(sys.path)

    expirment_timestamp_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.model_name is None:
        print("Starting training in testing mode...")

        args_project_root = "/home/christian/PycharmProjects/ethikchat-experiment-argument-classification"
        args_experiment_run = "v0"
        args_experiment_dir = "experiments_outputs"
        args_dataset_dir = "data/processed"
        args_dataset_name = "corpus_dataset_experiment_v0"
        args_model_name = "airnicco8/xlm-roberta-de"
        args_model_name_escaped = args_model_name.replace("/", "-")
        args_learning_rate = 2e-05
        args_batch_size = 4
        args_model_run_dir = os.path.join(args_project_root, args_experiment_dir, args_experiment_run, f"{args_model_name_escaped}_lr{args_learning_rate}_bs{args_batch_size}")


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
            "dataset_split_type": DatasetSplitType.Simple,
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

        main(experiment_config)
