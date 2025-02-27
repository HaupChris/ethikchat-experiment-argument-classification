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

from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from src.custom_trainer.custom_sentence_transformer_trainer import CustomSentenceTransformerTrainer
from src.data.create_corpus_dataset import DatasetSplitType
from src.data.dataset_splits import create_splits_from_corpus_dataset
from src.data_collation.custom_sentence_transformer_data_collator import CustomSentenceTransformerDataCollator
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
    # login to wandb by loading the API key from the .env file
    load_dotenv(exp_config.project_root + "/.env")
    api_key = os.getenv("WANDB_API_KEY")

    wandb.login(key=api_key)

    run_name = f"{exp_config.model_name_escaped}_lr{exp_config.learning_rate}_bs{exp_config.batch_size}_{exp_config.run_time}"

    # Initialize wandb
    run = wandb.init(
        project="argument-classification",
        name=f"{exp_config.model_name_escaped}_lr{exp_config.learning_rate}_bs{exp_config.batch_size}_{exp_config.run_time}",
        # Sync run name with wandb
        config=exp_config.model_dump(),
    )

    # Load model
    model = SentenceTransformer(exp_config.model_name)

    # Define loss
    loss = CachedMultipleNegativesRankingLoss(model=model,
                                              show_progress_bar=False,
                                              mini_batch_size=256)

    # Load dataset
    dataset = load_from_disk(f"{exp_config.project_root}/{exp_config.dataset_dir}/{exp_config.dataset_name}")
    splitted_dataset = create_splits_from_corpus_dataset(dataset, exp_config.dataset_split_type)

    if is_test_run:
        train_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["train"], 1)
        eval_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["validation"], 1)
        # train_pos = train_pos.shuffle(seed=42).select(range(10))
        # eval_pos = eval_pos.shuffle(seed=42).select(range(10))
    else:
        train_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["train"])
        eval_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["validation"])

    # Prepare evaluation data
    eval_dataset = splitted_dataset["validation"]
    eval_passages = {row["id"]: row["text"] for row in eval_dataset["passages"]}
    eval_queries = {row["id"]: row["text"] for row in eval_dataset["queries"]}
    eval_relevant_passages = {row["query_id"]: set(row["passages_ids"])
                              for row in eval_dataset["queries_relevant_passages_mapping"]}
    eval_trivial_passages = {row["query_id"]: set(row["passages_ids"])
                             for row in eval_dataset["queries_trivial_passages_mapping"]}

    excluding_ir_evaluator_eval = ExcludingInformationRetrievalEvaluator(
        corpus=eval_passages,
        queries=eval_queries,
        relevant_docs=eval_relevant_passages,
        excluded_docs=eval_trivial_passages,
        show_progress_bar=True,
        write_csv=True,
        log_top_k_predictions=5,  # log the top 5 docs per query
        run=run,
        query_labels={row["id"]: row["labels"] for row in eval_dataset["queries"]},
        doc_labels={row["id"]: row["label"] for row in eval_dataset["passages"]},
        log_label_confusion=True,
        log_fp_fn=True,
        log_rank_histogram=True,
        log_multilabel_coverage=True,
        log_tsne_embeddings=True,
        tsne_sample_size=1000,
    )


    # prepare test data
    test_dataset = splitted_dataset["test"]
    test_passages = {row["id"]: row["text"] for row in test_dataset["passages"]}
    test_queries = {row["id"]: row["text"] for row in test_dataset["queries"]}
    test_relevant_passages = {row["query_id"]: set(row["passages_ids"])
                                for row in test_dataset["queries_relevant_passages_mapping"]}
    test_trivial_passages = {row["query_id"]: set(row["passages_ids"])
                                for row in test_dataset["queries_trivial_passages_mapping"]}


    excluding_ir_evaluator_test = ExcludingInformationRetrievalEvaluator(
        corpus=test_passages,
        queries=test_queries,
        relevant_docs=test_relevant_passages,
        excluded_docs=test_trivial_passages,
        show_progress_bar=True,
        write_csv=True,
        log_top_k_predictions=5,  # log the top 5 docs per query
        run=run,
        query_labels={row["id"]: row["labels"] for row in test_dataset["queries"]},
        doc_labels={row["id"]: row["label"] for row in test_dataset["passages"]},
        log_label_confusion=True,
        log_fp_fn=True,
        log_rank_histogram=True,
        log_multilabel_coverage=True,
        log_tsne_embeddings=True,
        tsne_sample_size=1000,
    )

    pre_run_eval_results = excluding_ir_evaluator_eval(model)
    run.log(pre_run_eval_results)

    train_args = SentenceTransformerTrainingArguments(
        output_dir=exp_config.model_run_dir,
        num_train_epochs=exp_config.num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=exp_config.learning_rate,
        warmup_ratio=0.1,
        fp16=(not is_test_run),
        bf16=(not is_test_run),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=4000,
        save_strategy="steps",
        save_steps=4000,
        save_total_limit=2,
        run_name=run_name,  # Sync run name with wandb
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
    )


    trainer = CustomSentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_pos,
        eval_dataset=eval_pos,
        loss=loss,
        evaluator=excluding_ir_evaluator_eval,
        data_collator=CustomSentenceTransformerDataCollator(tokenize_fn=model.tokenize, skip_columns=["labels"]),
    )

    trainer.train()

    # Run final evaluation
    final_eval_results = excluding_ir_evaluator_eval(model)
    run.log(final_eval_results)

    # Run test evaluation
    test_eval_results = excluding_ir_evaluator_test(model)
    run.log(test_eval_results)

    # Save model to wandb as an artifact

    artifact = wandb.Artifact(name=run_name, type="model")
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
    parser.add_argument('--is_test_run', action='store_true', help='Run training in test mode.')
    args = parser.parse_args()

    sys.path.append("/home/ls6/hauptmann/ethikchat-experiment-argument-classification")
    print(sys.path)

    expirment_timestamp_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.model_name is None:
        print("Starting training in testing mode...")

        args_project_root = "/home/christian/PycharmProjects/ethikchat-experiment-argument-classification"
        args_experiment_run = "v1"
        args_experiment_dir = "experiments_outputs"
        args_dataset_dir = "data/processed"
        args_dataset_name = "corpus_dataset_experiment_v1"
        args_model_name = "deutsche-telekom/gbert-large-paraphrase-euclidean"
        args_model_name_escaped = args_model_name.replace("/", "-")
        args_learning_rate = 2e-05
        args_batch_size = 4
        args_model_run_dir = os.path.join(args_project_root, args_experiment_dir, args_experiment_run,
                                          f"{args_model_name_escaped}_lr{args_learning_rate}_bs{args_batch_size}")

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
