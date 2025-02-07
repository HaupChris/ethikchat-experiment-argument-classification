import argparse
import wandb

from datasets import load_from_disk
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
from src.features.build_features import create_dataset_for_multiple_negatives_ranking_loss

IS_TEST_RUN = True


def main(model_name, dataset_name, output_dir, learning_rate):
    # Initialize wandb
    wandb.init(
        project="sentence-transformers-training",
        name=output_dir,  # Set a meaningful name for this run
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "learning_rate": learning_rate,
            "batch_size": 4,
            "epochs": 1,
        }
    )

    # Load model
    model = SentenceTransformer(model_name)

    # Define loss
    loss = MultipleNegativesRankingLoss(model=model)

    # Load dataset
    dataset = load_from_disk(f"../../data/processed/{dataset_name}")
    splitted_dataset = create_splits_from_corpus_dataset(dataset, DatasetSplitType.Simple)




    if IS_TEST_RUN:
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

    ir_evaluator_eval = InformationRetrievalEvaluator(
        corpus=eval_corpus,
        queries=eval_queries,
        relevant_docs=eval_relevant_passages,
        name=f"{output_dir}-eval",
        show_progress_bar=True,
        write_csv=True,
    )

    ir_evaluator_eval(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=f"../../models/{output_dir}",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=29,
        run_name=output_dir,  # Sync run name with wandb
        load_best_model_at_end=True
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_pos,
        eval_dataset=eval_pos,
        loss=loss,
        evaluator=ir_evaluator_eval
    )

    trainer.train()

    # Run final evaluation
    ir_evaluator_eval(model)

    # Save model to wandb as an artifact
    model_dir = f"../../models/{output_dir}"
    artifact = wandb.Artifact(name=f"{output_dir}-model", type="model")
    artifact.add_dir(model_dir)
    wandb.log_artifact(artifact)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentence transformer model.')
    parser.add_argument('--model_name', type=str, help='Name of the model to train.')
    parser.add_argument('--dataset_name', type=str, help='Path to the dataset.')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate.')
    args = parser.parse_args()

    if args.model_name is None:
        print("Starting training in testing mode...")

        model_name = "airnicco8/xlm-roberta-de"
        dataset_name = "corpus_dataset_experiment_v0"
        learning_rate = 2e-05

        model_name_escaped = model_name.replace("/", "-")
        output_dir = f"output_test_crf/{dataset_name}/{model_name_escaped}/lr{learning_rate}"

        print(f"Running test training for model {model_name}")
        main(model_name, dataset_name, output_dir, learning_rate)
    else:
        main(args.model_name, args.dataset_name, args.output_dir, args.learning_rate)
