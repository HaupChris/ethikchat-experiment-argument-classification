import os
from typing import Dict

import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers

from src.data.dataset_splits import create_splits_from_corpus_dataset
from src.data.create_corpus_dataset import DatasetSplitType, Passage, Query, load_response_template_collection
from src.evaluation.excluding_information_retrieval_evaluator import ExcludingInformationRetrievalEvaluator
from src.features.build_features import create_dataset_for_multiple_negatives_ranking_loss
from src.models.experiment_config import ExperimentConfig




def load_argument_graphs(project_root) -> Dict[str, ResponseTemplateCollection]:
    argument_graph_med = load_response_template_collection("s1", project_root)
    argument_graph_jur = load_response_template_collection("s2", project_root)
    argument_graph_auto = load_response_template_collection("s3", project_root)
    argument_graph_ref = load_response_template_collection("s4", project_root)

    argument_graphs = {
        DiscussionSzenario.MEDAI.value: argument_graph_med,
        DiscussionSzenario.JURAI.value: argument_graph_jur,
        DiscussionSzenario.AUTOAI.value: argument_graph_auto,
        DiscussionSzenario.REFAI.value: argument_graph_ref
    }

    return argument_graphs


def main(is_test_run=False):
    # 1) Initialize W&B and read hyperparameters from wandb.config
    wandb.init(project="argument-classification")  # <--- adjust project name as needed
    config = wandb.config

    run_name = wandb.run.name if wandb.run.name else wandb.run.id  # e.g. "sandy-sweep-49"
    sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "manual"
    sweep_run_name = f"{run_name}"
    print(f"W&B assigned run name: {sweep_run_name}")

    # 2) Load environment variables
    project_root = config.get("project_root", "/home/ls6/hauptmann/ethikchat-experiment-argument-classification")
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path)

    # 3) Print debug info
    print(f"Project Root: {project_root}")
    print(f"Using model: {config.model_name}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Num Epochs: {config.num_epochs}")
    print(f"run_name: {sweep_run_name}")

    # 4) Login to W&B (key is usually read from env or netrc)
    wandb.login()

    # 5) Prepare training configuration
    exp_config_dict = {
        "project_root": project_root,
        "experiment_dir": config.experiment_dir,  # e.g., "experiments_outputs"
        "experiment_run": sweep_id,  # e.g., <sweep_id> if it is a sweep run else "manual"
        "dataset_dir": config.dataset_dir,  # e.g., "data/processed"
        "dataset_name": config.dataset_name,  # e.g., "corpus_dataset_experiment_v1"
        "model_name": config.model_name,
        "model_name_escaped": config.model_name.replace("/", "-"),
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "model_run_dir": os.path.join(project_root, config.experiment_dir, sweep_id, sweep_run_name),
        "dataset_split_type": DatasetSplitType.from_str(config.dataset_split_type),
        "split_dataset_name": config.dataset_split_name,
        "num_epochs": config.num_epochs,
        "loss_function": "MultipleNegativesRankingLoss",  # or config.get(...)
        "run_time": "sweep-run",  # just a placeholder
        "warmup_ratio": config.warmup_ratio

    }

    exp_config = ExperimentConfig(**exp_config_dict)

    # Make sure output directory exists
    os.makedirs(exp_config.model_run_dir, exist_ok=True)

    model = SentenceTransformer(exp_config.model_name)

    #  Define the loss
    loss = CachedMultipleNegativesRankingLoss(model=model, show_progress_bar=False, mini_batch_size=8)

    # argument_graphs = load_argument_graphs(exp_config.project_root)

    # Load dataset
    corpus_dataset_path = os.path.join(exp_config.project_root, exp_config.dataset_dir, exp_config.dataset_name)
    corpus_dataset = load_from_disk(corpus_dataset_path)

    split_dataset_folder = os.path.join(exp_config.project_root, exp_config.dataset_dir)

    splitted_dataset = create_splits_from_corpus_dataset(corpus_dataset=corpus_dataset,
                                                         dataset_split_type=exp_config.dataset_split_type,
                                                         save_folder=split_dataset_folder,
                                                         dataset_save_name=exp_config.split_dataset_name
                                                         )

    # # 9) Prepare train/eval data
    train_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["train"])
    eval_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["validation"])

    eval_dataset = splitted_dataset["validation"]
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
    # Prepare test data/evaluator
    test_dataset = splitted_dataset["test"]
    test_passages = {
        row["id"]: Passage(row["id"], row["text"], row["label"], row["discussion_scenario"], row["passage_source"]) for
        row in test_dataset["passages"]}

    test_queries = {row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"]) for row in
                    test_dataset["queries"]}
    test_relevant_passages = {
        row["query_id"]: set(row["passages_ids"]) for row in test_dataset["queries_relevant_passages_mapping"]
    }
    test_trivial_passages = {
        row["query_id"]: set(row["passages_ids"]) for row in test_dataset["queries_trivial_passages_mapping"]
    }

    ######################### smaller datasets for testing purposes #####################################
    # train_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["train"], 2)
    # eval_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["validation"], 2)
    # train_pos = train_pos.shuffle(seed=42).select(range(10))
    # eval_pos = eval_pos.shuffle(seed=42).select(range(10))
    #
    # eval_dataset = splitted_dataset["validation"]
    # eval_queries = eval_dataset["queries"].shuffle(seed=42).select(range(10))
    # eval_queries = {row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"]) for row in
    #                 eval_queries}
    #
    # eval_relevant_passages = {
    #     row["query_id"]: set(row["passages_ids"]) for row in eval_dataset["queries_relevant_passages_mapping"]
    #     if row["query_id"] in eval_queries.keys()
    # }
    #
    # # shorten the relevant passages to 2
    # for key in eval_relevant_passages.keys():
    #     eval_relevant_passages[key] = set(list(eval_relevant_passages[key])[:2])
    #
    # eval_trivial_passages = {
    #     row["query_id"]: set(row["passages_ids"]) for row in eval_dataset["queries_trivial_passages_mapping"]
    # }
    #
    # # adjust the passages set. Only keep the passage_ids that are in the relevant passages
    # all_relevant_passag_ids = set()
    # for key in eval_relevant_passages.keys():
    #     all_relevant_passag_ids.update(eval_relevant_passages[key])
    #
    # eval_passages = {
    #     row["id"]: Passage(row["id"], row["text"], row["label"], row["discussion_scenario"], row["passage_source"]) for
    #     row in eval_dataset["passages"]}
    #
    # test_dataset = splitted_dataset["test"]
    # test_queries = test_dataset["queries"].shuffle(seed=42).select(range(10))
    # test_queries = {row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"]) for row in
    #                 test_queries}
    #
    #
    # test_relevant_passages = {
    #     row["query_id"]: set(row["passages_ids"]) for row in test_dataset["queries_relevant_passages_mapping"]
    #     if row["query_id"] in test_queries.keys()
    # }
    # # shorten the relevant passages to 2
    # for key in test_relevant_passages.keys():
    #     test_relevant_passages[key] = set(list(test_relevant_passages[key])[:2])
    #
    # test_trivial_passages = {
    #     row["query_id"]: set(row["passages_ids"]) for row in test_dataset["queries_trivial_passages_mapping"]
    # }
    #
    # # adjust the passages set. Only keep the passage_ids that are in the relevant passages
    # all_relevant_passag_ids = set()
    # for key in test_relevant_passages.keys():
    #     all_relevant_passag_ids.update(test_relevant_passages[key])
    #
    # test_passages = {
    #     row["id"]: Passage(row["id"], row["text"], row["label"], row["discussion_scenario"], row["passage_source"]) for
    #     row in test_dataset["passages"] if row["id"] in all_relevant_passag_ids}


    #####################################################################################################

    excluding_ir_evaluator_eval = ExcludingInformationRetrievalEvaluator(
        corpus=eval_passages,
        queries=eval_queries,
        relevant_docs=eval_relevant_passages,
        excluded_docs=eval_trivial_passages,
        show_progress_bar=True,
        write_csv=True,
        log_top_k_predictions=5,
        run=wandb.run,
    )

    excluding_ir_evaluator_test = ExcludingInformationRetrievalEvaluator(
        corpus=test_passages,
        queries=test_queries,
        relevant_docs=test_relevant_passages,
        excluded_docs=test_trivial_passages,
        show_progress_bar=True,
        write_csv=True,
        log_top_k_predictions=5,
        run=wandb.run,
    )

    pretrain_eval_results = excluding_ir_evaluator_eval(model)
    prefixed_pretrain_eval_results = {f"eval_{key}": value for key, value in pretrain_eval_results.items()}
    wandb.log(prefixed_pretrain_eval_results)

    eval_save_steps = 4000 / (exp_config.batch_size / 32)

    # 11) Training arguments
    train_args = SentenceTransformerTrainingArguments(
        output_dir=exp_config.model_run_dir,
        num_train_epochs=exp_config.num_epochs,
        per_device_train_batch_size=exp_config.batch_size,
        per_device_eval_batch_size=8,
        learning_rate=exp_config.learning_rate,
        warmup_ratio=exp_config.warmup_ratio,
        fp16=(not is_test_run),
        eval_strategy="steps",
        eval_steps=eval_save_steps if not is_test_run else 5,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=2,
        run_name=f"sweep_{exp_config.model_name_escaped}",
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        batch_sampler=BatchSamplers.NO_DUPLICATES
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_pos,
        eval_dataset=eval_pos,
        loss=loss,
        evaluator=excluding_ir_evaluator_eval,
    )

    # 12) Train
    trainer.train()

    # 13) Evaluate
    final_eval_results = excluding_ir_evaluator_eval(model)
    prefixed_eval_results = {f"eval_{key}": value for key, value in final_eval_results.items()}
    wandb.log(prefixed_eval_results)

    test_eval_results = excluding_ir_evaluator_test(model)
    prefixed_test_eval_results = {f"test_{key}": value for key, value in test_eval_results.items()}
    wandb.log(prefixed_test_eval_results)

    wandb.finish()


if __name__ == "__main__":
    import sys

    # If you pass a --local-test argument, we'll run with a dummy config in offline mode
    if "--local-test" in sys.argv:
        # Example static config (pick simple/fast values)
        local_config = {
            "project_root": "/home/christian/PycharmProjects/ethikchat-experiment-argument-classification",
            "experiment_dir": "experiments_outputs",
            "experiment_run": "v1_local_debug",
            "dataset_dir": "data/processed",
            "dataset_name": "corpus_dataset_v1",
            "dataset_split_type": "in_distribution",
            "dataset_split_name": "dataset_split_in_distribution",
            "model_name": "deutsche-telekom/gbert-large-paraphrase-euclidean",
            "learning_rate": 1e-5,
            "batch_size": 2,
            "num_epochs": 2,
            "warmup_ratio": 0.1
        }

        wandb.init(
            project="argument-classification",  # or "argument-classification-test"
            config=local_config,
            mode="online"
        )
        main(is_test_run=True)
    else:
        # The normal sweep entry point
        main()
