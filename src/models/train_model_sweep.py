import os
import warnings
from typing import Dict, List

import wandb
import torch
from datasets import load_from_disk, DatasetDict, Dataset
from dotenv import load_dotenv
from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers
from transformers import PreTrainedTokenizer
from transformers.integrations import WandbCallback

from src.callbacks.custom_early_stopping_callback import EarlyStoppingWithLoggingCallback
from src.callbacks.wandb_logging_callback import WandbLoggingCallback
from src.data.dataset_splitting.dataset_splits import create_splits_from_corpus_dataset
from src.data.create_corpus_dataset import load_response_template_collection
from src.data.classes import Passage, Query, DatasetSplitType
from src.data_collation.custom_sentence_transformer_data_collator import CustomSentenceTransformerDataCollator
from src.evaluation.deep_dive_information_retrieval_evaluator import DeepDiveInformationRetrievalEvaluator
from src.evaluation.excluding_information_retrieval_evaluator import ExcludingInformationRetrievalEvaluator
from src.features.build_features import create_dataset_for_multiple_negatives_ranking_loss, add_context_to_texts, \
    add_scenario_tokens_to_texts, filter_queries_for_few_shot_setting, filter_passages_for_few_shot_setting
from src.losses.MaskedCachedMultipleNegativesRankingLoss import MaskedCachedMultipleNegativesRankingLoss, \
    MaskLoggingCallback
from src.models.experiment_config import ExperimentConfig


def load_argument_graphs(project_root, is_test_run=False) -> Dict[str, ResponseTemplateCollection]:
    argument_graph_med = load_response_template_collection("s1", project_root,
                                                           f"data/external/argument_graphs{'_test' if is_test_run else ''}")
    argument_graph_jur = load_response_template_collection("s2", project_root,
                                                           f"data/external/argument_graphs{'_test' if is_test_run else ''}")
    argument_graph_auto = load_response_template_collection("s3", project_root,
                                                            f"data/external/argument_graphs{'_test' if is_test_run else ''}")
    argument_graph_ref = load_response_template_collection("s4", project_root,
                                                           f"data/external/argument_graphs{'_test' if is_test_run else ''}")

    return {
        DiscussionSzenario.MEDAI.value: argument_graph_med,
        DiscussionSzenario.JURAI.value: argument_graph_jur,
        DiscussionSzenario.AUTOAI.value: argument_graph_auto,
        DiscussionSzenario.REFAI.value: argument_graph_ref
    }


def check_dataset_texts_for_truncation(tokenizer: PreTrainedTokenizer,
                                       split_dataset: DatasetDict,
                                       split_name: str,
                                       max_sequence_length: int,
                                       dataset_names: List[str] = ["queries", "passages"]) -> None:
    """
    Assumes a split_dataset with queries and passages and checks each of their "text" features if the encoded sequence
    length exceeds max_sequence_length.
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer of the model that is being fine-tuned
        split_dataset (): A Huggingface DatasetDict containing the datasets, e.g. "queries" and "passages"
        split_name: Name of of the split_dataset, e.g. "corpus_dataset" or "test" or "train"
        max_sequence_length (): The maximum sequence length of the fine-tuned model
        dataset_names:
    """

    def check_dataset_for_text_truncations(ds_name: str, dataset: Dataset):
        num_truncated_examples = 0
        for example in dataset:
            encoded_text = tokenizer.encode(example["text"])
            if len(encoded_text) > max_sequence_length:
                num_truncated_examples += 1
                warnings.warn(
                    f"Example with id {example['id']} in dataset {ds_name}-{split_name} exceeds length {max_sequence_length} ({len(encoded_text)}) and will be truncated during training!")
        print(
            f"{ds_name}-{split_name}: There are {num_truncated_examples} of {len(dataset)} examples that will be truncated during training.")

    for dataset_name in dataset_names:
        check_dataset_for_text_truncations(dataset_name, split_dataset[dataset_name])


def prepare_datasets(
        exp_config: ExperimentConfig,
        tokenizer: PreTrainedTokenizer,
        argument_graphs: Dict[str, ResponseTemplateCollection],
        maximum_sequence_length: int,
        is_test_run: bool = False
):
    """
    Loads the corpus dataset from disk, creates splits,
    and optionally uses smaller data for local testing if `is_test_run=True`.
    Returns all training data, evaluation data, and the evaluators.
    """
    # Load the dataset from disk
    corpus_dataset_path = os.path.join(exp_config.project_root, exp_config.dataset_dir, exp_config.dataset_name)
    corpus_dataset = load_from_disk(corpus_dataset_path)
    test_evaluator_name = "test_deepdive"
    test_evaluator_csv_output_dir = os.path.join(exp_config.model_run_dir, test_evaluator_name)
    os.makedirs(test_evaluator_csv_output_dir, exist_ok=True)

    # Create or load the splits
    split_dataset_folder = os.path.join(exp_config.project_root, exp_config.dataset_dir)
    splitted_dataset = create_splits_from_corpus_dataset(
        corpus_dataset=corpus_dataset,
        dataset_split_type=exp_config.dataset_split_type,
        test_scenario=exp_config.test_scenario,
        save_folder=split_dataset_folder,
        dataset_save_name=exp_config.split_dataset_name
    )
    tokenizer_sep_token = getattr(tokenizer, "sep_token", None)

    if tokenizer_sep_token is None:
        sep_token = "[SEP]"
    else:
        sep_token = tokenizer_sep_token

    train_split = splitted_dataset["train"]
    eval_split = splitted_dataset["validation"]
    test_split = splitted_dataset["test"]
    train_split = add_context_to_texts(train_split, exp_config.context_length, sep_token)
    eval_split = add_context_to_texts(eval_split, exp_config.context_length, sep_token)
    test_split = add_context_to_texts(test_split, exp_config.context_length, sep_token)

    if exp_config.add_discussion_scenario_info:
        train_split = add_scenario_tokens_to_texts(train_split)
        eval_split = add_scenario_tokens_to_texts(eval_split)
        test_split = add_scenario_tokens_to_texts(test_split)

    if exp_config.num_shots_queries > -1:
        train_split = filter_queries_for_few_shot_setting(train_split, exp_config.num_shots_queries)

    if exp_config.num_shots_passages > -1:
        train_split = filter_passages_for_few_shot_setting(train_split, exp_config.num_shots_passages)

    check_dataset_texts_for_truncation(tokenizer, train_split, "train", maximum_sequence_length)
    check_dataset_texts_for_truncation(tokenizer, eval_split, "eval", maximum_sequence_length)
    check_dataset_texts_for_truncation(tokenizer, test_split, "test", maximum_sequence_length)

    if not is_test_run:
        train_pos = create_dataset_for_multiple_negatives_ranking_loss(train_split,
                                                                       include_labels=exp_config.exclude_same_label_negatives)
        eval_pos = create_dataset_for_multiple_negatives_ranking_loss(eval_split,
                                                                      include_labels=exp_config.exclude_same_label_negatives)

        # Build references for EVAL
        eval_passages = {
            row["id"]: Passage(
                row["id"], row["text"], row["label"],
                row["discussion_scenario"], row["passage_source"]
            )
            for row in eval_split["passages"]
        }
        eval_queries = {
            row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"])
            for row in eval_split["queries"]
        }
        eval_relevant_passages = {
            row["query_id"]: set(row["passages_ids"])
            for row in eval_split["queries_relevant_passages_mapping"]
        }
        eval_trivial_passages = {
            row["query_id"]: set(row["passages_ids"])
            for row in eval_split["queries_trivial_passages_mapping"]
        }

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

    else:
        train_pos = create_dataset_for_multiple_negatives_ranking_loss(train_split,
                                                                       include_labels=exp_config.exclude_same_label_negatives)
        eval_pos = create_dataset_for_multiple_negatives_ranking_loss(eval_split,
                                                                      include_labels=exp_config.exclude_same_label_negatives)
        train_pos = train_pos.shuffle(seed=42).select(range(10))
        eval_pos = eval_pos.shuffle(seed=42).select(range(10))

        eval_queries = eval_split["queries"].shuffle(seed=42).select(range(10))
        eval_queries = {
            row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"])
            for row in eval_queries
        }
        eval_relevant_passages = {
            row["query_id"]: set(row["passages_ids"])
            for row in eval_split["queries_relevant_passages_mapping"]
            if row["query_id"] in eval_queries.keys()
        }
        # shorten the relevant passages to 2
        for key in eval_relevant_passages.keys():
            eval_relevant_passages[key] = set(list(eval_relevant_passages[key])[:2])

        eval_trivial_passages = {
            row["query_id"]: set(row["passages_ids"])
            for row in eval_split["queries_trivial_passages_mapping"]
        }

        # keep only relevant passages in the eval_passages dict
        all_eval_ids = set()
        for key in eval_relevant_passages.keys():
            all_eval_ids.update(eval_relevant_passages[key])
        eval_passages = {
            row["id"]: Passage(
                row["id"], row["text"], row["label"],
                row["discussion_scenario"], row["passage_source"]
            )
            for row in eval_split["passages"]
            if row["id"] in all_eval_ids
        }

        # 2) Build test references (small)
        test_queries = test_split["queries"].shuffle(seed=42).select(range(10))
        test_queries = {
            row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"])
            for row in test_queries
        }
        test_relevant_passages = {
            row["query_id"]: set(row["passages_ids"])
            for row in test_split["queries_relevant_passages_mapping"]
            if row["query_id"] in test_queries.keys()
        }
        # shorten the relevant passages to 2
        for key in test_relevant_passages.keys():
            test_relevant_passages[key] = set(list(test_relevant_passages[key])[:2])

        test_trivial_passages = {
            row["query_id"]: set(row["passages_ids"])
            for row in test_split["queries_trivial_passages_mapping"]
        }

        # keep only relevant passages in the test_passages dict
        all_test_ids = set()
        for key in test_relevant_passages.keys():
            all_test_ids.update(test_relevant_passages[key])
        test_passages = {
            row["id"]: Passage(
                row["id"], row["text"], row["label"],
                row["discussion_scenario"], row["passage_source"]
            )
            for row in test_split["passages"]
            if row["id"] in all_test_ids
        }
        noisy_queries = {
            row["id"]: Query(row["id"], row["text"], row["labels"], row["discussion_scenario"])
            for row in corpus_dataset["noisy_queries"]
        }

    # -------------------------------------------------
    # Create the actual evaluators with loaded references
    # -------------------------------------------------
    excluding_ir_evaluator_eval = ExcludingInformationRetrievalEvaluator(
        corpus=eval_passages,
        queries=eval_queries,
        accuracy_at_k=[1, 3, 5, 7, 10],
        relevant_docs=eval_relevant_passages,
        excluded_docs=eval_trivial_passages,
        show_progress_bar=True,
        write_csv=True,
        log_top_k_predictions=10,
        run=wandb.run,
        name="eval",
    )

    excluding_ir_evaluator_test = ExcludingInformationRetrievalEvaluator(
        corpus=test_passages,
        queries=test_queries,
        relevant_docs=test_relevant_passages,
        excluded_docs=test_trivial_passages,
        accuracy_at_k=[1, 3, 5, 7, 10],
        show_progress_bar=True,
        write_csv=True,
        log_top_k_predictions=10,
        run=wandb.run,
        name="test",
    )

    deep_dive_evaluator_test = DeepDiveInformationRetrievalEvaluator(
        corpus=test_passages,
        queries=test_queries,
        noisy_queries=noisy_queries,
        relevant_docs=test_relevant_passages,
        excluded_docs=test_trivial_passages,
        show_progress_bar=True,
        write_csv=True,
        run=wandb.run,
        argument_graphs=argument_graphs,
        confidence_threshold=0.7,
        confidence_threshold_steps=0.01,
        accuracy_at_k=[1, 3, 5, 7, 10],
        precision_at_k=[1, 3, 5, 7, 10],
        save_tables_as_csv=True,
        csv_output_dir=test_evaluator_csv_output_dir,
        name=test_evaluator_name
    )

    return (
        train_pos,
        eval_pos,
        excluding_ir_evaluator_eval,
        excluding_ir_evaluator_test,
        deep_dive_evaluator_test
    )


def main(is_test_run=False):
    # 1) Initialize W&B and read hyperparameters from wandb.config
    wandb.init(project="argument-classification")  # <--- adjust project name as needed
    config = wandb.config

    run_name = wandb.run.name if wandb.run.name else wandb.run.id
    sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "manual"
    sweep_run_name = f"{run_name}"
    print(f"W&B assigned run name: {sweep_run_name}")
    print(f"Cuda available: {torch.cuda.is_available()}")

    # 2) Load environment variables
    project_root = config.get("project_root", "/home/ls6/hauptmann/ethikchat-experiment-argument-classification")
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path)

    # 3) Login to W&B (key is usually read from env or netrc)
    wandb.login()

    test_scenario = config.test_scenario

    exp_config_dict = {
        "project_root": project_root,
        "experiment_dir": config.experiment_dir,
        "experiment_run": sweep_id,
        "dataset_dir": config.dataset_dir,
        "dataset_name": config.dataset_name,
        "model_name": config.model_name,
        "model_name_escaped": config.model_name.replace("/", "-"),
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "model_run_dir": os.path.join(project_root, config.experiment_dir, sweep_id, sweep_run_name),
        "dataset_split_type": DatasetSplitType.from_str(config.dataset_split_type),
        "split_dataset_name": config.dataset_split_name,
        "num_epochs": config.num_epochs,
        "loss_function": "MultipleNegativesRankingLoss",  # or config.get(...)
        "run_time": "sweep-run",
        "warmup_ratio": config.warmup_ratio,
        "context_length": config.context_length,
        "add_discussion_scenario_info": config.add_discussion_scenario_info,
        "test_scenario": test_scenario,
        "num_shots_passages": config.num_shots_passages,
        "num_shots_queries": config.num_shots_queries,
        "exclude_same_label_negatives": config.exclude_same_label_negatives
    }
    exp_config = ExperimentConfig(**exp_config_dict)

    print("\n\n ------------------------ Experiment configuration: ------------------------ ")
    for key, value in exp_config.model_dump().items():
        print(f"{key}: {value}")
    print(" ------------------------ ----------------------------- ------------------------ \n\n")

    # Make sure output directory exists
    os.makedirs(exp_config.model_run_dir, exist_ok=True)

    # 5) Load the model
    model = SentenceTransformer(exp_config.model_name)

    # 6) Define the loss
    if exp_config.exclude_same_label_negatives:
        loss = MaskedCachedMultipleNegativesRankingLoss(model=model,
                                                    show_progress_bar=True,
                                                    mini_batch_size=8,
                                                    exclude_same_label_negatives=exp_config.exclude_same_label_negatives)
    else:
        loss = CachedMultipleNegativesRankingLoss(model=model, show_progress_bar=True, mini_batch_size=8)

    # 7) Load argument graphs
    argument_graphs = load_argument_graphs(exp_config.project_root, is_test_run)

    # 8) Prepare train/eval/test data (including evaluators)
    (
        train_pos,
        eval_pos,
        excluding_ir_evaluator_eval,
        excluding_ir_evaluator_test,
        deep_dive_evaluator_test
    ) = prepare_datasets(exp_config, model.tokenizer, argument_graphs, model.max_seq_length, is_test_run=is_test_run)

    # 9) Pre-training evaluation on the eval set
    excluding_ir_evaluator_eval(model)

    # 10) Decide how often to evaluate and save
    # evaluate twice per epoch.
    eval_save_steps = int((len(train_pos) / config.batch_size) / 2)
    early_stopper = EarlyStoppingWithLoggingCallback(
        early_stopping_patience=7,  # you can change this value if needed
        early_stopping_threshold=0.001  # you can change this value if needed
    )

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
        save_steps=eval_save_steps if not is_test_run else 5,
        save_total_limit=2,
        run_name=f"sweep_{exp_config.model_name_escaped}",
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        metric_for_best_model="eval_cosine_accuracy@1",
        log_level="info",
        logging_steps=10,
        report_to="wandb",
    )
    callbacks = [early_stopper, WandbCallback(), WandbLoggingCallback()]
    data_collator = None

    if exp_config.exclude_same_label_negatives:
        data_collator = CustomSentenceTransformerDataCollator(
            tokenize_fn=model.tokenize,
            handle_specialized_label_columns=exp_config.exclude_same_label_negatives
        )
        callbacks.append(MaskLoggingCallback(loss, wandb.run))



    # 12) Create a trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_pos,
        eval_dataset=eval_pos,
        loss=loss,
        evaluator=excluding_ir_evaluator_eval,
        callbacks=callbacks,
        data_collator=data_collator
    )

    # 13) Train
    trainer.train()

    # 14) Final evaluation on the validation set
    excluding_ir_evaluator_eval(model)

    # 15) Evaluate on the test set
    excluding_ir_evaluator_test(model)

    # 16) Deep dive test evaluation
    deep_dive_evaluator_test(model)

    wandb.finish()


if __name__ == "__main__":
    import sys

    # If you pass a --local-test argument, we'll run with a dummy config in offline mode
    if "--local-test" in sys.argv:
        local_config = {
            "project_root": "/home/christian/PycharmProjects/ethikchat-experiment-argument-classification",
            "experiment_dir": "experiments_outputs",
            "experiment_run": "v1_local_debug",
            "dataset_dir": "data/processed/with_context",
            "dataset_name": "corpus_dataset_v3",
            "dataset_split_type": DatasetSplitType.InDistribution.value,
            "dataset_split_name": "dataset_split_in_distribution_from-v3",
            "model_name": "deutsche-telekom/gbert-large-paraphrase-euclidean",
            "learning_rate": 2e-5,
            "batch_size": 2,
            "num_epochs": 10,
            "warmup_ratio": 0.1,
            "context_length": 2,
            "add_discussion_scenario_info": True,
            "test_scenario": DiscussionSzenario.JURAI.value,
            "num_shots_queries": -1,
            "num_shots_passages": 23,
            "exclude_same_label_negatives": True
        }

        # 	add_discussion_scenario_info: True
        # 	batch_size: 128
        # 	context_length: 3
        # 	dataset_dir: data/processed/with_conte
        # 	dataset_name: corpus_dataset_v1
        # 	dataset_split_name: dataset_split_in_distribution
        # 	dataset_split_type: in_distribution
        # 	experiment_dir: experiments_outputs
        # 	learning_rate: 2e-05
        # 	model_name: T-Systems-onsite/cross-en-de-roberta-sentence-transformer
        # 	num_epochs: 8
        # 	num_shots_passages: 21
        # 	num_shots_queries: -1
        # 	project_root: /home/ls6/hauptmann/ethikchat-experiment-argument-classification
        # 	test_scenario: MEDAI
        # 	warmup_ratio: 0.1
        wandb.init(
            project="argument-classification",  # or "argument-classification-test"
            config=local_config,
            mode="online"
        )
        main(is_test_run=True)
    else:
        # Normal sweep entry point
        main()
