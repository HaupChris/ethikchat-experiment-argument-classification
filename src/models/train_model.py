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

# Load a model to train/finetune
model_name = "airnicco8/xlm-roberta-de"
model_name_escaped = model_name.replace("/", "-")
experiment_name = f"{model_name_escaped}-contrastive-loss"
model = SentenceTransformer(model_name)

# This loss requires pairs of text and a float similarity score as a label
loss = MultipleNegativesRankingLoss(model=model)

# Load an example training dataset that works with our loss function:
dataset = load_from_disk("../../data/processed/corpus_dataset_experiment_v0")
splitted_dataset = create_splits_from_corpus_dataset(dataset, DatasetSplitType.Simple)

train_dataset = splitted_dataset["train"]
eval_dataset = splitted_dataset["validation"]
test_dataset = splitted_dataset["test"]

# create datastructres for the InformationRetrievalEvaluator
# corpus = dict[passage_id, passage_text]
# queries = dict[query_id, query_text]
# relevant_passages = dict[query_id, set[passage_id]]

eval_corpus = {row["id"]: row["text"] for row in eval_dataset["passages"]}
eval_queries = {row["id"]: row["text"] for row in eval_dataset["queries"]}
eval_relevant_passages = {row["query_id"]: set(row["passages_ids"])
                        for row in eval_dataset["queries_relevant_passages_mapping"]}   # only scenario 1

ir_evaluator_eval = InformationRetrievalEvaluator(
    corpus=eval_corpus,
    queries=eval_queries,
    relevant_docs=eval_relevant_passages,
    name=f"{experiment_name}-eval",
    show_progress_bar=True,
    write_csv=True,

)

ir_evaluator_eval(model)


if IS_TEST_RUN:
    train_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["train"],1)
    eval_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["validation"], 1)
    train_pos = train_pos.shuffle(seed=42).select(range(10))
    eval_pos = eval_pos.shuffle(seed=42).select(range(10))

else:
    train_pos = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["train"])
    eval_pos  = create_dataset_for_multiple_negatives_ranking_loss(splitted_dataset["validation"])





args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"../../models/{experiment_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=29,
    run_name=experiment_name,  # Will be used in W&B if `wandb` is installed
    load_best_model_at_end=True,
)



# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_pos,
    eval_dataset=eval_pos,
    loss=loss,

)
trainer.train()

# Evaluator Klassen, die hier sinnvoll sind: InformationRetrievalEvaluator, TripletEvaluator, vielleicht für einen Crossencoder auch RerankingEvaluator
