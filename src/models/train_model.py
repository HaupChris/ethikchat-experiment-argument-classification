from datasets import load_dataset, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)

from sentence_transformers.losses import ContrastiveLoss
from sentence_transformers.training_args import BatchSamplers

from src.features.build_features import build_contrastive_dataset

# Load a model to train/finetune
model_name = "airnicco8/xlm-roberta-de"
model_name_escaped = model_name.replace("/", "-")
experiement_name = f"{model_name_escaped}-contrastive-loss"
model = SentenceTransformer(model_name)

# Initialize the CoSENTLoss
# This loss requires pairs of text and a float similarity score as a label
loss = ContrastiveLoss(model=model)

# Load an example training dataset that works with our loss function:
dataset = load_from_disk("../data/dummy_dataset")
contrastive_loss_dataset = build_contrastive_dataset(dataset, max_ratio_negatives_to_positives=None)
train_dataset = contrastive_loss_dataset["train"]
eval_dataset = contrastive_loss_dataset["validation"]
test_dataset = contrastive_loss_dataset["test"]

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"../../models/{experiement_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
)

