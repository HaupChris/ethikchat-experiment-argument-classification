#!/bin/bash

# Set the output directory name here
project_root_dir="/home/ls6/hauptmann/ethikchat-experiment-argument-classification"
experiment_dir="${project_root_dir}/experiments_outputs"
experiment_run="v1_pre_param_study"
experiment_run_dir="${experiment_dir}/${experiment_run}"
dataset_dir="data/processed"
dataset_split_type="simple"
num_epochs=8
loss_function="MultipleNegativesRankingLoss"

# add project root directory to python path
export PYTHONPATH="${project_root_dir}:${PYTHONPATH}"


# Initialize mode flags
TEST_MODE=0
CONT_MODE=0

# Check for command-line arguments
if [[ "$1" == "--test" ]]; then
  TEST_MODE=1
elif [[ "$1" == "--cont" ]]; then
  CONT_MODE=1
fi

# Activate the virtual environment
source /home/ls6/hauptmann/ethikchat-experiment-argument-classification/venv/bin/activate

model_names=(
  "google-bert/bert-base-uncased"
  "aari1995/German_Semantic_STS_V2"
  "aari1995/German_Semantic_V3b"
  "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
  "T-Systems-onsite/german-roberta-sentence-transformer-v2"
  "JoBeer/german-semantic-base"
  "deutsche-telekom/gbert-large-paraphrase-euclidea"
  "jinaai/jina-embeddings-v2-base-de"
)

# List of dataset names
dataset_names=(
  "corpus_dataset_experiment_v1"
)

# List of learning rates
learning_rates=(
  "2e-5"
)

batch_sizes=(
  "64"
)


# Initialize job counter
JOB_COUNTER=0

# Loop over each dataset name
for dataset_index in "${!dataset_names[@]}"; do
  dataset_name="${dataset_names[$dataset_index]}"
  for batch_size_index in "${!batch_sizes[@]}"; do # Inner loop over each model name
    batch_size="${batch_sizes[$batch_size_index]}"
    for model_index in "${!model_names[@]}"; do
      model_name="${model_names[$model_index]}"
      # Innermost loop over each learning rate
      for lr_index in "${!learning_rates[@]}"; do
        lr="${learning_rates[$lr_index]}"

        # Increment job counter
        JOB_COUNTER=$((JOB_COUNTER + 1))

        # Skip the first job if in continuation mode
        if [[ $CONT_MODE -eq 1 ]] && [[ $JOB_COUNTER -eq 1 ]]; then
          echo "Skipping the first job in continuation mode."
          continue
        fi

        # Create a sanitized model name for the job name (replace '/' with '_')
        model_name_escaped=$(echo "$model_name" | sed 's/\//_/g')

        # Define output directory
        model_run_dir="${experiment_run_dir}/${model_name_escaped}/lr${lr}/bs${batch_size}"


        # Ensure output directory exists
        mkdir -p "${model_run_dir}"

        # Define log file path
        log_file="${model_run_dir}/slurm_output.log"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${model_index}-${dataset_index}-${lr_index}d
#SBATCH --partition=ls6prio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=${log_file}
#SBATCH --error=${log_file}

# Activate the virtual environment
source ~/ethikchat-experiment-argument-classification/venv/bin/activate

# Print some debugging information
echo "Job started at: \$(date)"
echo "Running on node: \$(hostname)"
echo "Current directory: \$(pwd)"

# Run the training script with the current model name, dataset name, learning rate, and combined output directory
# Run the training script with all necessary parameters
python train_model.py \
  --project_root "$project_root_dir" \
  --experiment_dir "$experiment_dir" \
  --experiment_run "$experiment_run" \
  --dataset_dir "$dataset_dir" \
  --dataset_name "$dataset_name" \
  --model_name "$model_name" \
  --model_name_escaped "$model_name_escaped" \
  --learning_rate "$lr" \
  --batch_size "$batch_size" \
  --model_run_dir "$model_run_dir" \
  --dataset_split_type "$dataset_split_type" \
  --num_epochs "$num_epochs" \
  --loss_function "$loss_function"
  $([[ $TEST_MODE -eq 1 ]] && echo "--is_test_run")  # Add the flag if TEST_MODE is 1

# Print job completion message
echo "Job completed at: \$(date)"
EOF

        # If in test mode, exit after submitting one job
        if [[ $TEST_MODE -eq 1 ]]; then
          echo "Test job submitted. Exiting."
          exit 0
        fi
      done
    done
  done
done
