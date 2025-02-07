#!/bin/bash

# Set the output directory name here
OUTPUT_DIR_NAME="output_crf_experiment_no_german_chars_correct_context"

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
source ~/ethikchat-experiment-argument-classification/venv/bin/activate

model_names=(
  "airnicco8/xlm-roberta-de"
  "google-bert/bert-base-german-cased"
)

# List of dataset names
dataset_names=(
  "corpus_dataset_experiment_v0"
)

# List of learning rates
learning_rates=(
  "1e-5"
)

# Initialize job counter
JOB_COUNTER=0

# Loop over each dataset name
for dataset_index in "${!dataset_names[@]}"; do
  dataset_name="${dataset_names[$dataset_index]}"
  # Inner loop over each model name
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
      sanitized_model_name=$(echo "$model_name" | sed 's/\//_/g')

      # Define output directory
      output_dir="${OUTPUT_DIR_NAME}/${dataset_name}/${sanitized_model_name}/lr${lr}"

      # Ensure output directory exists
      mkdir -p "$output_dir"
      mkdir -p "${output_dir}/train_data_eval"

      # Define log file path
      log_file="${output_dir}/slurm_output.log"

      sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${moel_index}-${dataset_index}-${lr_index}d
#SBATCH --partition=ls6prio
#SBATCH --time=02:00:00
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
python train_model.py --model_name "$model_name" --dataset_name "$dataset_name" --learning_rate "$lr" --output_dir "$output_dir"

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
