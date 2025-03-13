#!/bin/bash
#SBATCH --job-name=resume_model_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=resume_run_%A_%a.out
#SBATCH --error=resume_run_%A_%a.err

# Define project and model directory paths
project_root="/home/stud/spiegel/ethikchat-experiment-argument-classification"
models_dir="${project_root}/models"

# Define lists for run_ids and model_paths
run_ids=(
  "run1"
  "run2"
  "run3"
)

model_paths=(
  "model1"
  "model2"
  "model3"
)

# Convert arrays to space-separated strings
run_ids_str="${run_ids[*]}"
model_paths_str="${model_paths[*]}"

# Export project_root to PYTHONPATH for the Python script
export PYTHONPATH="${project_root}:${PYTHONPATH}"

# Activate the virtual environment
source "${project_root}/venv/bin/activate"

# Run the Python script with the lists converted to arguments
python resume_model_run.py --run_ids "$run_ids_str" --model_paths "$model_paths_str" --project_root $project_root --models_dir $models_dir