#!/bin/bash
#SBATCH --job-name=resume_model_run
#SBATCH --partition=ls6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=resume_run_%A_%a.out
#SBATCH --error=resume_run_%A_%a.err

# Define project, model directory and dataset paths
project_root="/home/stud/spiegel/ethikchat-experiment-argument-classification"
models_dir="${project_root}/models"
dataset_path=""

# Define lists for run_ids and model_paths
run_ids=(
)

model_paths=(
)

# Export project_root to PYTHONPATH for the Python script
export PYTHONPATH="${project_root}:${PYTHONPATH}"

# Activate the virtual environment
source "${project_root}/venv/bin/activate"

# Run the python script
python resume_model_run.py --run_ids "${run_ids[@]}" --models "${model_paths[@]}" --project_root $project_root --models_dir $models_dir --dataset_path $dataset_path