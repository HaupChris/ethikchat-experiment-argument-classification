#!/bin/bash
#SBATCH --job-name=resume_model_run
#SBATCH --partition=ls6prio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=resume_run_%A_%a.out
#SBATCH --error=resume_run_%A_%a.err

# Define project, model directory and dataset paths
sweep_id="sl3evhzh"
project_root="/home/ls6/hauptmann/ethikchat-experiment-argument-classification"
models_dir="${project_root}/experiments_outputs/${sweep_id}"
dataset_path="${project_root}/data/processed/dataset_split_in_distribution/test"

# Define lists for run_ids and model_paths
run_ids=(
"r0zoth3d"
"njps9awi"
"2iodyojr"
"6srwgafk"
"7sv6qcje"
"pylazvbq"
"4umjmvdn"
"oip5hwwt"
"a0rpjtlf"
"mjxq5f2i"
)

run_names=(
"gentle-sweep-9/"
"colorful-sweep-6/"
"copper-sweep-3/"
"magic-sweep-4/"
"stilted-sweep-24/"
"polar-sweep-146/"
"feasible-sweep-165/"
"sweet-sweep-124/"
"polar-sweep-136/"
"celestial-sweep-25/"
)

# Export project_root to PYTHONPATH for the Python script
export PYTHONPATH="${project_root}:${PYTHONPATH}"

# Activate the virtual environment
source "${project_root}/venv/bin/activate"

# Run the python script
python resume_model_run.py --run_ids "${run_ids[@]}" --run_names "${run_names[@]}" --project_root $project_root --models_dir $models_dir --dataset_path $dataset_path