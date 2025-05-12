#!/bin/bash
#SBATCH --job-name=resume_model_run
#SBATCH --partition=ls6prio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=resume_run_%A_%a.out
#SBATCH --error=resume_run_%A_%a.err
#SBATCH --exclude=gpu8a

# Define project, model directory and dataset paths
sweep_id="6te7vzul"
project_root="/home/ls6/hauptmann/ethikchat-experiment-argument-classification"
models_dir="${project_root}/experiments_outputs/${sweep_id}"
test_dataset_path="${project_root}/data/processed/with_context/dataset_split_in_distribution_from_v3/test"
corpus_dataset_path="${project_root}/data/processed/with_context/corpus_dataset_v3"

# Lists of length 20
run_ids=(
"dixffits"
"494m8hcc"
"9izanmce"
"9ry9pir3"
"kfaeabcz"
"4lc0lrg8"
"ly0so9wf"
"rrq5oask"
)
run_names=(
"autumn-sweep-64/"
"fiery-sweep-133/"
"iconic-sweep-129/"
"honest-sweep-51/"
"bright-sweep-55/"
"crisp-sweep-53/"
"good-sweep-141/"
"misunderstood-sweep-56/"
)

# pick out this task’s index
idx=$SLURM_ARRAY_TASK_ID
run_id=${run_ids[$idx]}
run_name=${run_names[$idx]}


# Export project_root to PYTHONPATH for the Python script
export PYTHONPATH="${project_root}:${PYTHONPATH}"

# Activate the virtual environment
source "${project_root}/venv/bin/activate"

# Run the python script
python resume_model_run.py --run_ids "${run_ids[@]}" --run_names "${run_names[@]}" --project_root $project_root --models_dir $models_dir --test_dataset_path $test_dataset_path --corpus_dataset_path $corpus_dataset_path

