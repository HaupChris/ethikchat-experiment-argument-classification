#!/bin/bash
#SBATCH --job-name=resume_model_run
#SBATCH --partition=ls6prio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=resume_run_%A_%a.out
#SBATCH --error=resume_run_%A_%a.err
# here: one task per run; with 20 runs this becomes 0-19
#SBATCH --array=0-19

# Define project, model directory and dataset paths
sweep_id="6te7vzul"
project_root="/home/ls6/hauptmann/ethikchat-experiment-argument-classification"
models_dir="${project_root}/experiments_outputs/${sweep_id}"
test_dataset_path="${project_root}/data/processed/with_context/dataset_split_in_distribution_from_v3/test"
corpus_dataset_path="${project_root}/data/processed/with_context/corpus_dataset_v3"

# Lists of length 20
run_ids=(
"nq1xf4g7"
"ly0so9wf"
"rrq5oask"
"3sn1ynic"
"jefhym4d"
"f2g9karn"
"f99yk1ld"
"jtqy48u0"
"zvtxkghe"
"494m8hcc"
"o11zgn5e"
"dixffits"
"u6zhl1pb"
"4lc0lrg8"
"9izanmce"
"9ry9pir3"
"68xtlm4l"
"ac6jg8nt"
"kfaeabcz"
"72oloxb7"
"zyouwaia"
)
run_names=(
"lilac-sweep-65/"
"good-sweep-141/"
"misunderstood-sweep-56/"
"eager-sweep-52/"
"major-sweep-60/"
"whole-sweep-57/"
"eternal-sweep-132/"
"zany-sweep-49/"
"feasible-sweep-137/"
"fiery-sweep-133/"
"royal-sweep-61/"
"autumn-sweep-64/"
"fluent-sweep-136/"
"crisp-sweep-53/"
"iconic-sweep-129/"
"honest-sweep-51/"
"colorful-sweep-140/"
"atomic-sweep-50/"
"bright-sweep-55/"
"spring-sweep-54/"
"pious-sweep-125/"
)

# pick out this task’s index
idx=$SLURM_ARRAY_TASK_ID
run_id=${run_ids[$idx]}
run_name=${run_names[$idx]}

# Export and activate
export PYTHONPATH="${project_root}:${PYTHONPATH}"
source "${project_root}/venv/bin/activate"

# Run only one
python resume_model_run.py \
  --run_ids "$run_id" \
  --run_names "$run_name" \
  --project_root "$project_root" \
  --models_dir "$models_dir" \
  --test_dataset_path "$test_dataset_path" \
  --corpus_dataset_path "$corpus_dataset_path"
