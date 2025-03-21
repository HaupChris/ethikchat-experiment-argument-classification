#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --array=1-4           # 4 parallel tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1          # 1 GPU per task
#SBATCH --output=sweep_agent_%A_%a.out
#SBATCH --error=sweep_agent_%A_%a.err


project_root_dir="/home/ls6/hauptmann/ethikchat-experiment-argument-classification"

# add project root directory to python path
export PYTHONPATH="${project_root_dir}:${PYTHONPATH}"

# Load your environment
source /home/ls6/hauptmann/ethikchat-experiment-argument-classification/venv/bin/activate



# Make sure you already created the sweep and have the SWEEP_ID. Example:
# wandb sweep sweep_config.yaml
# which prints e.g. "Run wandb agent <USERNAME>/<PROJECT>/<SWEEP_ID>"

SWEEP_ID="christian-hauptmann-uniwue/argument-classification/3unf2cdm"  # Replace with your actual sweep ID

echo "Starting W&B agent on task $SLURM_ARRAY_TASK_ID..."
wandb agent $SWEEP_ID
echo "Agent on task $SLURM_ARRAY_TASK_ID finished."
