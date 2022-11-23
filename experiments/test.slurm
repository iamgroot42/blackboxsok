
#!/bin/bash

## Run command as:
## sbatch --array=0-1 --export ALL --job-name=testing this_file.slurm
# Generic format will be 0-N when there are N+1 jobs to be run

# Number of tasks (same as number of scripts)
#SBATCH --ntasks=1

# Allocation group
#SBATCH -A uvasrg

# Total memory requested on system
# Maximum 4 nodes per job allowed
# Maximum 32GB/core allowed
#SBATCH --mem=8G

# Partition of machine
#SBATCH -p gpu

# Which GPU (and how many) to request
# Prefer V100 over 2080
#SBATCH --gres=gpu:a100:1

# Request specific number of CPUs for the task
# Maximum 10 cores per job allowed
#SBATCH --cpus-per-task=4

# Time limit (go for max allowed: 3 days for GPU)
#SBATCH --time=3-00:00:00

# Output file paths (make sure this path exists in the folder where you run the script, or else you'll get an eror)
#SBATCH --output=log_folder/%x-%j.out
#SBATCH --error=log_folder/%x-%j.err

CONFIGS=(
    configs/image/transfer_methods/target/eps_4/inceptionv3/targeted_FGSM_transfer.json
    configs/image/transfer_methods/target/eps_4/inceptionv3/targeted_IFGSM_transfer.json
)

CUDA_VISIBLE_DEVICES=0 knocky python transferred_attack_experiment.py --config "${CONFIGS[SLURM_ARRAY_TASK_ID]}"
# 'knocky' above is my alias to knockknock and is useful for sending notifications when jobs finish

