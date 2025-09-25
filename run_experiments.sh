#!/bin/bash
#SBATCH -D /users/adgs945/Individual_project_code_/
#SBATCH --job-name gen_exp                           # Job name
#SBATCH --partition=gengpu                           # GPU nodes
#SBATCH --nodes=1                                    # Single node
#SBATCH --ntasks-per-node=1                          # Single task per node
#SBATCH --cpus-per-task=4                            # 4 CPU cores
#SBATCH --mem=16GB                                   # 16GB RAM
#SBATCH --time=5:00:00                               # Time limit
#SBATCH --gres=gpu:1                                 # 1 GPU
#SBATCH --array=1-12                                 # Number of experiments to run
#SBATCH -e slurm_jobs/%x_%A_%a.e                     # Error logs
#SBATCH -o slurm_jobs/%x_%A_%a.o                     # Output logs

# Enable modules via Flight Centre
source /opt/flight/etc/setup.sh
flight env activate gridware
module purge

# Activate environment
module add compilers/gcc gnu
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
cd /users/adgs945/Individual_project_code_

# Read the line corresponding to this array task
EXPERIMENT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments.txt)

# Run Python script with these parameters
DATASET=$(echo $EXPERIMENT | cut -d ' ' -f1)
ALGORITHM=$(echo $EXPERIMENT | cut -d ' ' -f2)
DIR=$(echo $EXPERIMENT | cut -d ' ' -f3)

python3 fedalg.py \
  --dataset $DATASET \
  --algorithm $ALGORITHM \
  --dirichlet $DIR \
  --n_epochs 50 \
  --seed 42

echo "Finished task $SLURM_ARRAY_TASK_ID at $(date)"