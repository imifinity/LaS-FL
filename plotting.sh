#!/bin/bash
#SBATCH -D /users/adgs945/Individual_project_code_/
#SBATCH --job-name plotting                     # Job name
#SBATCH --partition=nodes                       # General-purpose GPU nodes
#SBATCH --nodes=1                               # Single node
#SBATCH --ntasks-per-node=1                     # Single task per node
#SBATCH --cpus-per-task=4                       # 4 CPU cores
#SBATCH --mem=16GB                              # 16GB RAM
#SBATCH --time=0:30:00                          # Set time limit
#SBATCH -e plot_jobs/%x_%A.e                     # Error logs
#SBATCH -o plot_jobs/%x_%A.o                     # Output logs

# Enable modules via Flight Centre
source /opt/flight/etc/setup.sh
flight env activate gridware
module purge

# Activate environment
module add compilers/gcc gnu
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
cd /users/adgs945/Individual_project_code_

# Run the Python training script
python3 plotting.py \
  --csv_dir "metrics" \
  --output_dir "final plots/TIN" \
  --algorithm las \
  --dataset TinyImageNet

echo "Finished array task $SLURM_ARRAY_TASK_ID at $(date)"