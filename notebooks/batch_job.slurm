#!/bin/bash
#SBATCH --job-name=02Biii
#SBATCH --output 02Biii_%j.out
#SBATCH -e 02Biii_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=500G
#SBATCH --time=23:59:59

## User python environment
ENV_NAME="metastatic_potential"
CONDA_PATH="/nobackup/users/hmbaghda/Software/miniforge3/"

echo "start"
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate $ENV_NAME
python 02Biii_model_selection_joint.py
echo "complete"
