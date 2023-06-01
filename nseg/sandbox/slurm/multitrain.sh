#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=32
#SBATCH --time=7-0
#SBATCH --array=0-1

# srun echo ${SLURM_ARRAY_TASK_ID} $(nvidia-smi -L) >> ~/jobarrtest.log

/cajal/nvmescratch/users/mdraw/anaconda/condabin/conda activate nseg

srun python -c 'import numpy; print(numpy.__version__)' >> ~/jobarrtest.log
