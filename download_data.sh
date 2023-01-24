#!/bin/bash -l

#SBATCH -o ./zfdl-out.%j
#SBATCH -e ./zfdl-err.%j
#SBATCH -D ./
#SBATCH -J ZEBRAFINCH_DL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-0

module purge
module load anaconda/3/2021.11

conda activate lsd2

srun /cajal/nvmescratch/users/mdraw/anaconda/envs/lsd2/bin/python /u/mdraw/mlsd/download_data.py
