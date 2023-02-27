#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G

. ./env.sh

module load pytorch/1.13
pip install matplotlib tqdm gif scikit-learn ptflops

plot.py -d $(pwd)/$1/$SLURM_JOB_NAME --gif
