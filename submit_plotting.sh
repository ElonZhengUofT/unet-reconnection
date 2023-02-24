#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

. ./env.sh

module load pytorch/1.13
pip install matplotlib tqdm gif scikit-learn ptflops

plot.py -d $(pwd)/$1/$SLURM_JOB_NAME --gif
