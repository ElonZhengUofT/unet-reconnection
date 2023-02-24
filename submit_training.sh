#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G


print_usage() {
  printf "Usage: -f feature flags, -p preprocessing flags, -k kernel size"
}

while getopts "f:p:k:d:" arg; do
  case $arg in
    f) feature_flags=$(echo "$OPTARG" | tr ',' ' ');;
    p) preprocessing_flag=$OPTARG;;
    k) kernel_size=$OPTARG;;
    d) directory=$OPTARG;;
    *) print_usage
       exit 1;;
  esac
done

. ./env.sh

module load pytorch/1.13
pip install matplotlib tqdm gif scikit-learn ptflops

train.py -i $(pwd)/sample/data -o $(pwd)/$directory/$SLURM_JOB_NAME --data-splits 0.6 0.2 0.2 --epochs 5 $feature_flags $preprocessing_flag --kernel-size $kernel_size
