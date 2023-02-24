#!/bin/bash

PROJECT="project_2004522"
FEATURE_SET=("em" "raw" "custom")
PREPROCESSING=("none" "normalized" "standardized")
KERNEL_SIZES=("1" "3" "5" "7" "9")

submit_training_job () {
    local f="$1"
    local k="$2"
    local p="$3"
    local job_name="${f}-${p}-${k}"
    local log_dir="slurm/train/%j-${job_name}.out"
    local feature_flags=""
    local preprocessing_flag=""

    if [ "$f" = "raw" ]; then
        feature_flags="--velocity,--rho"
    elif [ "$f" = "custom" ]; then
        feature_flags="--velocity,--rho,--anisotropy,--agyrotropy"
    fi

    if [ "$p" = "normalized" ]; then
        preprocessing_flag="--normalize"
    elif [ "$p" = "standardized" ]; then
        preprocessing_flag="--standardize"
    fi

    command=("sbatch" "--account" "$PROJECT" "--job-name" "$job_name" "--output" "$log_dir" "submit_training.sh" "-d" "$DIR" "-k" "$k")
    if [ -n "$feature_flags" ]; then
        command+=("-f" "$feature_flags")
    fi
    if [ -n "$preprocessing_flag" ]; then
        command+=("-p" "$preprocessing_flag")
    fi

    echo "${command[@]}"
    "${command[@]}"
}

submit_plotting_job () {
    local f="$1"
    local k="$2"
    local p="$3"
    local job_name="${f}-${p}-${k}"
    local log_dir="slurm/plot/%j-${job_name}.out"

    local command=("sbatch" "--account" "${PROJECT}" "--job-name" "${job_name}" "--output" "${log_dir}" "submit_plotting.sh" "$DIR")

    echo "${command[@]}"
    "${command[@]}"
}

while getopts "p:t:d:" opt; do
    case ${opt} in
        p )
            PROJECT="$OPTARG"
            ;;
        t )
            TASK="$OPTARG"
            ;;
        d )
            DIR="$OPTARG"
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            exit 1
            ;;
        : )
            echo "Option -$OPTARG requires an argument." 1>&2
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

if [ -z "$TASK" ]; then
    echo "Task not specified. Please specify either 'train' or 'plot' with the -t flag."
    exit 1
fi
if [ -z "$DIR" ]; then
    echo "Directory not specified."
    exit 1
fi

for f in "${FEATURE_SET[@]}"; do
    for p in "${PREPROCESSING[@]}"; do
        for k in "${KERNEL_SIZES[@]}"; do
            if [ "$TASK" = "train" ]; then
                submit_training_job "$f" "$k" "$p"
            fi
            if [ "$TASK" = "plot" ]; then
                submit_plotting_job "$f" "$k" "$p"
            fi
        done
    done
done
