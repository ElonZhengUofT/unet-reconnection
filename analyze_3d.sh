#!/bin/bash

. ./env.sh

# Loop through every subfolder of the ./results folder
for SUBFOLDER in ./results/*/; do
    SUBFOLDER=${SUBFOLDER%*/}
    SUBFOLDER=${SUBFOLDER##*/}

    # Check if optimizer_best_epoch.pt exists in the folder
    if [ ! -f "results/$SUBFOLDER/optimizer_best_epoch.pt" ]; then
        echo "Skipping $SUBFOLDER: optimizer_best_epoch.pt not found."
        continue
    fi

    # Run predict.py with the appropriate arguments
    predict.py -m "results/$SUBFOLDER" -i "3d-data" -o "3d-results/$SUBFOLDER/test"

    # Run plot.py with the appropriate arguments
    plot.py -d "3d-results/$SUBFOLDER" -m "results/$SUBFOLDER"
done