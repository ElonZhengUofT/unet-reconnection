#!/bin/bash

docker run -v $(pwd):/work/unet-reconnection -w /work/unet-reconnection --user $(id -u):$(id -g) --shm-size 16g --gpus all -it unet-reconnection